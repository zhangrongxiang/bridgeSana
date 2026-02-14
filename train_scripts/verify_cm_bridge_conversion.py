#!/usr/bin/env python
"""verify_cm_bridge_conversion.py

Two verification modes are supported:

1) timestep_only (DEFAULT):
    Keep the inference loop identical to batch_inference_visualize.py (ViBT stepping
    in bridge/linear space). The only change is that the timestep sequence is built
    from a TrigFlow time grid via the CM time mapping, i.e. t_trig -> tau -> r -> t_bridge.
    No state projection and no velocity conversion happen in this mode.

2) full_cm:
    Run a full geometric conversion loop: sample in TrigFlow geometry, project state
    to linear geometry for the Bridge model, convert velocity back to TrigFlow, and
    step in TrigFlow time. This is a stricter but different experiment than "only
    change the timesteps".
"""

import argparse
import sys
import os
from pathlib import Path
import contextlib
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import numpy as np
from tqdm import tqdm
import json

import diffusers

from diffusers import SanaPipeline, AutoencoderDC, SanaTransformer2DModel
from transformers import AutoTokenizer, Gemma2Model
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict

from cm_bridge_converter import CMBridgeConverter


def parse_args():
    parser = argparse.ArgumentParser(description="Verify CM-Bridge conversion")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to base Sana model",
    )
    parser.add_argument(
        "--lora_path",
        type=str,
        required=True,
        help="Path to trained Bridge LoRA checkpoint",
    )
    parser.add_argument(
        "--cond_proj_path",
        type=str,
        default=None,
        help="Path to concat conditioning adapter weights",
    )
    parser.add_argument(
        "--source_image",
        type=str,
        required=True,
        help="Path to source image for testing",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Convert the style to 3D Chibi Style",
        help="Text prompt",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./cm_bridge_verification",
        help="Output directory for results",
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=28,
        help="Number of sampling steps",
    )
    parser.add_argument(
        "--conversion_mode",
        type=str,
        default="timestep_only",
        choices=["timestep_only", "full_cm"],
        help="Verification mode. 'timestep_only' keeps the ViBT inference loop unchanged and only remaps timesteps from TrigFlow; 'full_cm' runs full state/velocity conversion.",
    )
    parser.add_argument(
        "--scheduler_mode",
        type=str,
        default="vibt",
        choices=["model", "vibt"],
        help="Scheduler timesteps source: 'model' uses checkpoint scheduler; 'vibt' wraps with ViBT (mirrors batch_inference_visualize).",
    )
    parser.add_argument(
        "--vibt_noise_scale",
        type=float,
        default=0.2,
        help="ViBT scheduler noise scale (only used when --scheduler_mode=vibt)",
    )
    parser.add_argument(
        "--vibt_shift_gamma",
        type=float,
        default=5.0,
        help="ViBT scheduler flow_shift (only used when --scheduler_mode=vibt)",
    )
    parser.add_argument(
        "--trig_t_start",
        type=float,
        default=0.0,
        help="TrigFlow start time (radians) for building the timestep grid in timestep_only mode.",
    )
    parser.add_argument(
        "--trig_t_end",
        type=float,
        default=(math.pi / 2),
        help="TrigFlow end time (radians) for building the timestep grid in timestep_only mode.",
    )
    parser.add_argument(
        "--sigma_data",
        type=float,
        default=1.0,
        help="Latent space scale constant",
    )
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=128,
        help="LoRA rank",
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=128,
        help="LoRA alpha",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--booting_noise_scale",
        type=float,
        default=0.1,
        help="Booting noise scale for concat conditioning",
    )
    return parser.parse_args()


def encode_image(vae, image_path, device, resolution=1024):
    """Encode image to latent space"""
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(resolution),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        encoded = vae.encode(image_tensor)
        if hasattr(encoded, "latent_dist"):
            latents = encoded.latent_dist.mode()
        elif hasattr(encoded, "latents"):
            latents = encoded.latents
        elif hasattr(encoded, "latent"):
            latents = encoded.latent
        else:
            latents = encoded

        scaling_factor = getattr(getattr(vae, "config", None), "scaling_factor", None)
        if scaling_factor is not None:
            latents = latents * scaling_factor

    return latents, image


def decode_latents(vae, latents):
    """Decode latents to image"""
    scaling_factor = getattr(getattr(vae, "config", None), "scaling_factor", None)
    if scaling_factor is not None:
        latents = latents / scaling_factor
    decoded = vae.decode(latents)
    if hasattr(decoded, "sample"):
        return decoded.sample
    return decoded


def _get_prev_sample(step_out):
    """Match the scheduler.step output handling used by batch_inference_visualize.py."""
    if hasattr(step_out, "prev_sample"):
        return step_out.prev_sample
    if isinstance(step_out, dict) and "prev_sample" in step_out:
        return step_out["prev_sample"]
    if isinstance(step_out, (tuple, list)) and len(step_out) > 0:
        return step_out[0]
    raise TypeError(f"Unsupported scheduler.step output type: {type(step_out)}")


def _build_bridge_timesteps_from_trigflow(
    converter: CMBridgeConverter,
    num_steps: int,
    device,
    trig_t_start: float,
    trig_t_end: float,
):
    """Build a bridge-timestep sequence (≈[1000..0]) from a TrigFlow time grid.

    Mapping:
      t_trig -> tau = T(t_trig)
      r = 1 - tau
      t_bridge = 1000 * r

    The resulting sequence is decreasing when t_trig increases.
    """
    t_trig_seq = torch.linspace(
        trig_t_start,
        trig_t_end,
        steps=num_steps,
        device=device,
        dtype=torch.float32,
    )
    tau_seq = converter.time_trig_to_linear(t_trig_seq).clamp(0.0, 1.0)
    r_seq = (1.0 - tau_seq).clamp(0.0, 1.0)
    t_bridge_seq = (r_seq * 1000.0).clamp(0.0, 1000.0)

    # Ensure strictly descending/monotonic non-increasing as expected by schedulers.
    # If numerical jitter breaks monotonicity, sort descending.
    if (t_bridge_seq[1:] > t_bridge_seq[:-1]).any():
        t_bridge_seq, _ = torch.sort(t_bridge_seq, descending=True)

    return t_trig_seq, tau_seq, t_bridge_seq


@torch.no_grad()
def sample_with_timestep_only_conversion(
    transformer,
    cond_proj,
    converter,
    scheduler,
    source_latents,
    cond_latents,
    encoder_hidden_states,
    encoder_attention_mask,
    num_steps,
    device,
    trig_t_start: float,
    trig_t_end: float,
):
    """Inference loop identical to batch_inference_visualize.py.

    Only difference: the timestep sequence comes from a TrigFlow time grid via the
    CM time mapping (no state projection, no velocity conversion).
    """
    t_trig_seq, tau_seq, t_bridge_seq = _build_bridge_timesteps_from_trigflow(
        converter=converter,
        num_steps=num_steps,
        device=device,
        trig_t_start=trig_t_start,
        trig_t_end=trig_t_end,
    )

    # Initialize scheduler internal state, then override timesteps for stepping.
    scheduler.set_timesteps(num_steps, device=device)
    try:
        scheduler.timesteps = t_bridge_seq
    except Exception:
        # Some schedulers might expose timesteps as a property; in that case we still
        # run the loop with our custom t values but delta_t logic may not match.
        pass

    metrics = []

    latents = source_latents.clone()

    autocast_ctx = (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if str(device).startswith("cuda")
        else contextlib.nullcontext()
    )

    print(f"\n{'='*60}")
    print("Timestep-only verification (no velocity conversion)")
    print(f"{'='*60}")
    print(f"TrigFlow t range: [{float(t_trig_seq[0].item()):.6f}, {float(t_trig_seq[-1].item()):.6f}]")
    print(f"Linear tau range: [{float(tau_seq[0].item()):.6f}, {float(tau_seq[-1].item()):.6f}]")
    print(f"Bridge timestep range: [{float(t_bridge_seq[-1].item()):.3f}, {float(t_bridge_seq[0].item()):.3f}]")
    print()

    with autocast_ctx:
        for step_idx, t_bridge in enumerate(t_bridge_seq):
            model_input = cond_proj(torch.cat([latents, cond_latents], dim=1))

            # Match reference inference: timestep is a float tensor, value taken from the loop scalar.
            t_scalar = t_bridge
            if not torch.is_tensor(t_scalar):
                t_scalar = torch.tensor(t_scalar, device=device)
            timesteps = torch.full(
                (model_input.shape[0],),
                float(t_scalar.item()),
                device=device,
                dtype=torch.float32,
            )

            model_out = transformer(
                hidden_states=model_input,
                timestep=timesteps,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
            ).sample

            step_out = scheduler.step(model_out, t_scalar, latents, return_dict=True)
            latents = _get_prev_sample(step_out)

            metrics.append(
                {
                    "mode": "timestep_only",
                    "step": int(step_idx),
                    "t_trig": float(t_trig_seq[step_idx].item()),
                    "tau_linear": float(tau_seq[step_idx].item()),
                    "t_bridge": float(t_bridge.item()) if torch.is_tensor(t_bridge) else float(t_bridge),
                }
            )

            if step_idx % 5 == 0 or step_idx == num_steps - 1:
                print(
                    f"Step {step_idx:3d}/{num_steps}: "
                    f"t_trig={float(t_trig_seq[step_idx].item()):.4f}, "
                    f"tau={float(tau_seq[step_idx].item()):.4f}, "
                    f"t_bridge={float(t_bridge_seq[step_idx].item()):.2f}"
                )

    return latents, metrics


@torch.no_grad()
def encode_prompts(tokenizer, text_encoder, prompt, device):
    """Encode text prompt"""
    text_inputs = tokenizer(
        [prompt],
        padding="max_length",
        max_length=300,
        truncation=True,
        return_tensors="pt",
    ).to(device)
    hs = text_encoder(
        input_ids=text_inputs.input_ids,
        attention_mask=text_inputs.attention_mask,
    )[0]
    return hs, text_inputs.attention_mask


def load_cond_proj(vae, device, cond_proj_path, dtype=torch.bfloat16):
    """Load concat conditioning adapter"""
    latent_channels = getattr(getattr(vae, "config", None), "latent_channels", 4)
    cond_proj = nn.Conv2d(latent_channels * 2, latent_channels, kernel_size=1, bias=True)
    nn.init.zeros_(cond_proj.weight)
    nn.init.zeros_(cond_proj.bias)
    with torch.no_grad():
        for c in range(latent_channels):
            cond_proj.weight[c, c, 0, 0] = 1.0

    if cond_proj_path and Path(cond_proj_path).exists():
        state = torch.load(cond_proj_path, map_location="cpu")
        cond_proj.load_state_dict(state)
        print(f"✅ Loaded concat adapter from {cond_proj_path}")
    else:
        print("⚠️  Using default initialized concat adapter")

    return cond_proj.to(device=device, dtype=dtype)


def load_scheduler_from_model(model_path: str):
    """Load the exact scheduler class specified by the model checkpoint.

    This matches the logic in batch_inference_visualize.py to avoid timestep mismatch.
    """
    scheduler_cfg = Path(model_path) / "scheduler" / "scheduler_config.json"
    if scheduler_cfg.exists():
        with open(scheduler_cfg, "r") as f:
            cfg = json.load(f)
        class_name = cfg.get("_class_name")
        if class_name and hasattr(diffusers, class_name):
            scheduler_cls = getattr(diffusers, class_name)
            return scheduler_cls.from_pretrained(model_path, subfolder="scheduler")

    base_pipe = SanaPipeline.from_pretrained(model_path, torch_dtype=torch.bfloat16)
    scheduler = base_pipe.scheduler
    del base_pipe
    return scheduler


def sample_with_full_cm_conversion_verification(
    transformer,
    cond_proj,
    converter,
    scheduler,
    source_latents,
    cond_latents,
    encoder_hidden_states,
    encoder_attention_mask,
    num_steps,
    device,
):
    """
    Sample using TrigFlow geometry with CM-Bridge conversion.

    At each step:
    1. Current state is in TrigFlow: x_t = cos(t)x0 + sin(t)x1
    2. Convert time and state to Linear Bridge geometry
    3. Get Bridge model prediction
    4. Convert velocity back to TrigFlow
    5. Update state using TrigFlow dynamics

    Returns trajectory and conversion metrics.
    """
    # Initialize trajectory storage
    trajectory = []
    metrics = []

    # IMPORTANT: use the same timestep schedule as the reference inference code.
    # In batch_inference_visualize.py, inference uses scheduler.timesteps in Bridge time units (≈[1000..0]).
    # Here we reuse scheduler.timesteps and map it to TrigFlow time via:
    #   r = t_bridge/1000  (reversed bridge time, r=1 at source, r=0 at target)
    #   tau = 1 - r        (forward linear time, tau=0 at source, tau=1 at target)
    #   t_trig = T^{-1}(tau)
    scheduler.set_timesteps(num_steps, device=device)
    t_bridge_seq = scheduler.timesteps
    if not torch.is_tensor(t_bridge_seq):
        t_bridge_seq = torch.tensor(t_bridge_seq, device=device)
    t_bridge_seq = t_bridge_seq.to(device=device)

    # Create a "next" timestep list; last step goes to 0.
    t_bridge_next_seq = torch.cat([t_bridge_seq[1:], torch.zeros_like(t_bridge_seq[:1])], dim=0)

    # Start from source
    x_t = source_latents.clone()

    print(f"\n{'='*60}")
    print("Full CM conversion loop (state/velocity conversion)")
    print(f"{'='*60}")
    print(f"Number of steps: {num_steps}")
    # t_trig endpoints depend on scheduler; print a quick summary
    r0 = (t_bridge_seq[0].float() / 1000.0).clamp(0, 1)
    r1 = (t_bridge_seq[-1].float() / 1000.0).clamp(0, 1)
    tau0 = (1 - r0).item()
    tau1 = (1 - r1).item()
    print(f"Bridge timestep range: [{t_bridge_seq[-1].item():.3f}, {t_bridge_seq[0].item():.3f}]")
    print(f"Linear tau range: [{tau0:.4f}, {tau1:.4f}]")
    print()

    # Setup autocast context for mixed precision (match batch_inference_visualize.py)
    autocast_ctx = (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if str(device).startswith("cuda")
        else contextlib.nullcontext()
    )

    with autocast_ctx:
        for step_idx in range(num_steps):
            # Bridge time (diffusers/Sana convention): pass this directly to the transformer, same as reference code.
            t_bridge = t_bridge_seq[step_idx]
            t_bridge_next = t_bridge_next_seq[step_idx]

            # Normalize to r in [0,1] (reversed bridge time)
            r = (t_bridge.float() / 1000.0).clamp(0, 1)
            r_next = (t_bridge_next.float() / 1000.0).clamp(0, 1)

            # Forward linear time tau in [0,1]
            tau = 1.0 - r
            tau_next = 1.0 - r_next

            # Convert to TrigFlow time
            t_trig = converter.time_linear_to_trig(tau)
            t_trig_next = converter.time_linear_to_trig(tau_next)
            dt_trig = t_trig_next - t_trig

            # Ensure scalar time is converted to batch tensor
            t_batch_trig = torch.full((x_t.shape[0],), float(t_trig.item()), device=device)

            # === Step 2: Project state to Linear geometry ===
            h_in = converter.project_state_trig_to_linear(x_t, t_batch_trig)

            # === Step 3: Get Bridge model prediction ===
            # Apply concat conditioning
            model_input = cond_proj(torch.cat([h_in, cond_latents], dim=1))

            # Match reference inference: timestep is a float tensor, value taken from scheduler.timesteps.
            if not torch.is_tensor(t_bridge):
                t_bridge = torch.tensor(t_bridge, device=device)
            t_bridge_tensor = torch.full(
                (model_input.shape[0],),
                float(t_bridge.item()),
                device=device,
                dtype=torch.float32,
            )

            # Get velocity prediction from Bridge model
            with torch.no_grad():
                v_bridge = transformer(
                    hidden_states=model_input,
                    timestep=t_bridge_tensor,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                ).sample

            # === Step 4: Convert velocity from Linear to TrigFlow ===
            # Bridge model is trained on REVERSED time r (r=1 at source, r=0 at target).
            # Converter expects v_linear = dx_lin/dtau where tau is FORWARD linear time (tau=0 at source, tau=1 at target).
            # Since tau = 1 - r, we have: dx/dtau = -dx/dr.
            v_linear_tau = -v_bridge
            u_trig = converter.convert_velocity_linear_to_trig(v_linear_tau, x_t, t_batch_trig)

            # === Step 5: Verification - convert back ===
            v_linear_reconstructed = converter.convert_velocity_trig_to_linear(u_trig, x_t, t_batch_trig)
            conversion_error = F.mse_loss(v_linear_reconstructed, v_linear_tau).item()

            cos_sim = F.cosine_similarity(
                v_linear_tau.reshape(v_linear_tau.shape[0], -1),
                v_linear_reconstructed.reshape(v_linear_reconstructed.shape[0], -1),
                dim=1
            ).mean().item()

            # === Step 6: Update state using TrigFlow dynamics ===
            # Euler step in TrigFlow time: x_{t+dt} = x_t + u_trig * dt_trig
            x_t = x_t + u_trig * dt_trig

            # Store metrics
            step_metrics = {
                'step': step_idx,
                't_trig': float(t_trig.item()),
                'tau_linear': float(tau.item()),
                't_bridge': float(t_bridge.item()),
                'conversion_error': conversion_error,
                'cosine_similarity': cos_sim,
                'velocity_norm': torch.norm(u_trig).item(),
            }
            metrics.append(step_metrics)

            # Store trajectory
            trajectory.append(x_t.clone())

            if step_idx % 5 == 0 or step_idx == num_steps - 1:
                print(f"Step {step_idx:3d}/{num_steps}: "
                      f"t_trig={float(t_trig.item()):.4f}, tau={float(tau.item()):.4f}, "
                      f"conv_err={conversion_error:.2e}, cos_sim={cos_sim:.6f}")

    return x_t, trajectory, metrics


def main():
    args = parse_args()

    print("="*60)
    print("CM-Bridge Conversion Verification")
    print("="*60)

    # Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load models
    print(f"\n🔄 Loading models from {args.model_path}")

    vae = AutoencoderDC.from_pretrained(
        args.model_path, subfolder="vae", torch_dtype=torch.float32
    ).to(device)

    text_encoder = Gemma2Model.from_pretrained(
        args.model_path, subfolder="text_encoder", torch_dtype=torch.bfloat16
    ).to(device)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, subfolder="tokenizer"
    )

    transformer = SanaTransformer2DModel.from_pretrained(
        args.model_path, subfolder="transformer", torch_dtype=torch.bfloat16
    )

    # Load LoRA
    print(f"🔄 Loading LoRA from {args.lora_path}")
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
    )
    transformer = get_peft_model(transformer, lora_config)
    lora_weights = torch.load(args.lora_path, map_location="cpu")
    set_peft_model_state_dict(transformer, lora_weights)
    transformer = transformer.to(device)
    transformer.eval()

    # Load concat adapter
    cond_proj_path = args.cond_proj_path
    if cond_proj_path is None:
        # Auto-detect
        cand = Path(args.lora_path).resolve().parent / "cond_proj_weights.bin"
        if cand.exists():
            cond_proj_path = str(cand)

    cond_proj = load_cond_proj(vae, device, cond_proj_path)
    cond_proj.eval()

    # Load scheduler (for timesteps alignment)
    scheduler = load_scheduler_from_model(args.model_path)
    if args.scheduler_mode == "vibt":
        try:
            from diffusers.schedulers import UniPCMultistepScheduler
            from vibt.scheduler import ViBTScheduler

            if isinstance(scheduler, UniPCMultistepScheduler):
                scheduler = ViBTScheduler.from_scheduler(
                    scheduler,
                    noise_scale=args.vibt_noise_scale,
                    shift_gamma=args.vibt_shift_gamma,
                )
            else:
                base_unipc = UniPCMultistepScheduler.from_config(scheduler.config)
                scheduler = ViBTScheduler.from_scheduler(
                    base_unipc,
                    noise_scale=args.vibt_noise_scale,
                    shift_gamma=args.vibt_shift_gamma,
                )
            print(f"✅ Using ViBT scheduler timesteps: {type(scheduler).__name__}")
        except Exception as e:
            print(f"⚠️  Failed to enable ViBT scheduler timesteps, falling back to model scheduler: {e}")
    else:
        print(f"✅ Using model scheduler timesteps: {type(scheduler).__name__}")

    # Create converter
    print(f"\n🔧 Creating CM-Bridge converter (sigma_data={args.sigma_data})")
    converter = CMBridgeConverter(sigma_data=args.sigma_data)

    # Test geometric transformation
    print("\n" + "="*60)
    print("Testing Geometric Transformation")
    print("="*60)

    # Create dummy latents for testing
    dummy_x0 = torch.randn(1, 4, 32, 32, device=device)
    dummy_x1 = torch.randn(1, 4, 32, 32, device=device)
    test_times = torch.linspace(0.1, 1.5, 10, device=device)

    print("\nTime mapping verification:")
    for t in test_times[:5]:
        tau = converter.time_trig_to_linear(t)
        t_recon = converter.time_linear_to_trig(tau)
        error = abs(t.item() - t_recon.item())
        print(f"  t={t:.4f} → tau={tau:.4f} → t'={t_recon:.4f}, error={error:.2e}")

    print("\nVector field transformation verification:")
    verification_results = converter.verify_transformation(dummy_x0, dummy_x1, test_times[5])
    for key, value in verification_results.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.6e}")

    # Load source image and encode
    print(f"\n🖼️  Loading source image: {args.source_image}")
    source_latents, source_img = encode_image(vae, args.source_image, device)

    # Apply booting noise if specified
    cond_latents = source_latents
    if args.booting_noise_scale > 0:
        noise = torch.randn_like(source_latents)
        cond_latents = source_latents + args.booting_noise_scale * noise
        print(f"  Applied booting noise (scale={args.booting_noise_scale})")

    # Encode prompt
    print(f"💬 Encoding prompt: {args.prompt}")
    encoder_hidden_states, encoder_attention_mask = encode_prompts(
        tokenizer, text_encoder, args.prompt, device
    )

    # Run verification / sampling
    print("\n" + "="*60)
    if args.conversion_mode == "timestep_only":
        print("Running timestep-only verification")
    else:
        print("Running full CM conversion verification")
    print("="*60)

    trajectory = None
    if args.conversion_mode == "timestep_only":
        final_latents, metrics = sample_with_timestep_only_conversion(
            transformer=transformer,
            cond_proj=cond_proj,
            converter=converter,
            scheduler=scheduler,
            source_latents=source_latents,
            cond_latents=cond_latents,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            num_steps=args.num_steps,
            device=device,
            trig_t_start=args.trig_t_start,
            trig_t_end=args.trig_t_end,
        )
    else:
        final_latents, trajectory, metrics = sample_with_full_cm_conversion_verification(
            transformer=transformer,
            cond_proj=cond_proj,
            converter=converter,
            scheduler=scheduler,
            source_latents=source_latents,
            cond_latents=cond_latents,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            num_steps=args.num_steps,
            device=device,
        )

    # Decode final result
    print(f"\n🎨 Decoding final result...")
    final_image = decode_latents(vae, final_latents)
    final_image = torch.clamp((final_image + 1) / 2, 0, 1)
    final_image = (final_image[0].detach().permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    final_image = Image.fromarray(final_image)

    # Save results
    print(f"\n💾 Saving results to {output_dir}")

    # Save images
    source_img.save(output_dir / "source.png")
    final_image.save(output_dir / "result.png")

    # Create comparison
    comparison = Image.new('RGB', (source_img.width * 2, source_img.height))
    comparison.paste(source_img, (0, 0))
    comparison.paste(final_image, (source_img.width, 0))
    comparison.save(output_dir / "comparison.png")

    # Save metrics
    metrics_file = output_dir / "conversion_metrics.json"
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2)

    # Print summary
    print("\n" + "=" * 60)
    print("Verification Summary")
    print("=" * 60)

    if args.conversion_mode == "timestep_only":
        t_bridge_vals = [m["t_bridge"] for m in metrics]
        monotonic = all(t_bridge_vals[i + 1] <= t_bridge_vals[i] + 1e-6 for i in range(len(t_bridge_vals) - 1))
        print("\nTimestep mapping:")
        print(f"  Steps: {len(metrics)}")
        print(f"  Bridge range: [{min(t_bridge_vals):.3f}, {max(t_bridge_vals):.3f}]")
        print(f"  Monotonic non-increasing: {'YES' if monotonic else 'NO'}")
        print("\nNote: timestep_only mode does not do velocity/state conversion; it only tests the time-grid remapping with the original ViBT inference loop.")
    else:
        avg_conversion_error = np.mean([m["conversion_error"] for m in metrics])
        avg_cos_sim = np.mean([m["cosine_similarity"] for m in metrics])
        max_conversion_error = np.max([m["conversion_error"] for m in metrics])
        min_cos_sim = np.min([m["cosine_similarity"] for m in metrics])

        print("\nConversion Quality:")
        print(f"  Average conversion error: {avg_conversion_error:.6e}")
        print(f"  Maximum conversion error: {max_conversion_error:.6e}")
        print(f"  Average cosine similarity: {avg_cos_sim:.8f}")
        print(f"  Minimum cosine similarity: {min_cos_sim:.8f}")

        # Check if conversion is lossless (within numerical precision)
        is_lossless = max_conversion_error < 1e-5 and min_cos_sim > 0.9999

        print(f"\n{'✅' if is_lossless else '⚠️'} Conversion is {'LOSSLESS' if is_lossless else 'LOSSY'}")
        print("  (Threshold: error < 1e-5, cosine similarity > 0.9999)")

    print(f"\nResults saved to: {output_dir}")
    print(f"  - source.png: Original image")
    print(f"  - result.png: Converted result")
    print(f"  - comparison.png: Side-by-side")
    print(f"  - conversion_metrics.json: Detailed metrics")

    print("\n" + "="*60)
    print("✅ Verification complete!")
    print("="*60)


if __name__ == "__main__":
    main()
