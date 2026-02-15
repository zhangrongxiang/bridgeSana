#!/usr/bin/env python
"""train_bridge_scm_ladd_distill_lora_sana.py

Implements the algorithm described in new-cm-bridge.md (sCM + LADD) for
Sana Bridge models trained with concat_text conditioning.

Key constraints (project-specific):
- Bridge model time convention uses reversed time r in (0,1) and is passed to Sana as
  a 0..1000 timestep scalar (decreasing schedule 999->0 during inference).
- ODE bridge: per-step noise is disabled (noise_scale=0). Any randomness is injected
  as a one-time perturbation of the initial x0 latents (optional).

This script:
- Loads a frozen teacher Bridge model (base Sana + LoRA + optional cond_proj).
- Trains a student LoRA (and optional cond_proj) using:
  * sCM loss in TrigFlow time parameterization with CM-Bridge reference field u_hat.
  * LADD adversarial loss using a discriminator over teacher features.

The implementation adapts the document formulas to the 0..1000 reversed-time convention:
  t_trig -> tau = sin(t)/(sin(t)+cos(t))
  r = 1 - tau
  t_bridge = 1000 * r
  teacher predicts v_bridge = dx/dr; forward linear velocity is v_tau = dx/dtau = -v_bridge.

"""

import argparse
import contextlib
import json
import logging
import math
import os
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers import AutoencoderDC, SanaPipeline, SanaTransformer2DModel
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, is_wandb_available
from peft import LoraConfig, get_peft_model

try:
    # peft>=0.7
    from peft import set_peft_model_state_dict
except Exception:  # pragma: no cover
    try:
        from peft.utils import set_peft_model_state_dict
    except Exception:
        set_peft_model_state_dict = None

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import AutoTokenizer, Gemma2Model

from cm_bridge_converter import CMBridgeConverter

if is_wandb_available():
    import wandb

check_min_version("0.32.0.dev0")

logger = get_logger(__name__)


class BridgeDataset(Dataset):
    """Dataset for Bridge training with source-target image pairs."""

    def __init__(self, data_dir: str, resolution: int = 1024):
        self.data_dir = Path(data_dir)
        self.resolution = resolution

        jsonl_path = self.data_dir / "train_teacher_synth.jsonl"
        self.data = []
        with open(jsonl_path, "r") as f:
            for line in f:
                self.data.append(json.loads(line))

        self.transform = transforms.Compose(
            [
                transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(resolution),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        item = self.data[idx]
        src_rel = item["src"]
        tar_rel = item["tar"]

        data_dir_name = self.data_dir.name
        if src_rel.startswith(data_dir_name + "/"):
            src_rel = src_rel[len(data_dir_name) + 1 :]
        if tar_rel.startswith(data_dir_name + "/"):
            tar_rel = tar_rel[len(data_dir_name) + 1 :]

        src_path = self.data_dir / src_rel
        tar_path = self.data_dir / tar_rel

        src_image = Image.open(src_path).convert("RGB")
        tar_image = Image.open(tar_path).convert("RGB")

        src_tensor = self.transform(src_image)
        tar_tensor = self.transform(tar_image)

        prompt = item.get("prompt", "Convert the style to 3D Chibi Style")
        return {
            "source_images": src_tensor,
            "target_images": tar_tensor,
            "prompts": prompt,
        }


def collate_fn(examples):
    source_images = torch.stack([ex["source_images"] for ex in examples])
    target_images = torch.stack([ex["target_images"] for ex in examples])
    prompts = [ex["prompts"] for ex in examples]
    return {"source_images": source_images, "target_images": target_images, "prompts": prompts}


def _get_viz_batch_from_dataset(dataset: Dataset, num_images: int, offset: int):
    """Fetch `num_images` examples directly from dataset (CPU tensors), starting at `offset` (wrap-around)."""
    if dataset is None:
        raise ValueError("dataset is None")
    if len(dataset) == 0:
        raise ValueError("dataset is empty")

    n = int(max(1, num_images))
    items = [dataset[(offset + i) % len(dataset)] for i in range(n)]
    return collate_fn(items)


def encode_images(vae, images):
    """Encode images to latent space (handles scaling_factor like SanaPipeline)."""
    with torch.no_grad():
        encoded = vae.encode(images)
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
    return latents


@torch.no_grad()
def decode_latents_to_images(vae, latents: torch.Tensor) -> torch.Tensor:
    """Decode latents to images in [0,1] with shape [B,3,H,W]."""
    scaling_factor = getattr(getattr(vae, "config", None), "scaling_factor", None)
    z = latents
    if scaling_factor is not None:
        z = z / scaling_factor

    decoded = vae.decode(z)
    if hasattr(decoded, "sample"):
        images = decoded.sample
    elif hasattr(decoded, "samples"):
        images = decoded.samples
    else:
        images = decoded

    images = (images / 2.0 + 0.5).clamp(0.0, 1.0)
    return images


@torch.no_grad()
def encode_prompts(tokenizer, text_encoder, prompts, device):
    text_inputs = tokenizer(
        prompts,
        padding="max_length",
        max_length=300,
        truncation=True,
        return_tensors="pt",
    ).to(device)

    hs = text_encoder(input_ids=text_inputs.input_ids, attention_mask=text_inputs.attention_mask)[0]
    return hs, text_inputs.attention_mask


def load_cond_proj(vae, device, cond_proj_path: str | None, dtype: torch.dtype):
    latent_channels = getattr(getattr(vae, "config", None), "latent_channels", 4)
    cond_proj = nn.Conv2d(latent_channels * 2, latent_channels, kernel_size=1, bias=True)
    nn.init.zeros_(cond_proj.weight)
    nn.init.zeros_(cond_proj.bias)
    with torch.no_grad():
        for c in range(latent_channels):
            cond_proj.weight[c, c, 0, 0] = 1.0

    if cond_proj_path is not None and Path(cond_proj_path).exists():
        state = torch.load(cond_proj_path, map_location="cpu")
        cond_proj.load_state_dict(state)
        logger.info(f"Loaded cond_proj from {cond_proj_path}")

    return cond_proj.to(device=device, dtype=dtype)


def _get_prev_sample(step_out):
    if hasattr(step_out, "prev_sample"):
        return step_out.prev_sample
    if isinstance(step_out, dict) and "prev_sample" in step_out:
        return step_out["prev_sample"]
    if isinstance(step_out, (tuple, list)) and len(step_out) > 0:
        return step_out[0]
    raise TypeError(f"Unsupported scheduler.step output type: {type(step_out)}")


@torch.no_grad()
def run_bridge_sampling(
    *,
    model,
    cond_proj,
    scheduler,
    x0_latents: torch.Tensor,
    x0_cond: torch.Tensor | None,
    encoder_hidden_states,
    encoder_attention_mask,
    num_inference_steps: int,
    device,
    autocast_ctx,
):
    """Run a minimal bridge sampling loop from x0_latents to x1_pred.

    This is the same core idea as train_scripts/batch_inference_visualize.py concat loop:
    - latents start from x0
    - each step injects concat conditioning via cond_proj([x_t, x0_cond])
    - scheduler.step(...) updates the latent

    IMPORTANT: does not use x1 ground truth.
    """
    sched = deepcopy(scheduler)
    if hasattr(sched, "set_timesteps"):
        try:
            sched.set_timesteps(int(num_inference_steps), device=device)
        except TypeError:
            sched.set_timesteps(int(num_inference_steps))

    latents = x0_latents
    bsz = latents.shape[0]
    model_dtype = next(model.parameters()).dtype
    latents = latents.to(device=device, dtype=model_dtype)
    if x0_cond is not None:
        x0_cond = x0_cond.to(device=device, dtype=model_dtype)

    timesteps = getattr(sched, "timesteps", None)
    if timesteps is None:
        raise ValueError("Scheduler has no timesteps; did set_timesteps() run?")

    for t in timesteps:
        # Sana expects timestep as [B] float32 (bridge convention 0..1000)
        t_val = float(t.item()) if torch.is_tensor(t) else float(t)
        t_batch = torch.full((bsz,), t_val, device=device, dtype=torch.float32)

        with autocast_ctx:
            model_in = latents
            if cond_proj is not None:
                if x0_cond is None:
                    raise ValueError("cond_proj is set but x0_cond is None")
                model_in = cond_proj(torch.cat([latents, x0_cond], dim=1))

            model_out = model(
                hidden_states=model_in,
                timestep=t_batch,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
            ).sample

        step_out = sched.step(model_out, t, latents)
        latents = _get_prev_sample(step_out)

    return latents


class DiscriminatorHead(nn.Module):
    """Small discriminator head over teacher feature maps."""

    def __init__(self, in_channels: int = 4, hidden_channels: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.proj = nn.Linear(hidden_channels, 1)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        # feat: [B,C,H,W]
        x = self.net(feat).flatten(1)
        return self.proj(x).squeeze(-1)


def trig_to_bridge_timestep(converter: CMBridgeConverter, t_trig: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Map trig time t_trig (radians) to (tau, r, t_bridge_0_1000).

    t_trig: shape [B]
    Returns:
      tau: shape [B]
      r: shape [B]
      t_bridge: shape [B] float32 in [0,1000]
    """
    tau = converter.time_trig_to_linear(t_trig).clamp(0.0, 1.0)
    r = (1.0 - tau).clamp(0.0, 1.0)
    t_bridge = (r * 1000.0).clamp(0.0, 1000.0)
    return tau, r, t_bridge


def teacher_feature(
    teacher,
    teacher_cond_proj,
    x_trig: torch.Tensor,
    x0_cond: torch.Tensor,
    t_trig: torch.Tensor,
    converter: CMBridgeConverter,
    encoder_hidden_states,
    encoder_attention_mask,
    device,
):
    """Compute teacher feature Φ by running the frozen bridge model.

    Uses the exact CM projection: h_in = x_trig / (sin+cos) which equals x_lin(tau).
    Time is mapped: t_trig -> tau -> r -> t_bridge.

    Returns v_bridge (dx/dr) in physical latent scale.
    """
    t_batch = t_trig.to(device=device, dtype=torch.float32)
    h_in = converter.project_state_trig_to_linear(x_trig, t_batch)
    _, _, t_bridge = trig_to_bridge_timestep(converter, t_batch)
    t_bridge = t_bridge.to(device=device, dtype=torch.float32)

    model_dtype = next(teacher.parameters()).dtype
    h_in = h_in.to(device=device, dtype=model_dtype)
    x0_cond = x0_cond.to(device=device, dtype=model_dtype)

    model_in = h_in
    if teacher_cond_proj is not None:
        model_in = teacher_cond_proj(torch.cat([h_in, x0_cond], dim=1))

    use_autocast = str(device).startswith("cuda") and model_dtype in (torch.float16, torch.bfloat16)
    autocast_ctx = (
        torch.autocast("cuda", dtype=model_dtype)
        if use_autocast
        else contextlib.nullcontext()
    )
    with autocast_ctx:
        v_bridge = teacher(
            hidden_states=model_in,
            timestep=t_bridge,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
        ).sample
    return v_bridge


def compute_u_hat_and_derivative(
    *,
    teacher,
    teacher_cond_proj,
    x0_trig: torch.Tensor,
    x1_trig: torch.Tensor,
    x0_cond: torch.Tensor,
    t_trig: torch.Tensor,
    converter: CMBridgeConverter,
    sigma_data: float,
    fd_dt: float,
    encoder_hidden_states,
    encoder_attention_mask,
    device,
):
    """Compute CM-Bridge reference field u_hat (normalized) and its total time derivative.

    u_hat_norm(t) = alpha(t) * (x_trig(t) / sigma) + beta(t) * (v_tau(t) / sigma)
    where v_tau = dx/dtau = -dx/dr = -v_bridge.

    We need the TOTAL derivative along the TrigFlow trajectory x_trig(t)=cos(t)x0+sin(t)x1:
      d/dt u_hat_norm(x_trig(t), t)
    This is computed with torch.func.jvp (like train_scm_ladd.py). If unavailable,
    fall back to a finite difference that recomputes x_trig(t+dt) from (x0,x1).
    """

    t0 = t_trig.to(device=device, dtype=torch.float32)
    t_view = t0.view(-1, 1, 1, 1)
    x_trig0 = torch.cos(t_view) * x0_trig + torch.sin(t_view) * x1_trig
    dx_dt = -torch.sin(t_view) * x0_trig + torch.cos(t_view) * x1_trig

    sigma = float(sigma_data)

    def u_hat_fn(x_trig_in: torch.Tensor, t_in: torch.Tensor) -> torch.Tensor:
        alpha, beta = converter.compute_vector_field_coefficients(t_in)
        alpha = alpha.view(-1, 1, 1, 1)
        beta = beta.view(-1, 1, 1, 1)

        v_bridge = teacher_feature(
            teacher,
            teacher_cond_proj,
            x_trig=x_trig_in,
            x0_cond=x0_cond,
            t_trig=t_in,
            converter=converter,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            device=device,
        )
        v_tau = -v_bridge
        x_norm = x_trig_in / sigma
        return alpha * x_norm + beta * (v_tau / sigma)

    # Preferred: JVP along (dx/dt, dt=1)
    # NOTE: PyTorch forward-AD does not support some fused SDPA kernels. When doing JVP,
    # force math-SDPA on CUDA to avoid NotImplementedError.
    if hasattr(torch, "func") and hasattr(torch.func, "jvp"):
        sdp_ctx = contextlib.nullcontext()
        if str(device).startswith("cuda") and hasattr(torch.backends, "cuda"):
            try:
                sdp_ctx = torch.backends.cuda.sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True)
            except Exception:
                sdp_ctx = contextlib.nullcontext()

        try:
            with sdp_ctx:
                with torch.enable_grad():
                    u_hat0, du_dt = torch.func.jvp(
                        u_hat_fn,
                        (x_trig0, t0),
                        (dx_dt, torch.ones_like(t0)),
                    )
            return u_hat0.detach(), du_dt.detach()
        except NotImplementedError as e:
            logger.warning("torch.func.jvp failed (likely SDPA forward-AD kernel limitation); falling back to FD. Error: %s", str(e))

    # Fallback: trajectory-consistent finite difference
    t1 = (t0 + float(fd_dt)).clamp(0.0, (math.pi / 2) - 1e-6)
    t1_view = t1.view(-1, 1, 1, 1)
    x_trig1 = torch.cos(t1_view) * x0_trig + torch.sin(t1_view) * x1_trig
    with torch.no_grad():
        u_hat0 = u_hat_fn(x_trig0, t0)
        u_hat1 = u_hat_fn(x_trig1, t1)
        du_dt = (u_hat1 - u_hat0) / float(fd_dt)
    return u_hat0, du_dt


def parse_args():
    p = argparse.ArgumentParser(description="sCM+LADD distillation for Sana Bridge")

    # Paths
    p.add_argument("--pretrained_model_name_or_path", type=str, required=True)
    p.add_argument("--revision", type=str, default=None)
    p.add_argument("--train_data_dir", type=str, default=None, help="Dataset dir containing train.jsonl (required unless --dry_run)")

    # Teacher init
    p.add_argument("--teacher_lora_path", type=str, required=True, help="Path to teacher LoRA (dir or pytorch_lora_weights.bin)")
    p.add_argument("--teacher_cond_proj_path", type=str, default=None)

    # Student init
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--init_student_lora_path", type=str, default=None, help="Optional: init student LoRA from a checkpoint")

    # LoRA
    p.add_argument("--lora_rank", type=int, default=128)
    p.add_argument("--lora_alpha", type=int, default=128)
    p.add_argument("--lora_dropout", type=float, default=0.0)

    # Conditioning
    p.add_argument(
        "--conditioning",
        type=str,
        default="concat_text",
        choices=["concat", "text", "concat_text"],
    )
    p.add_argument("--booting_noise_scale", type=float, default=0.0)

    # Training
    p.add_argument("--resolution", type=int, default=1024)
    p.add_argument("--train_batch_size", type=int, default=1)
    p.add_argument("--num_train_epochs", type=int, default=100)
    p.add_argument("--max_train_steps", type=int, default=20000)
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--lr_scheduler", type=str, default="constant")
    p.add_argument("--lr_warmup_steps", type=int, default=500)

    # Loss weights
    p.add_argument("--scm_lambda", type=float, default=1.0)
    p.add_argument("--adv_lambda", type=float, default=0.5)
    p.add_argument("--discriminator_lr", type=float, default=1e-4)

    # LADD re-noising (ViBT-LADD): use same (s, eps) for real/fake
    p.add_argument(
        "--ladd_noise_scale",
        type=float,
        default=0.0,
        help="Std scale for shared re-noising eps added to x_s in LADD. 0 disables. Noise is scaled by sqrt(tau*(1-tau)).",
    )

    # TrigFlow sampling
    p.add_argument("--t_trig_min", type=float, default=1e-3)
    p.add_argument("--t_trig_max", type=float, default=(math.pi / 2) - 1e-3)
    p.add_argument("--scm_fd_dt", type=float, default=1e-3, help="Finite-difference dt in trig time for du_hat/dt")
    p.add_argument("--scm_weighting", type=str, default="none", choices=["none", "inv_tan"], help="Optional weighting lambda(t)")

    # ODE-style randomness
    p.add_argument("--init_latents_noise_scale", type=float, default=0.0, help="One-time noise added to x0 before constructing trig states")

    # sigma
    p.add_argument("--sigma_data", type=float, default=1.0)

    # Misc
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--mixed_precision", type=str, default="bf16", choices=["no", "fp16", "bf16"])
    p.add_argument("--report_to", type=str, default="tensorboard")
    p.add_argument("--dataloader_num_workers", type=int, default=0)
    p.add_argument("--logging_steps", type=int, default=10)
    p.add_argument("--checkpointing_steps", type=int, default=500)

    # TensorBoard image logging (optional)
    p.add_argument(
        "--image_logging_steps",
        type=int,
        default=0,
        help="If >0, log source/target/student/teacher images every N steps to the active tracker (tensorboard).",
    )
    p.add_argument("--num_log_images", type=int, default=4, help="Number of images from the batch to decode and log.")

    # TensorBoard visualization: run real sampling from x0 -> x1_pred
    p.add_argument(
        "--viz_num_inference_steps",
        type=int,
        default=4,
        help="When logging images, run N-step sampling from x0 (no x1 leakage).",
    )
    p.add_argument(
        "--viz_scheduler_mode",
        type=str,
        default="vibt",
        choices=["model", "vibt"],
        help="Scheduler used for image logging: 'model' uses checkpoint scheduler; 'vibt' wraps UniPC with ViBTScheduler.",
    )
    p.add_argument("--viz_vibt_noise_scale", type=float, default=0.0, help="ViBT noise_scale for viz (0.0 for ODE bridge)")
    p.add_argument("--viz_vibt_shift_gamma", type=float, default=5.0, help="ViBT shift_gamma for viz")

    # Debug
    p.add_argument("--dry_run", action="store_true", help="Run a few synthetic steps to validate shapes/dtypes without loading a dataset")
    p.add_argument("--dry_run_steps", type=int, default=1)

    return p.parse_args()


def _resolve_lora_file(path: str) -> Path:
    p = Path(path)
    if p.is_dir():
        return p / "pytorch_lora_weights.bin"
    return p


def main():
    args = parse_args()

    if not args.dry_run and not args.train_data_dir:
        raise ValueError("--train_data_dir is required unless --dry_run is set")

    logging_dir = Path(args.output_dir) / "logs"
    accel_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accel_config,
    )
    if accelerator.is_main_process:
        accelerator.init_trackers("bridge_scm_ladd_distill", config=vars(args))

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)

    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

    tb_writer = None
    if accelerator.is_main_process and args.report_to == "tensorboard":
        try:
            tb_writer = accelerator.get_tracker("tensorboard").writer
        except Exception:
            tb_writer = None

    weight_dtype = torch.float32
    if args.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif args.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Load base components
    vae = AutoencoderDC.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision)
    vae.requires_grad_(False)
    vae.to(accelerator.device, dtype=torch.float32)

    text_encoder = Gemma2Model.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision, torch_dtype=weight_dtype
    )
    text_encoder.requires_grad_(False)
    text_encoder.to(accelerator.device)

    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer", revision=args.revision)

    base_transformer = SanaTransformer2DModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="transformer",
        revision=args.revision,
        torch_dtype=weight_dtype,
    )

    # Scheduler only needed for validation sampling; keep consistent with checkpoint
    base_pipe = SanaPipeline.from_pretrained(args.pretrained_model_name_or_path, revision=args.revision, torch_dtype=weight_dtype)
    scheduler_for_val = base_pipe.scheduler
    del base_pipe

    if args.viz_scheduler_mode == "vibt":
        try:
            from diffusers.schedulers import UniPCMultistepScheduler
            from vibt.scheduler import ViBTScheduler

            if isinstance(scheduler_for_val, UniPCMultistepScheduler):
                scheduler_for_val = ViBTScheduler.from_scheduler(
                    scheduler_for_val,
                    noise_scale=float(args.viz_vibt_noise_scale),
                    shift_gamma=float(args.viz_vibt_shift_gamma),
                )
            else:
                base_unipc = UniPCMultistepScheduler.from_config(scheduler_for_val.config)
                scheduler_for_val = ViBTScheduler.from_scheduler(
                    base_unipc,
                    noise_scale=float(args.viz_vibt_noise_scale),
                    shift_gamma=float(args.viz_vibt_shift_gamma),
                )
            logger.info(
                "Viz scheduler: ViBT (noise_scale=%.3g shift_gamma=%.3g)",
                float(args.viz_vibt_noise_scale),
                float(args.viz_vibt_shift_gamma),
            )
        except Exception as e:
            logger.warning("Failed to build ViBT viz scheduler; falling back to model scheduler. Error: %s", str(e))

    # Converter
    converter = CMBridgeConverter(sigma_data=args.sigma_data)

    vae_scaling_factor = getattr(getattr(vae, "config", None), "scaling_factor", None)
    logger.info("VAE scaling_factor = %s", str(vae_scaling_factor))
    logger.info("args.sigma_data = %.6g | converter.sigma_data = %.6g", float(args.sigma_data), float(converter.sigma_data))

    # Build cond_proj (shared structure for teacher/student)
    cond_proj = None
    if args.conditioning in {"concat", "concat_text"}:
        # We'll train student cond_proj (optional); teacher cond_proj loaded separately.
        cond_proj = load_cond_proj(vae, accelerator.device, cond_proj_path=None, dtype=weight_dtype)

    # Teacher: frozen bridge model
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        init_lora_weights="gaussian",
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
        lora_dropout=args.lora_dropout,
    )

    teacher = get_peft_model(deepcopy(base_transformer), lora_config)
    teacher_lora_file = _resolve_lora_file(args.teacher_lora_path)
    if not teacher_lora_file.exists():
        raise FileNotFoundError(f"Teacher LoRA not found: {teacher_lora_file}")
    if set_peft_model_state_dict is None:
        raise RuntimeError("set_peft_model_state_dict is not available; please update peft")

    teacher_state = torch.load(teacher_lora_file, map_location="cpu")
    set_peft_model_state_dict(teacher, teacher_state)
    teacher = teacher.to(accelerator.device)
    teacher.requires_grad_(False)
    teacher.eval()

    teacher_cond_proj = None
    if cond_proj is not None:
        teacher_cond_proj = load_cond_proj(
            vae,
            accelerator.device,
            cond_proj_path=args.teacher_cond_proj_path,
            dtype=weight_dtype,
        )
        teacher_cond_proj.requires_grad_(False)
        teacher_cond_proj.eval()

    # Student: trainable LoRA (and optional cond_proj)
    student = get_peft_model(base_transformer, lora_config)
    student.enable_gradient_checkpointing()

    if args.init_student_lora_path:
        student_lora_file = _resolve_lora_file(args.init_student_lora_path)
        if not student_lora_file.exists():
            raise FileNotFoundError(f"Student init LoRA not found: {student_lora_file}")
        state = torch.load(student_lora_file, map_location="cpu")
        set_peft_model_state_dict(student, state)
        logger.info(f"Initialized student LoRA from {student_lora_file}")

    student = student.to(accelerator.device)

    # Discriminator
    disc = DiscriminatorHead(in_channels=getattr(getattr(vae, "config", None), "latent_channels", 4), hidden_channels=64)
    disc.to(accelerator.device, dtype=weight_dtype)

    # Optimizers
    student_params = list(student.parameters()) + (list(cond_proj.parameters()) if cond_proj is not None else [])
    optimizer_G = torch.optim.AdamW(student_params, lr=args.learning_rate, betas=(0.9, 0.999), weight_decay=1e-2, eps=1e-8)
    optimizer_D = torch.optim.AdamW(disc.parameters(), lr=args.discriminator_lr, betas=(0.0, 0.99), weight_decay=0.0, eps=1e-8)

    # LR scheduler for G
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer_G,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )

    # Prepare (data loader only in non-dry-run)
    if args.dry_run:
        if cond_proj is not None:
            student, cond_proj, disc, optimizer_G, optimizer_D = accelerator.prepare(
                student, cond_proj, disc, optimizer_G, optimizer_D
            )
        else:
            student, disc, optimizer_G, optimizer_D = accelerator.prepare(student, disc, optimizer_G, optimizer_D)
        train_dataset = None
        train_dataloader = None
    else:
        train_dataset = BridgeDataset(args.train_data_dir, resolution=args.resolution)
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=args.train_batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=args.dataloader_num_workers,
        )
        if cond_proj is not None:
            student, cond_proj, disc, optimizer_G, optimizer_D, train_dataloader = accelerator.prepare(
                student, cond_proj, disc, optimizer_G, optimizer_D, train_dataloader
            )
        else:
            student, disc, optimizer_G, optimizer_D, train_dataloader = accelerator.prepare(
                student, disc, optimizer_G, optimizer_D, train_dataloader
            )

    logger.info("***** Running sCM+LADD distillation *****")
    if train_dataset is not None:
        logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Batch size = {args.train_batch_size}")
    logger.info(f"  Max steps = {args.max_train_steps}")
    logger.info(f"  sigma_data = {args.sigma_data}")

    global_step = 0
    max_steps = int(args.dry_run_steps) if args.dry_run else int(args.max_train_steps)
    progress_bar = tqdm(
        range(max_steps),
        initial=global_step,
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )

    # For deterministic visualization sampling (independent from train batch size)
    viz_offset = 0

    autocast_ctx = (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if str(accelerator.device).startswith("cuda") and args.mixed_precision == "bf16"
        else contextlib.nullcontext()
    )

    for epoch in range(args.num_train_epochs if not args.dry_run else 1):
        student.train()
        if cond_proj is not None:
            cond_proj.train()

        if args.dry_run:
            # Synthetic batch in image space (normalized to [-1, 1])
            bsz = int(args.train_batch_size)
            res = int(args.resolution)
            source_images = (torch.rand(bsz, 3, res, res, device=accelerator.device) * 2.0 - 1.0).clamp(-1.0, 1.0)
            target_images = (torch.rand(bsz, 3, res, res, device=accelerator.device) * 2.0 - 1.0).clamp(-1.0, 1.0)
            batch = {
                "source_images": source_images,
                "target_images": target_images,
                "prompts": ["dry run"] * bsz,
            }
            batch_iter = [batch] * max_steps
        else:
            batch_iter = train_dataloader

        for step, batch in enumerate(batch_iter):
            if global_step >= max_steps:
                break

            with accelerator.accumulate(student):
                source_images = batch["source_images"].to(accelerator.device)
                target_images = batch["target_images"].to(accelerator.device)
                prompts = batch["prompts"]

                with torch.no_grad():
                    x0 = encode_images(vae, source_images)
                    x1 = encode_images(vae, target_images)

                # Optional one-time noise on x0 for ODE-style randomness
                if args.init_latents_noise_scale and args.init_latents_noise_scale > 0:
                    x0_noisy = x0 + args.init_latents_noise_scale * torch.randn_like(x0)
                else:
                    x0_noisy = x0

                # Booting noise for concat conditioning (conditioning only)
                x0_cond = x0
                if cond_proj is not None and args.booting_noise_scale and args.booting_noise_scale > 0:
                    x0_cond = x0 + args.booting_noise_scale * torch.randn_like(x0)

                encoder_hidden_states, encoder_attention_mask = None, None
                if args.conditioning in {"text", "concat_text"}:
                    with torch.no_grad():
                        encoder_hidden_states, encoder_attention_mask = encode_prompts(
                            tokenizer, text_encoder, prompts, accelerator.device
                        )

                bsz = x0.shape[0]
                t_trig = args.t_trig_min + (args.t_trig_max - args.t_trig_min) * torch.rand(bsz, device=accelerator.device)

                t_view = t_trig.view(-1, 1, 1, 1)
                x_trig = torch.cos(t_view) * x0_noisy + torch.sin(t_view) * x1

                # Student forward (physical scale)
                with autocast_ctx:
                    model_in = x_trig
                    if cond_proj is not None:
                        model_in = cond_proj(torch.cat([x_trig, x0_cond], dim=1))

                    # Feed bridge-domain timestep derived from trig time (keeps t in 0..1000)
                    _, _, t_bridge = trig_to_bridge_timestep(converter, t_trig)
                    t_bridge = t_bridge.to(device=accelerator.device, dtype=torch.float32)

                    u_pred_phys = student(
                        hidden_states=model_in,
                        timestep=t_bridge,
                        encoder_hidden_states=encoder_hidden_states,
                        encoder_attention_mask=encoder_attention_mask,
                    ).sample

                # Normalized quantities
                sigma = float(args.sigma_data)
                z_t = x_trig / sigma
                F_norm = u_pred_phys / sigma

                # ======== Discriminator update (hinge) ========
                disc.train()
                student.eval()
                if cond_proj is not None:
                    cond_proj.eval()

                # Sample s, and a shared eps for real/fake (variance reduction).
                # Note: x1 in this script is expected to be teacher-synth target when using train_teacher_synth.jsonl.
                s_trig = args.t_trig_min + (args.t_trig_max - args.t_trig_min) * torch.rand(bsz, device=accelerator.device)
                s_view = s_trig.view(-1, 1, 1, 1)
                ladd_noise = None
                if args.ladd_noise_scale and float(args.ladd_noise_scale) > 0.0:
                    with torch.no_grad():
                        eps = torch.randn_like(x0)
                        tau_s = converter.time_trig_to_linear(s_trig).clamp(0.0, 1.0).view(-1, 1, 1, 1)
                        ladd_noise = float(args.ladd_noise_scale) * torch.sqrt(tau_s * (1.0 - tau_s) + 1e-8) * eps

                with torch.no_grad():
                    # predicted x1 (normalized), then to physical
                    x1_hat_norm = torch.sin(t_view) * z_t + torch.cos(t_view) * F_norm
                    x1_hat = x1_hat_norm * sigma

                    # Construct re-noised bridge states at the same s with shared eps
                    x_s_gen = torch.cos(s_view) * x0 + torch.sin(s_view) * x1_hat
                    x_s_data = torch.cos(s_view) * x0 + torch.sin(s_view) * x1
                    if ladd_noise is not None:
                        x_s_gen = x_s_gen + ladd_noise
                        x_s_data = x_s_data + ladd_noise

                    feat_fake = teacher_feature(
                        teacher,
                        teacher_cond_proj,
                        x_trig=x_s_gen,
                        x0_cond=x0_cond,
                        t_trig=s_trig,
                        converter=converter,
                        encoder_hidden_states=encoder_hidden_states,
                        encoder_attention_mask=encoder_attention_mask,
                        device=accelerator.device,
                    )
                    feat_real = teacher_feature(
                        teacher,
                        teacher_cond_proj,
                        x_trig=x_s_data,
                        x0_cond=x0_cond,
                        t_trig=s_trig,
                        converter=converter,
                        encoder_hidden_states=encoder_hidden_states,
                        encoder_attention_mask=encoder_attention_mask,
                        device=accelerator.device,
                    )

                pred_fake = disc(feat_fake.detach())
                pred_real = disc(feat_real.detach())
                loss_real = torch.mean(F.relu(1.0 - pred_real))
                loss_fake = torch.mean(F.relu(1.0 + pred_fake))
                loss_D = 0.5 * (loss_real + loss_fake)

                optimizer_D.zero_grad(set_to_none=True)
                accelerator.backward(loss_D)
                optimizer_D.step()

                # ======== Student update ========
                disc.eval()
                student.train()
                if cond_proj is not None:
                    cond_proj.train()

                u_hat, du_hat_dt = compute_u_hat_and_derivative(
                    teacher=teacher,
                    teacher_cond_proj=teacher_cond_proj,
                    x0_trig=x0_noisy,
                    x1_trig=x1,
                    x0_cond=x0_cond,
                    t_trig=t_trig,
                    converter=converter,
                    sigma_data=args.sigma_data,
                    fd_dt=args.scm_fd_dt,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    device=accelerator.device,
                )

                sin_t = torch.sin(t_view)
                cos_t = torch.cos(t_view)

                # sCM residual (normalized)
                residual = sin_t * (F_norm - u_hat) - cos_t * (z_t + du_hat_dt)
                loss_scm = torch.mean(residual * residual)

                if args.scm_weighting == "inv_tan":
                    weight = 1.0 / (torch.tan(t_view).clamp(min=1e-4))
                    loss_scm = torch.mean(weight * (residual * residual))

                # Generator adversarial loss: -D(feat_fake)
                # IMPORTANT: do NOT wrap in no_grad; we need gradients to flow from D -> teacher_feature -> x_s_gen -> student
                # Recompute x_s_gen WITH gradients (the discriminator update computed x_s_gen under no_grad).
                s_trig_G = s_trig
                s_view_G = s_trig_G.view(-1, 1, 1, 1)
                x1_hat_G = (torch.sin(t_view) * z_t + torch.cos(t_view) * F_norm) * sigma
                x_s_gen_G = torch.cos(s_view_G) * x0 + torch.sin(s_view_G) * x1_hat_G
                if ladd_noise is not None:
                    x_s_gen_G = x_s_gen_G + ladd_noise

                feat_fake_G = teacher_feature(
                    teacher,
                    teacher_cond_proj,
                    x_trig=x_s_gen_G,
                    x0_cond=x0_cond,
                    t_trig=s_trig_G,
                    converter=converter,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    device=accelerator.device,
                )

                pred_fake_G = disc(feat_fake_G)
                loss_gen = -torch.mean(pred_fake_G)

                total_loss = args.scm_lambda * loss_scm + args.adv_lambda * loss_gen
                total_loss = total_loss / args.gradient_accumulation_steps

                optimizer_G.zero_grad(set_to_none=True)
                accelerator.backward(total_loss)
                optimizer_G.step()
                lr_scheduler.step()

            if accelerator.sync_gradients:
                # Optional TensorBoard image logging (main process only)
                if (
                    accelerator.is_main_process
                    and args.image_logging_steps
                    and args.image_logging_steps > 0
                    and (global_step + 1) % int(args.image_logging_steps) == 0
                ):
                    with torch.no_grad():
                        # IMPORTANT:
                        # Don't rely on current train batch size (often 1). Instead, fetch N items
                        # directly from the dataset like train_bridge_lora_sana_concat.py validation.
                        if args.dry_run:
                            viz_batch = batch
                        else:
                            viz_batch = _get_viz_batch_from_dataset(train_dataset, int(args.num_log_images), viz_offset)
                            viz_offset = (viz_offset + int(args.num_log_images)) % max(1, len(train_dataset))

                        viz_source_images = viz_batch["source_images"].to(accelerator.device)
                        viz_target_images = viz_batch["target_images"].to(accelerator.device)
                        viz_prompts = viz_batch["prompts"]

                        x0_viz = encode_images(vae, viz_source_images)
                        x1_viz = encode_images(vae, viz_target_images)

                        # conditioning latents
                        x0_cond_viz = x0_viz
                        if cond_proj is not None and args.booting_noise_scale and args.booting_noise_scale > 0:
                            x0_cond_viz = x0_viz + args.booting_noise_scale * torch.randn_like(x0_viz)

                        enc_hs_viz, enc_mask_viz = None, None
                        if args.conditioning in {"text", "concat_text"}:
                            enc_hs_viz, enc_mask_viz = encode_prompts(tokenizer, text_encoder, viz_prompts, accelerator.device)

                        # Deterministic trig time for visualization (middle point)
                        # True inference visualization (x0 -> x1_pred), no x1 leakage
                        x1_hat_viz = run_bridge_sampling(
                            model=student,
                            cond_proj=cond_proj,
                            scheduler=scheduler_for_val,
                            x0_latents=x0_viz,
                            x0_cond=x0_cond_viz if cond_proj is not None else None,
                            encoder_hidden_states=enc_hs_viz,
                            encoder_attention_mask=enc_mask_viz,
                            num_inference_steps=int(args.viz_num_inference_steps),
                            device=accelerator.device,
                            autocast_ctx=autocast_ctx,
                        )
                        x1_hat_teacher_viz = run_bridge_sampling(
                            model=teacher,
                            cond_proj=teacher_cond_proj,
                            scheduler=scheduler_for_val,
                            x0_latents=x0_viz,
                            x0_cond=x0_cond_viz if teacher_cond_proj is not None else None,
                            encoder_hidden_states=enc_hs_viz,
                            encoder_attention_mask=enc_mask_viz,
                            num_inference_steps=int(args.viz_num_inference_steps),
                            device=accelerator.device,
                            autocast_ctx=autocast_ctx,
                        )

                        # Decode to [0,1] CPU for mosaic
                        src_img = decode_latents_to_images(vae, x0_viz.float()).detach().cpu()
                        tar_img = decode_latents_to_images(vae, x1_viz.float()).detach().cpu()
                        stu_img = decode_latents_to_images(vae, x1_hat_viz.float()).detach().cpu()
                        tea_img = decode_latents_to_images(vae, x1_hat_teacher_viz.float()).detach().cpu()

                        rows = []
                        for i in range(src_img.shape[0]):
                            row = torch.cat([src_img[i], tar_img[i], stu_img[i], tea_img[i]], dim=2)  # [3,H,4W]
                            rows.append(row)
                        comp = torch.cat(rows, dim=1)  # [3, N*H, 4*W]

                        if tb_writer is not None:
                            tb_writer.add_image("images/comparison", comp, global_step + 1)

                        if is_wandb_available() and "wandb" in str(args.report_to).split(","):
                            img_np = (
                                (comp.permute(1, 2, 0).numpy() * 255.0)
                                .round()
                                .clip(0, 255)
                                .astype("uint8")
                            )
                            wandb.log({"images/comparison": wandb.Image(img_np)}, step=global_step + 1)

                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process and global_step % args.logging_steps == 0:
                    logs = {
                        "loss_scm": accelerator.gather(loss_scm.detach().float().reshape(1)).mean().item(),
                        "loss_D": accelerator.gather(loss_D.detach().float().reshape(1)).mean().item(),
                        "loss_gen": accelerator.gather(loss_gen.detach().float().reshape(1)).mean().item(),
                        "total_loss": accelerator.gather(total_loss.detach().float().reshape(1)).mean().item(),
                        "lr": lr_scheduler.get_last_lr()[0],
                    }
                    accelerator.log(logs, step=global_step)
                    logger.info(
                        "Step %d: total=%.4f scm=%.4f gen=%.4f D=%.4f lr=%.3e",
                        global_step,
                        logs["total_loss"],
                        logs["loss_scm"],
                        logs["loss_gen"],
                        logs["loss_D"],
                        logs["lr"],
                    )

                if accelerator.is_main_process and global_step % args.checkpointing_steps == 0:
                    save_dir = Path(args.output_dir) / f"checkpoint-{global_step}"
                    save_dir.mkdir(parents=True, exist_ok=True)

                    # Save LoRA weights
                    lora_state = student.state_dict()
                    torch.save(lora_state, save_dir / "pytorch_lora_weights.bin")

                    # Save cond_proj if used
                    if cond_proj is not None:
                        torch.save(accelerator.unwrap_model(cond_proj).state_dict(), save_dir / "cond_proj_weights.bin")

                    # Save discriminator
                    torch.save(accelerator.unwrap_model(disc).state_dict(), save_dir / "disc_head.bin")

                    logger.info(f"Saved checkpoint to {save_dir}")

    accelerator.end_training()


if __name__ == "__main__":
    main()
