#!/usr/bin/env python
"""experiment_teacher_vs_dataset_bridge.py

Goal: verify how well the trained bridge teacher maps dataset source images x0 to target images x1.

For a small subset of pairs from a bridge dataset (train.jsonl with {src, tar, prompt}):
 - take x0 (source image) from the dataset;
 - run the teacher bridge inference from x0 using the existing batch_inference_visualize pipeline
   (same scheduler, same concat_text conditioning);
 - compare the rendered teacher outputs x1_hat with the dataset targets x1.

Outputs:
 - per-pair comparison images: [x0 | x1 (dataset target) | x1_hat (teacher)]
 - an optional summary grid.

This is meant as a diagnostic to check distribution mismatch: if teacher was trained on
one distribution and the dataset (x0,x1) is very different, x1_hat may systematically
deviate from x1 even when the teacher bridge is internally consistent.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from diffusers import AutoencoderDC, SanaTransformer2DModel, SanaPipeline
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
from torchvision import transforms
from transformers import AutoTokenizer, Gemma2Model

from batch_inference_visualize import load_scheduler_from_model, load_cond_proj, encode_prompts, decode_latents


def parse_args():
    p = argparse.ArgumentParser(description="Teacher bridge vs dataset (x0,x1) comparison experiment")
    p.add_argument("--model_path", type=str, required=True, help="Base Sana model path (same as training)")
    p.add_argument("--teacher_lora_path", type=str, required=True, help="Teacher LoRA checkpoint (pytorch_lora_weights.bin or dir)")
    p.add_argument("--data_dir", type=str, required=True, help="Bridge dataset dir containing train.jsonl with src/tar/prompt")
    p.add_argument("--output_dir", type=str, required=True, help="Output dir for comparison images")
    p.add_argument("--max_pairs", type=int, default=8, help="Max number of (x0,x1) pairs to visualize")
    p.add_argument("--resolution", type=int, default=1024)

    p.add_argument("--conditioning", type=str, default="concat_text", choices=["text", "concat", "concat_text"], help="Conditioning mode (must match training)")
    p.add_argument("--cond_proj_path", type=str, default=None, help="Optional cond_proj_weights.bin for concat/concat_text")
    p.add_argument("--booting_noise_scale", type=float, default=0.0, help="Booting noise scale for concat conditioning")

    p.add_argument("--num_inference_steps", type=int, default=28)
    p.add_argument("--scheduler_mode", type=str, default="vibt", choices=["model", "vibt"], help="Scheduler backend (match your training/inference)")
    p.add_argument("--vibt_noise_scale", type=float, default=0.0, help="ViBT noise_scale (0.0 for ODE bridge)")
    p.add_argument("--vibt_shift_gamma", type=float, default=5.0, help="ViBT flow_shift gamma")

    p.add_argument("--lora_rank", type=int, default=128)
    p.add_argument("--lora_alpha", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def load_pairs(data_dir: Path, max_pairs: int):
    jsonl_path = data_dir / "train.jsonl"
    data = []
    with open(jsonl_path, "r") as f:
        for line in f:
            data.append(json.loads(line))
            if len(data) >= max_pairs:
                break
    return data


def build_transform(resolution: int):
    return transforms.Compose(
        [
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )


def encode_image_tensor(vae, image_tensor: torch.Tensor):
    """Encode a preprocessed image tensor [1,3,H,W] to latents with scaling_factor."""
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
    return latents


def make_triplet_image(x0_img: Image.Image, x1_img: Image.Image, x1_hat_img: Image.Image, caption: str | None = None) -> Image.Image:
    """Create [x0 | x1 (dataset) | x1_hat (teacher)] canvas with optional caption."""
    w, h = x0_img.size
    add_text = caption is not None
    text_h = 60 if add_text else 0
    canvas = Image.new("RGB", (w * 3, h + text_h), "white")
    canvas.paste(x0_img, (0, 0))
    canvas.paste(x1_img, (w, 0))
    canvas.paste(x1_hat_img, (2 * w, 0))

    if add_text:
        draw = ImageDraw.Draw(canvas)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
        except Exception:
            font = ImageFont.load_default()
        draw.text((10, h + 10), "x0 (source)", fill="black", font=font)
        draw.text((w + 10, h + 10), "x1 (dataset target)", fill="black", font=font)
        draw.text((2 * w + 10, h + 10), "x1_hat (teacher)", fill="black", font=font)
        if caption:
            draw.text((10, h + 35), caption[:80], fill="black", font=font)

    return canvas


def main():
    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_path = Path(args.model_path)
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load a few pairs from dataset
    pairs = load_pairs(data_dir, args.max_pairs)
    if not pairs:
        raise RuntimeError(f"No pairs loaded from {data_dir}/train.jsonl")

    # Models
    vae = AutoencoderDC.from_pretrained(model_path, subfolder="vae", torch_dtype=torch.float32).to(device)
    text_encoder = Gemma2Model.from_pretrained(model_path, subfolder="text_encoder", torch_dtype=torch.bfloat16).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_path, subfolder="tokenizer")
    base_transformer = SanaTransformer2DModel.from_pretrained(
        model_path,
        subfolder="transformer",
        torch_dtype=torch.bfloat16,
    )

    # Teacher LoRA
    lora_cfg = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
        lora_dropout=0.0,
    )
    teacher = get_peft_model(base_transformer, lora_cfg)
    state = torch.load(args.teacher_lora_path, map_location="cpu")
    set_peft_model_state_dict(teacher, state)
    teacher = teacher.to(device)

    # Scheduler & pipeline
    scheduler = load_scheduler_from_model(str(model_path))
    if args.scheduler_mode == "vibt":
        from diffusers.schedulers import UniPCMultistepScheduler
        from vibt.scheduler import ViBTScheduler

        if hasattr(scheduler, "config") and not isinstance(scheduler, UniPCMultistepScheduler):
            scheduler = UniPCMultistepScheduler.from_config(scheduler.config)
        scheduler = ViBTScheduler.from_scheduler(
            scheduler,
            noise_scale=args.vibt_noise_scale,
            shift_gamma=args.vibt_shift_gamma,
        )

    pipe = SanaPipeline(vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, transformer=teacher, scheduler=scheduler).to(device)
    pipe.vae.eval()
    pipe.text_encoder.eval()
    pipe.transformer.eval()

    # Conditioning
    use_concat_conditioning = args.conditioning in {"concat", "concat_text"}
    cond_proj = None
    if use_concat_conditioning:
        cond_proj_path = args.cond_proj_path
        if cond_proj_path is None:
            cand = Path(args.teacher_lora_path).resolve().parent / "cond_proj_weights.bin"
            if cand.exists():
                cond_proj_path = str(cand)
        if cond_proj_path is None or not Path(cond_proj_path).exists():
            raise FileNotFoundError("concat/concat_text requires cond_proj_weights.bin; set --cond_proj_path or keep next to teacher_lora.")
        cond_proj = load_cond_proj(vae, device, cond_proj_path)
        cond_proj.eval()

    transform = build_transform(args.resolution)
    gen = torch.Generator(device=device).manual_seed(args.seed)

    for idx, item in enumerate(pairs):
        src_rel = item["src"]
        tar_rel = item["tar"]
        # Handle paths that already include the dataset dir name prefix
        # (e.g., data_dir=/.../3D_Chibi and src="3D_Chibi/src/014.png").
        data_dir_name = data_dir.name
        prefix = data_dir_name + "/"
        if src_rel.startswith(prefix):
            src_rel = src_rel[len(prefix) :]
        if tar_rel.startswith(prefix):
            tar_rel = tar_rel[len(prefix) :]
        prompt = item.get("prompt", "Convert the style to 3D Chibi Style")

        src_path = data_dir / src_rel
        tar_path = data_dir / tar_rel

        # Load & preprocess images
        x0_img = Image.open(src_path).convert("RGB")
        x1_img = Image.open(tar_path).convert("RGB")

        x0_tensor = transform(x0_img).unsqueeze(0).to(device)
        x1_tensor = transform(x1_img).unsqueeze(0).to(device)

        # Encode to latents
        x0_latents = encode_image_tensor(vae, x0_tensor)
        x1_latents = encode_image_tensor(vae, x1_tensor)

        # Teacher bridge inference from x0_latents -> x1_hat_latents
        cond_latents = x0_latents
        if use_concat_conditioning and args.booting_noise_scale and args.booting_noise_scale > 0:
            noise = torch.randn_like(x0_latents, generator=gen)
            cond_latents = x0_latents + args.booting_noise_scale * noise

        with torch.inference_mode():
            autocast_ctx = torch.autocast("cuda", dtype=torch.bfloat16) if device == "cuda" else torch.no_grad()
            with autocast_ctx:
                if not use_concat_conditioning:
                    # Direct pipeline mode: treat x0 as initial latents and run bridge
                    out = pipe(
                        prompt="Convert the style to 3D Chibi Style",
                        num_inference_steps=args.num_inference_steps,
                        guidance_scale=1.0,
                        latents=x0_latents,
                        generator=gen,
                    )
                    x1_hat_img = out.images[0]
                else:
                    # Custom concat loop, matching batch_inference_visualize concat backend
                    prompt_hs, prompt_mask = encode_prompts(tokenizer, text_encoder, prompt, device)
                    encoder_hs, encoder_mask = prompt_hs, prompt_mask

                    sched = pipe.scheduler
                    sched.set_timesteps(args.num_inference_steps, device=device)
                    latents = x0_latents
                    for t in sched.timesteps:
                        model_in = cond_proj(torch.cat([latents, cond_latents], dim=1))
                        t_scalar = t
                        if not torch.is_tensor(t_scalar):
                            t_scalar = torch.tensor(t_scalar, device=device)
                        timesteps = torch.full((model_in.shape[0],), t_scalar.item(), device=device, dtype=torch.float32)

                        model_out = teacher(
                            hidden_states=model_in,
                            timestep=timesteps,
                            encoder_hidden_states=encoder_hs,
                            encoder_attention_mask=encoder_mask,
                        ).sample

                        step = sched.step(model_out, t, latents, return_dict=True)
                        if hasattr(step, "prev_sample"):
                            latents = step.prev_sample
                        else:
                            latents = step[0]

                    img_tensor = decode_latents(vae, latents).clamp(-1, 1)
                    img = (img_tensor[0].detach().float().cpu() + 1) / 2
                    img = img.permute(1, 2, 0).numpy()
                    img = (img * 255).round().clip(0, 255).astype(np.uint8)
                    x1_hat_img = Image.fromarray(img)

        # Ensure all three are same size for visualization
        x0_vis = x0_img.resize(x1_hat_img.size, Image.Resampling.LANCZOS)
        x1_vis = x1_img.resize(x1_hat_img.size, Image.Resampling.LANCZOS)

        triplet = make_triplet_image(x0_vis, x1_vis, x1_hat_img, caption=prompt)
        out_path = output_dir / f"pair_{idx:03d}.png"
        triplet.save(out_path)
        print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
