#!/usr/bin/env python
"""build_teacher_synthetic_dataset.py

Build a teacher-synthesized target dataset for bridge distillation.

Motivation:
- If the original dataset targets x1 are out-of-distribution for the teacher bridge model,
  then distilling a student against those targets can create "artifact superposition".
- This script discards the original target image, and replaces it with a teacher-generated
  target x1_hat inferred from the source x0 (and prompt) using the same inference loop
  as batch_inference_visualize.py.

Input:
- Existing dataset directory containing a jsonl (default: train.jsonl) with fields:
  {"src": <path>, "tar": <path>, "prompt": <optional>}

Output:
- New images saved under a subdirectory in the same dataset folder (default: tar_teacher/)
- A new jsonl (default: train_teacher_synth.jsonl) that preserves "src" and prompt,
  but rewrites "tar" to point to the synthesized teacher target.

Notes:
- Keeps path conventions compatible with existing dataset loaders.
- Uses concat_text conditioning if requested (requires cond_proj_weights.bin).
- For ODE bridge, set --scheduler_mode vibt --vibt_noise_scale 0.0.
"""

import argparse
import json
from pathlib import Path
import contextlib

import numpy as np
import torch
from PIL import Image
from diffusers import AutoencoderDC, SanaPipeline, SanaTransformer2DModel
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
from torchvision import transforms
from transformers import AutoTokenizer, Gemma2Model

from batch_inference_visualize import (
    decode_latents,
    encode_prompts,
    load_cond_proj,
    load_scheduler_from_model,
    _get_prev_sample,
)


def parse_args():
    p = argparse.ArgumentParser(description="Build teacher-synthesized bridge dataset")

    p.add_argument("--model_path", type=str, required=True, help="Base Sana model path")
    p.add_argument(
        "--teacher_lora_path",
        type=str,
        required=True,
        help="Teacher LoRA checkpoint (pytorch_lora_weights.bin or file)",
    )

    p.add_argument("--data_dir", type=str, required=True, help="Dataset root directory")
    p.add_argument("--train_jsonl", type=str, default="train.jsonl", help="Input jsonl filename (relative to data_dir)")
    p.add_argument("--output_jsonl", type=str, default="train_teacher_synth.jsonl", help="Output jsonl filename (relative to data_dir)")
    p.add_argument("--new_tar_subdir", type=str, default="tar_teacher", help="Subdir under data_dir to save synthesized targets")

    p.add_argument("--max_items", type=int, default=None, help="Optional cap on number of items")
    p.add_argument("--resolution", type=int, default=1024)

    p.add_argument("--conditioning", type=str, default="concat_text", choices=["text", "concat", "concat_text"])
    p.add_argument("--cond_proj_path", type=str, default=None)
    p.add_argument("--booting_noise_scale", type=float, default=0.0)

    p.add_argument("--num_inference_steps", type=int, default=28)
    p.add_argument("--scheduler_mode", type=str, default="vibt", choices=["model", "vibt"])
    p.add_argument("--vibt_noise_scale", type=float, default=0.0)
    p.add_argument("--vibt_shift_gamma", type=float, default=5.0)

    p.add_argument("--guidance_scale", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--lora_rank", type=int, default=128)
    p.add_argument("--lora_alpha", type=int, default=128)

    p.add_argument("--overwrite", action="store_true", help="Overwrite existing synthesized images")

    return p.parse_args()


def _resolve_lora_file(path: str) -> Path:
    p = Path(path)
    if p.is_dir():
        return p / "pytorch_lora_weights.bin"
    return p


def _normalize_relpath(rel: str, data_dir: Path) -> str:
    """Match BridgeDataset behavior: strip a leading '<data_dir.name>/' prefix if present."""
    data_dir_name = data_dir.name
    prefix = data_dir_name + "/"
    if rel.startswith(prefix):
        return rel[len(prefix) :]
    return rel


def build_transform(resolution: int):
    return transforms.Compose(
        [
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )


@torch.no_grad()
def encode_image_tensor(vae, image_tensor: torch.Tensor):
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


def latents_to_pil(vae, latents: torch.Tensor) -> Image.Image:
    # Ensure latents match VAE dtype (VAE is loaded in float32 in this script).
    # In the teacher loop we run under bf16 autocast, so latents can become bf16;
    # cast back to float32 here to avoid conv/bias dtype mismatches during decode.
    latents = latents.to(torch.float32)
    img_tensor = decode_latents(vae, latents).clamp(-1, 1)
    img = (img_tensor[0].detach().float().cpu() + 1) / 2
    img = img.permute(1, 2, 0).numpy()
    img = (img * 255).round().clip(0, 255).astype(np.uint8)
    return Image.fromarray(img)


@torch.inference_mode()
def infer_teacher_target_latents(
    *,
    teacher,
    vae,
    tokenizer,
    text_encoder,
    scheduler,
    cond_proj,
    conditioning: str,
    prompt: str,
    source_latents: torch.Tensor,
    booting_noise_scale: float,
    num_inference_steps: int,
    guidance_scale: float,
    generator: torch.Generator,
    device: str,
):
    use_concat = conditioning in {"concat", "concat_text"}
    use_text = conditioning in {"text", "concat_text"}

    cond_latents = source_latents
    if use_concat and booting_noise_scale and booting_noise_scale > 0:
        cond_latents = source_latents + booting_noise_scale * torch.randn(source_latents.shape, generator=generator,                    device=source_latents.device,
                    dtype=source_latents.dtype,)

    if not use_concat:
        # Direct pipeline path (kept for completeness). Most bridge concat models should use the concat loop.
        pipe = SanaPipeline(vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, transformer=teacher, scheduler=scheduler).to(device)
        out = pipe(
            prompt=prompt if use_text else "",
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            latents=source_latents,
            generator=generator,
        )
        # diffusers returns decoded images; we need latents for saving via VAE decode later.
        # So we fall back to concat loop even for non-concat conditioning.
        raise RuntimeError("Direct pipeline mode is not supported for latent output; use --conditioning concat_text")

    if cond_proj is None:
        raise RuntimeError("concat/concat_text requires cond_proj")

    if use_text:
        prompt_hs, prompt_mask = encode_prompts(tokenizer, text_encoder, prompt, device)
        do_cfg = guidance_scale is not None and float(guidance_scale) != 1.0
        if do_cfg:
            uncond_hs, uncond_mask = encode_prompts(tokenizer, text_encoder, "", device)
            encoder_hs = torch.cat([uncond_hs, prompt_hs], dim=0)
            encoder_mask = torch.cat([uncond_mask, prompt_mask], dim=0)
        else:
            encoder_hs, encoder_mask = prompt_hs, prompt_mask
    else:
        encoder_hs, encoder_mask = None, None
        do_cfg = False

    scheduler.set_timesteps(num_inference_steps, device=device)

    latents = source_latents

    autocast_ctx = (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if device == "cuda"
        else contextlib.nullcontext()
    )

    with autocast_ctx:
        for t in scheduler.timesteps:
            model_in = cond_proj(torch.cat([latents, cond_latents], dim=1))
            if do_cfg:
                model_in = torch.cat([model_in, model_in], dim=0)

            t_scalar = t
            if not torch.is_tensor(t_scalar):
                t_scalar = torch.tensor(t_scalar, device=device)
            timesteps = torch.full((model_in.shape[0],), float(t_scalar.item()), device=device, dtype=torch.float32)

            model_out = teacher(
                hidden_states=model_in,
                timestep=timesteps,
                encoder_hidden_states=encoder_hs,
                encoder_attention_mask=encoder_mask,
            ).sample

            if do_cfg:
                uncond, text = model_out.chunk(2)
                model_out = uncond + guidance_scale * (text - uncond)

            step_out = scheduler.step(model_out, t, latents, return_dict=True)
            latents = _get_prev_sample(step_out)

    return latents


def main():
    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)

    model_path = Path(args.model_path)
    data_dir = Path(args.data_dir)
    input_jsonl = data_dir / args.train_jsonl
    output_jsonl = data_dir / args.output_jsonl

    out_img_dir = data_dir / args.new_tar_subdir
    out_img_dir.mkdir(parents=True, exist_ok=True)

    # Read dataset items
    items = []
    with open(input_jsonl, "r") as f:
        for line in f:
            items.append(json.loads(line))
            if args.max_items is not None and len(items) >= args.max_items:
                break

    if not items:
        raise RuntimeError(f"No items found in {input_jsonl}")

    # Load models
    vae = AutoencoderDC.from_pretrained(model_path, subfolder="vae", torch_dtype=torch.float32).to(device)
    text_encoder = Gemma2Model.from_pretrained(model_path, subfolder="text_encoder", torch_dtype=torch.bfloat16).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_path, subfolder="tokenizer")

    base_transformer = SanaTransformer2DModel.from_pretrained(model_path, subfolder="transformer", torch_dtype=torch.bfloat16)

    lora_cfg = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
        lora_dropout=0.0,
    )
    teacher = get_peft_model(base_transformer, lora_cfg)

    lora_file = _resolve_lora_file(args.teacher_lora_path)
    if not lora_file.exists():
        raise FileNotFoundError(f"Teacher LoRA not found: {lora_file}")
    state = torch.load(lora_file, map_location="cpu")
    set_peft_model_state_dict(teacher, state)
    teacher = teacher.to(device)
    teacher.eval()

    # Scheduler
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

    # Conditioning
    use_concat = args.conditioning in {"concat", "concat_text"}
    if not use_concat:
        raise ValueError("This builder currently supports concat/concat_text only (needs cond_proj).")

    cond_proj_path = args.cond_proj_path
    if cond_proj_path is None:
        cand = lora_file.resolve().parent / "cond_proj_weights.bin"
        if cand.exists():
            cond_proj_path = str(cand)
    if cond_proj_path is None or not Path(cond_proj_path).exists():
        raise FileNotFoundError("cond_proj_weights.bin not found; pass --cond_proj_path")

    cond_proj = load_cond_proj(vae, device, cond_proj_path)
    cond_proj.eval()

    # Preprocess
    transform = build_transform(args.resolution)
    gen = torch.Generator(device=device).manual_seed(args.seed)

    out_lines = []
    for i, item in enumerate(items):
        src_rel = _normalize_relpath(item["src"], data_dir)
        prompt = item.get("prompt", "Convert the style to 3D Chibi Style")

        src_path = data_dir / src_rel
        if not src_path.exists():
            raise FileNotFoundError(f"Missing src image: {src_path}")

        # Synthesize target filename: keep src stem for easy trace
        out_name = f"{Path(src_rel).stem}_teacher.png"
        out_rel = str(Path(args.new_tar_subdir) / out_name)
        out_path = data_dir / out_rel

        if out_path.exists() and not args.overwrite:
            # Reuse existing image
            pass
        else:
            x0_img = Image.open(src_path).convert("RGB")
            x0_tensor = transform(x0_img).unsqueeze(0).to(device)
            x0_latents = encode_image_tensor(vae, x0_tensor)

            latents_out = infer_teacher_target_latents(
                teacher=teacher,
                vae=vae,
                tokenizer=tokenizer,
                text_encoder=text_encoder,
                scheduler=scheduler,
                cond_proj=cond_proj,
                conditioning=args.conditioning,
                prompt=prompt,
                source_latents=x0_latents,
                booting_noise_scale=args.booting_noise_scale,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                generator=gen,
                device=device,
            )

            img = latents_to_pil(vae, latents_out)
            img.save(out_path)

        new_item = dict(item)
        # Replace target with teacher synth
        new_item["tar"] = out_rel
        out_lines.append(new_item)

        if (i + 1) % 10 == 0:
            print(f"[{i+1}/{len(items)}] synthesized")

    with open(output_jsonl, "w") as f:
        for obj in out_lines:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print("Done")
    print(f"Wrote jsonl: {output_jsonl}")
    print(f"Synth images dir: {out_img_dir}")


if __name__ == "__main__":
    main()
