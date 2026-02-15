#!/usr/bin/env python
"""infer_distilled_few_steps.py

Load a distilled student (LoRA + optional cond_proj) and run *few-step* inference
using the same concat loop/scheduler behavior as batch_inference_visualize.py.

This is intended for quick sanity checks after distillation checkpoints.

Example:
  python infer_distilled_few_steps.py \
    --model_path /cache/SANA1.5_4.8B_1024px_diffusers \
    --distill_output /cache/sanaoutput/bridge_scm_ladd_distill_3d_chibi \
    --checkpoint latest \
    --input_dir /cache/omnic/3D_Chibi/src \
    --output_dir ./fewstep_out \
    --num_inference_steps 2 \
    --conditioning concat_text \
    --scheduler_mode vibt --vibt_noise_scale 0.0
"""

import argparse
import contextlib
import re
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from diffusers import AutoencoderDC, SanaPipeline
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
from transformers import AutoTokenizer, Gemma2Model

from batch_inference_visualize import (
    _get_prev_sample,
    create_comparison_image,
    decode_latents,
    encode_image,
    encode_prompts,
    load_cond_proj,
    load_scheduler_from_model,
)


def parse_args():
    p = argparse.ArgumentParser(description="Infer distilled bridge model with few steps")

    p.add_argument("--model_path", type=str, required=True)
    p.add_argument(
        "--distill_output",
        type=str,
        required=True,
        help="Path to distillation output dir (contains checkpoint-* or final_checkpoint), or a pytorch_lora_weights.bin file.",
    )
    p.add_argument(
        "--checkpoint",
        type=str,
        default="latest",
        help="Which checkpoint to load if distill_output is a directory: latest | final | checkpoint-XXXX | XXXX",
    )

    p.add_argument("--input_dir", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)

    p.add_argument("--prompt", type=str, default="Convert the style to 3D Chibi Style")

    p.add_argument(
        "--inference_mode",
        type=str,
        default=None,
        choices=["direct", "concat"],
        help="direct: use SanaPipeline(..., latents=source_latents); concat: custom loop that injects concat conditioning every step. If omitted, inferred from --conditioning.",
    )

    p.add_argument(
        "--conditioning",
        type=str,
        default="concat_text",
        choices=["text", "concat", "concat_text"],
        help="Use concat/concat_text for distilled bridge models trained with cond_proj.",
    )
    p.add_argument(
        "--cond_proj_path",
        type=str,
        default=None,
        help="Optional explicit path to cond_proj_weights.bin (auto-detected next to checkpoint if omitted).",
    )
    p.add_argument(
        "--booting_noise_scale",
        type=float,
        default=0.0,
        help="Std of Gaussian noise added to x0_cond for concat conditioning.",
    )

    p.add_argument("--num_inference_steps", type=int, default=2)
    p.add_argument("--guidance_scale", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_images", type=int, default=9)

    p.add_argument(
        "--scheduler_mode",
        type=str,
        default="model",
        choices=["model", "vibt"],
        help="model: checkpoint scheduler; vibt: wrap with ViBTScheduler",
    )
    p.add_argument("--vibt_noise_scale", type=float, default=0.0)
    p.add_argument("--vibt_shift_gamma", type=float, default=5.0)

    p.add_argument("--resolution", type=int, default=1024)

    p.add_argument("--lora_rank", type=int, default=128)
    p.add_argument("--lora_alpha", type=int, default=128)

    return p.parse_args()


def _resolve_ckpt_file(distill_output: Path, checkpoint: str) -> Path:
    """Resolve to a concrete pytorch_lora_weights.bin path."""
    if distill_output.is_file():
        return distill_output

    if not distill_output.exists():
        raise FileNotFoundError(f"distill_output not found: {distill_output}")

    # Directly inside dir
    direct = distill_output / "pytorch_lora_weights.bin"
    if direct.exists():
        return direct

    # final_checkpoint
    final = distill_output / "final_checkpoint" / "pytorch_lora_weights.bin"
    if checkpoint in {"final", "final_checkpoint"} and final.exists():
        return final
    if final.exists() and checkpoint in {"final", "final_checkpoint"}:
        return final

    # checkpoint-* selection
    def parse_step(p: Path) -> int:
        m = re.match(r"checkpoint-(\d+)$", p.name)
        return int(m.group(1)) if m else -1

    ckpt_dir = None
    if checkpoint == "latest":
        candidates = [p for p in distill_output.glob("checkpoint-*") if (p / "pytorch_lora_weights.bin").exists()]
        if not candidates and final.exists():
            return final
        if not candidates:
            raise FileNotFoundError(f"No checkpoint-* or final_checkpoint found under: {distill_output}")
        ckpt_dir = sorted(candidates, key=parse_step)[-1]
    else:
        ck = checkpoint
        if ck.isdigit():
            ck = f"checkpoint-{ck}"
        ckpt_dir = distill_output / ck
        if ckpt_dir.is_dir() and (ckpt_dir / "pytorch_lora_weights.bin").exists():
            pass
        elif final.exists() and checkpoint in {"final", "final_checkpoint"}:
            return final
        else:
            raise FileNotFoundError(f"Requested checkpoint not found: {ckpt_dir}")

    return ckpt_dir / "pytorch_lora_weights.bin"


def _resolve_cond_proj(args_cond_proj: str | None, lora_file: Path) -> Path | None:
    if args_cond_proj is not None:
        p = Path(args_cond_proj)
        return p

    cand = lora_file.parent / "cond_proj_weights.bin"
    if cand.exists():
        return cand

    # If lora_file is in .../final_checkpoint/, sometimes cond_proj is stored there too.
    cand2 = lora_file.parent.parent / "cond_proj_weights.bin"
    if cand2.exists():
        return cand2

    return None


def latents_to_pil(vae, latents: torch.Tensor) -> Image.Image:
    latents = latents.to(torch.float32)
    img_tensor = decode_latents(vae, latents).clamp(-1, 1)
    img = (img_tensor[0].detach().float().cpu() + 1) / 2
    img = img.permute(1, 2, 0).numpy()
    img = (img * 255).round().clip(0, 255).astype(np.uint8)
    return Image.fromarray(img)


def main():
    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.output_dir)
    (out_dir / "comparisons").mkdir(parents=True, exist_ok=True)
    (out_dir / "results_only").mkdir(parents=True, exist_ok=True)

    # Resolve checkpoint files
    lora_file = _resolve_ckpt_file(Path(args.distill_output), args.checkpoint)
    if not lora_file.exists():
        raise FileNotFoundError(f"LoRA weights not found: {lora_file}")

    cond_proj_file = _resolve_cond_proj(args.cond_proj_path, lora_file)

    use_concat = args.conditioning in {"concat", "concat_text"}
    use_text = args.conditioning in {"text", "concat_text"}
    if use_concat and cond_proj_file is None:
        raise FileNotFoundError(
            "concat/concat_text requires cond_proj_weights.bin. "
            "Pass --cond_proj_path or place it next to the checkpoint."
        )

    inference_mode = args.inference_mode
    if inference_mode is None:
        inference_mode = "concat" if use_concat else "direct"
    if inference_mode == "concat" and not use_concat:
        raise ValueError("--inference_mode=concat requires --conditioning=concat or --conditioning=concat_text")
    if inference_mode == "direct" and args.conditioning != "text":
        raise ValueError("--inference_mode=direct currently supports only --conditioning=text")

    # Load base components
    vae = AutoencoderDC.from_pretrained(args.model_path, subfolder="vae", torch_dtype=torch.float32).to(device)
    text_encoder = Gemma2Model.from_pretrained(args.model_path, subfolder="text_encoder", torch_dtype=torch.bfloat16).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, subfolder="tokenizer")

    from diffusers import SanaTransformer2DModel

    transformer = SanaTransformer2DModel.from_pretrained(args.model_path, subfolder="transformer", torch_dtype=torch.bfloat16)

    # Attach LoRA
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
        lora_dropout=0.0,
    )
    transformer = get_peft_model(transformer, lora_config)
    state = torch.load(lora_file, map_location="cpu")
    if not any("lora" in k.lower() for k in state.keys()):
        raise ValueError(f"LoRA state dict does not look like LoRA weights: {lora_file}")
    set_peft_model_state_dict(transformer, state)
    transformer = transformer.to(device)
    transformer.eval()

    # Scheduler
    scheduler = load_scheduler_from_model(args.model_path)
    if args.scheduler_mode == "vibt":
        from diffusers.schedulers import UniPCMultistepScheduler
        from vibt.scheduler import ViBTScheduler

        if not isinstance(scheduler, UniPCMultistepScheduler):
            scheduler = UniPCMultistepScheduler.from_config(scheduler.config)
        scheduler = ViBTScheduler.from_scheduler(
            scheduler,
            noise_scale=args.vibt_noise_scale,
            shift_gamma=args.vibt_shift_gamma,
        )

    # Build pipeline (keeps component wiring consistent with batch_inference_visualize)
    pipeline = SanaPipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        transformer=transformer,
        scheduler=scheduler,
    ).to(device)
    pipeline.vae.eval()
    pipeline.text_encoder.eval()
    pipeline.transformer.eval()

    # Cond proj
    cond_proj = None
    if use_concat:
        cond_proj = load_cond_proj(vae, device, str(cond_proj_file) if cond_proj_file is not None else None)
        cond_proj = cond_proj.to(device=device, dtype=transformer.dtype)
        cond_proj.eval()

    print("=" * 60)
    print("Distilled few-step inference")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"LoRA: {lora_file}")
    print(f"cond_proj: {str(cond_proj_file) if cond_proj_file is not None else '(none)'}")
    print(f"conditioning: {args.conditioning}")
    print(f"inference_mode: {inference_mode}")
    print(f"scheduler_mode: {args.scheduler_mode} ({type(pipeline.scheduler).__name__})")
    if args.scheduler_mode == "vibt":
        print(f"  vibt_noise_scale={args.vibt_noise_scale} vibt_shift_gamma={args.vibt_shift_gamma}")

    # Files
    input_dir = Path(args.input_dir)
    exts = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]
    image_files = []
    for ext in exts:
        image_files += list(input_dir.glob(f"*{ext}"))
        image_files += list(input_dir.glob(f"*{ext.upper()}"))
    image_files = sorted(set(image_files))

    if args.max_images is not None:
        image_files = image_files[: int(args.max_images)]

    if not image_files:
        raise RuntimeError(f"No images found under: {input_dir}")

    generator = torch.Generator(device=device).manual_seed(int(args.seed))

    autocast_ctx = torch.autocast("cuda", dtype=torch.bfloat16) if device == "cuda" else contextlib.nullcontext()

    for idx, image_path in enumerate(image_files):
        source_latents, source_img = encode_image(vae, str(image_path), device, resolution=args.resolution)

        cond_latents = source_latents
        if use_concat and args.booting_noise_scale and args.booting_noise_scale > 0:
            noise = torch.randn(
                source_latents.shape,
                generator=generator,
                device=source_latents.device,
                dtype=source_latents.dtype,
            )
            cond_latents = source_latents + float(args.booting_noise_scale) * noise

        with torch.inference_mode():
            with autocast_ctx:
                if inference_mode == "direct":
                    out = pipeline(
                        prompt=args.prompt,
                        num_inference_steps=int(args.num_inference_steps),
                        guidance_scale=float(args.guidance_scale),
                        latents=source_latents,
                        generator=generator,
                    )
                    result_img = out.images[0]
                else:
                    if use_text:
                        prompt_hs, prompt_mask = encode_prompts(tokenizer, text_encoder, args.prompt, device)
                        do_cfg = args.guidance_scale is not None and float(args.guidance_scale) != 1.0
                        if do_cfg:
                            uncond_hs, uncond_mask = encode_prompts(tokenizer, text_encoder, "", device)
                            encoder_hs = torch.cat([uncond_hs, prompt_hs], dim=0)
                            encoder_mask = torch.cat([uncond_mask, prompt_mask], dim=0)
                        else:
                            encoder_hs, encoder_mask = prompt_hs, prompt_mask
                    else:
                        encoder_hs, encoder_mask = None, None
                        do_cfg = False

                    scheduler = pipeline.scheduler
                    scheduler.set_timesteps(int(args.num_inference_steps), device=device)
                    latents = source_latents

                    for t in scheduler.timesteps:
                        model_in = cond_proj(torch.cat([latents, cond_latents], dim=1))
                        if do_cfg:
                            model_in = torch.cat([model_in, model_in], dim=0)

                        t_scalar = t
                        if not torch.is_tensor(t_scalar):
                            t_scalar = torch.tensor(t_scalar, device=device)
                        timesteps = torch.full(
                            (model_in.shape[0],),
                            float(t_scalar.item()),
                            device=device,
                            dtype=torch.float32,
                        )

                        model_out = pipeline.transformer(
                            hidden_states=model_in,
                            timestep=timesteps,
                            encoder_hidden_states=encoder_hs,
                            encoder_attention_mask=encoder_mask,
                        ).sample

                        if do_cfg:
                            uncond, text = model_out.chunk(2)
                            model_out = uncond + float(args.guidance_scale) * (text - uncond)

                        step_out = scheduler.step(model_out, t, latents, return_dict=True)
                        latents = _get_prev_sample(step_out)

                    result_img = latents_to_pil(vae, latents)

        # Save
        result_filename = f"{image_path.stem}_result.png"
        result_img.save(out_dir / "results_only" / result_filename)

        comp = create_comparison_image(source_img, result_img, args.prompt)
        comp_filename = f"{image_path.stem}_comparison.png"
        comp.save(out_dir / "comparisons" / comp_filename)

        print(f"[{idx+1}/{len(image_files)}] {image_path.name} -> {result_filename}")

    print("Done")
    print(f"Output: {out_dir}")


if __name__ == "__main__":
    main()
