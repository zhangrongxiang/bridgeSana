#!/usr/bin/env python
"""
Batch Bridge Inference with Visualization
Loads checkpoint, processes multiple images, and creates side-by-side comparisons
"""

import argparse
import os
from pathlib import Path
import json
import torch
from PIL import Image, ImageDraw, ImageFont
from diffusers import SanaPipeline, AutoencoderDC
import diffusers
from transformers import AutoTokenizer, Gemma2Model
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
from torchvision import transforms
from tqdm import tqdm
import numpy as np
import contextlib
import torch.nn as nn


def parse_args():
    parser = argparse.ArgumentParser(description="Batch Bridge inference with visualization")
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
        help="Path to trained LoRA checkpoint",
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing source images",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./inference_results",
        help="Output directory for results",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Convert the style to 3D Chibi Style",
        help="Text prompt for translation",
    )
    parser.add_argument(
        "--inference_mode",
        type=str,
        default="direct",
        choices=["direct", "concat"],
        help="Inference backend: 'direct' uses SanaPipeline(..., latents=source_latents); 'concat' uses a custom loop that injects concat conditioning every step.",
    )
    parser.add_argument(
        "--conditioning",
        type=str,
        default="text",
        choices=["text", "concat", "concat_text"],
        help="Conditioning mode. 'concat'/'concat_text' expects a cond_proj_weights.bin saved from training.",
    )
    parser.add_argument(
        "--cond_proj_path",
        type=str,
        default=None,
        help="Path to cond_proj_weights.bin (optional; auto-detected from LoRA checkpoint directory if omitted).",
    )
    parser.add_argument(
        "--booting_noise_scale",
        type=float,
        default=0.0,
        help="Std of Gaussian booting noise added to source latents for concat conditioning: x0_cond = x0 + scale * eps. 0 disables.",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=28,
        help="Number of inference steps",
    )
    parser.add_argument(
        "--scheduler_mode",
        type=str,
        default="model",
        choices=["model", "vibt"],
        help="Scheduler to use: 'model' uses the checkpoint scheduler; 'vibt' uses ViBT custom scheduler.",
    )
    parser.add_argument(
        "--vibt_noise_scale",
        type=float,
        default=1.0,
        help="ViBT scheduler noise scale (only used when --scheduler_mode=vibt)",
    )
    parser.add_argument(
        "--vibt_shift_gamma",
        type=float,
        default=5.0,
        help="ViBT scheduler flow_shift (only used when --scheduler_mode=vibt)",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=1.0,
        help="Guidance scale",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--max_images",
        type=int,
        default=None,
        help="Maximum number of images to process (None for all)",
    )
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=128,
        help="LoRA rank (must match training)",
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=128,
        help="LoRA alpha (must match training)",
    )
    return parser.parse_args()


def encode_image(vae, image_path, device, resolution=1024):
    """Encode image to latent space"""
    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')

    transform = transforms.Compose([
        transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(resolution),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    image_tensor = transform(image).unsqueeze(0).to(device)

    # Encode to latents
    with torch.no_grad():
        encoded = vae.encode(image_tensor)

        # Support different VAE output formats
        if hasattr(encoded, "latent_dist"):
            latents = encoded.latent_dist.mode()
        elif hasattr(encoded, "latents"):
            latents = encoded.latents
        elif hasattr(encoded, "latent"):
            latents = encoded.latent
        else:
            latents = encoded

        # Apply scaling factor if available
        scaling_factor = getattr(getattr(vae, "config", None), "scaling_factor", None)
        if scaling_factor is not None:
            latents = latents * scaling_factor

    return latents, image


def decode_latents(vae, latents):
    scaling_factor = getattr(getattr(vae, "config", None), "scaling_factor", None)
    if scaling_factor is not None:
        latents = latents / scaling_factor
    decoded = vae.decode(latents)
    if hasattr(decoded, "sample"):
        return decoded.sample
    return decoded


@torch.no_grad()
def encode_prompts(tokenizer, text_encoder, prompt, device):
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


def load_cond_proj(vae, device, cond_proj_path: str | None):
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

    return cond_proj.to(device)


def _get_prev_sample(step_out):
    if hasattr(step_out, "prev_sample"):
        return step_out.prev_sample
    if isinstance(step_out, dict) and "prev_sample" in step_out:
        return step_out["prev_sample"]
    if isinstance(step_out, (tuple, list)) and len(step_out) > 0:
        return step_out[0]
    raise TypeError(f"Unsupported scheduler.step output type: {type(step_out)}")


def create_comparison_image(source_img, result_img, prompt, add_text=True):
    """Create side-by-side comparison image"""
    # Ensure both images are the same size
    width, height = source_img.size

    # Create canvas for side-by-side comparison
    if add_text:
        text_height = 60
        canvas = Image.new('RGB', (width * 2, height + text_height), 'white')
    else:
        canvas = Image.new('RGB', (width * 2, height), 'white')

    # Paste images
    canvas.paste(source_img, (0, 0))
    canvas.paste(result_img, (width, 0))

    # Add text labels
    if add_text:
        draw = ImageDraw.Draw(canvas)
        try:
            # Try to use a nice font
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
        except:
            # Fallback to default font
            font = ImageFont.load_default()

        # Add labels
        draw.text((10, height + 10), "Source Image", fill='black', font=font)
        draw.text((width + 10, height + 10), f"Result: {prompt[:30]}...", fill='black', font=font)

    return canvas


def load_scheduler_from_model(model_path: str):
    """Load the exact scheduler class specified by the model checkpoint.

    Training-time validation gets the scheduler via SanaPipeline.from_pretrained (model_index.json).
    Here we emulate that behavior without hard-coding a specific scheduler class.
    """
    scheduler_cfg = Path(model_path) / "scheduler" / "scheduler_config.json"
    if scheduler_cfg.exists():
        with open(scheduler_cfg, "r") as f:
            cfg = json.load(f)
        class_name = cfg.get("_class_name")
        if class_name and hasattr(diffusers, class_name):
            scheduler_cls = getattr(diffusers, class_name)
            return scheduler_cls.from_pretrained(model_path, subfolder="scheduler")

    # Fallback: let SanaPipeline resolve the scheduler (heavier, but correct)
    base_pipe = SanaPipeline.from_pretrained(model_path, torch_dtype=torch.bfloat16)
    scheduler = base_pipe.scheduler
    del base_pipe
    return scheduler


def main():
    args = parse_args()

    print("=" * 60)
    print("Batch Bridge Inference with Visualization")
    print("=" * 60)

    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "comparisons").mkdir(exist_ok=True)
    (output_dir / "results_only").mkdir(exist_ok=True)

    print(f"\n📁 Output directory: {output_dir}")

    # Load models
    print(f"\n🔄 Loading base model from {args.model_path}")

    # Load VAE
    vae = AutoencoderDC.from_pretrained(
        args.model_path,
        subfolder="vae",
        torch_dtype=torch.float32,
    ).to(device)

    # Load text encoder
    text_encoder = Gemma2Model.from_pretrained(
        args.model_path,
        subfolder="text_encoder",
        torch_dtype=torch.bfloat16,
    ).to(device)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        subfolder="tokenizer",
    )

    # Load transformer
    from diffusers import SanaTransformer2DModel
    transformer = SanaTransformer2DModel.from_pretrained(
        args.model_path,
        subfolder="transformer",
        torch_dtype=torch.bfloat16,
    )

    print(f"✅ Base model loaded")

    # Load LoRA weights
    print(f"\n🔄 Loading LoRA weights from {args.lora_path}")
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
        lora_dropout=0.0,
    )
    transformer = get_peft_model(transformer, lora_config)

    lora_weights = torch.load(args.lora_path, map_location="cpu")
    set_peft_model_state_dict(transformer, lora_weights)
    transformer = transformer.to(device)

    print(f"✅ LoRA weights loaded")

    # Create pipeline
    print(f"\n🔄 Creating inference pipeline")
    scheduler = load_scheduler_from_model(args.model_path)

    if args.scheduler_mode == "vibt":
        print("   Scheduler mode: vibt")
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
                # Try converting the model scheduler config into a UniPC base first
                base_unipc = UniPCMultistepScheduler.from_config(scheduler.config)
                scheduler = ViBTScheduler.from_scheduler(
                    base_unipc,
                    noise_scale=args.vibt_noise_scale,
                    shift_gamma=args.vibt_shift_gamma,
                )
        except Exception as e:
            print(f"   ⚠️  Failed to enable ViBT scheduler, falling back to model scheduler: {e}")

    print(f"   Scheduler: {type(scheduler).__name__}")
    pipeline = SanaPipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        transformer=transformer,
        scheduler=scheduler,
    )
    pipeline = pipeline.to(device)
    pipeline.vae.eval()
    pipeline.text_encoder.eval()
    pipeline.transformer.eval()
    print(f"✅ Pipeline ready")

    use_concat_backend = args.inference_mode == "concat"
    use_concat_conditioning = args.conditioning in {"concat", "concat_text"}
    if use_concat_backend and not use_concat_conditioning:
        raise ValueError("--inference_mode=concat requires --conditioning=concat or --conditioning=concat_text")

    cond_proj = None
    cond_proj_path = args.cond_proj_path
    if use_concat_backend and cond_proj_path is None:
        # Try auto-detect in the same directory as LoRA weights
        cand = Path(args.lora_path).resolve().parent / "cond_proj_weights.bin"
        if cand.exists():
            cond_proj_path = str(cand)
    if use_concat_backend:
        if cond_proj_path is None or not Path(cond_proj_path).exists():
            raise FileNotFoundError(
                "conditioning=concat requires cond_proj_weights.bin. "
                "Pass --cond_proj_path or place it next to the LoRA checkpoint."
            )
        print("   Inference mode: concat")
        print(f"   Concat conditioning: {args.conditioning}")
        print(f"   cond_proj_path: {cond_proj_path}")
        cond_proj = load_cond_proj(vae, device, cond_proj_path)
        cond_proj.eval()
    else:
        print("   Inference mode: direct")

    # Find all images in input directory
    input_dir = Path(args.input_dir)
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']
    image_files = []
    for ext in image_extensions:
        image_files.extend(list(input_dir.glob(f"*{ext}")))
        image_files.extend(list(input_dir.glob(f"*{ext.upper()}")))

    image_files = sorted(image_files)

    if args.max_images:
        image_files = image_files[:args.max_images]

    print(f"\n📸 Found {len(image_files)} images to process")

    if len(image_files) == 0:
        print(f"❌ No images found in {input_dir}")
        return

    # Set random seed
    generator = torch.Generator(device=device).manual_seed(args.seed)

    # Process images
    print(f"\n🎨 Starting inference...")
    print(f"   Prompt: {args.prompt}")
    print(f"   Steps: {args.num_inference_steps}")
    print(f"   Guidance scale: {args.guidance_scale}")
    print()

    results = []

    for idx, image_path in enumerate(tqdm(image_files, desc="Processing images")):
        try:
            # Encode source image
            source_latents, source_img = encode_image(vae, image_path, device)

            cond_latents = source_latents
            if use_concat_backend and args.booting_noise_scale and args.booting_noise_scale > 0:
                noise = torch.randn(
                    source_latents.shape,
                    generator=generator,
                    device=source_latents.device,
                    dtype=source_latents.dtype,
                )
                cond_latents = source_latents + args.booting_noise_scale * noise

            with torch.inference_mode():
                autocast_ctx = (
                    torch.autocast("cuda", dtype=torch.bfloat16)
                    if device == "cuda"
                    else contextlib.nullcontext()
                )
                with autocast_ctx:
                    if not use_concat_backend:
                        output = pipeline(
                            prompt=args.prompt,
                            num_inference_steps=args.num_inference_steps,
                            guidance_scale=args.guidance_scale,
                            latents=source_latents,
                            generator=generator,
                        )
                        result_img = output.images[0]
                    else:
                        # Minimal denoising loop that injects concat conditioning every step
                        prompt_hs, prompt_mask = encode_prompts(tokenizer, text_encoder, args.prompt, device)
                        # CFG support (optional)
                        do_cfg = args.guidance_scale is not None and args.guidance_scale != 1.0
                        if do_cfg:
                            uncond_hs, uncond_mask = encode_prompts(tokenizer, text_encoder, "", device)
                            encoder_hs = torch.cat([uncond_hs, prompt_hs], dim=0)
                            encoder_mask = torch.cat([uncond_mask, prompt_mask], dim=0)
                        else:
                            encoder_hs, encoder_mask = prompt_hs, prompt_mask

                        scheduler = pipeline.scheduler
                        scheduler.set_timesteps(args.num_inference_steps, device=device)
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
                                t_scalar.item(),
                                device=device,
                                dtype=torch.float32,
                            )

                            model_out = transformer(
                                hidden_states=model_in,
                                timestep=timesteps,
                                encoder_hidden_states=encoder_hs,
                                encoder_attention_mask=encoder_mask,
                            ).sample

                            if do_cfg:
                                noise_uncond, noise_text = model_out.chunk(2)
                                model_out = noise_uncond + args.guidance_scale * (noise_text - noise_uncond)

                            step_out = scheduler.step(model_out, t, latents, return_dict=True)
                            latents = _get_prev_sample(step_out)

                        img_tensor = decode_latents(vae, latents).clamp(-1, 1)
                        img = (img_tensor[0].detach().float().cpu() + 1) / 2
                        img = img.permute(1, 2, 0).numpy()
                        img = (img * 255).round().clip(0, 255).astype(np.uint8)
                        result_img = Image.fromarray(img)

            # Save individual result
            result_filename = f"{image_path.stem}_result.png"
            result_img.save(output_dir / "results_only" / result_filename)

            # Create and save comparison
            comparison = create_comparison_image(source_img, result_img, args.prompt)
            comparison_filename = f"{image_path.stem}_comparison.png"
            comparison.save(output_dir / "comparisons" / comparison_filename)

            results.append({
                'source': image_path.name,
                'result': result_filename,
                'comparison': comparison_filename,
            })

        except Exception as e:
            print(f"\n❌ Error processing {image_path.name}: {e}")
            continue

    print(f"\n✅ Processing complete!")
    print(f"   Processed: {len(results)}/{len(image_files)} images")
    print(f"   Results saved to: {output_dir}")
    print(f"   - Comparisons: {output_dir / 'comparisons'}")
    print(f"   - Results only: {output_dir / 'results_only'}")

    # Create summary grid
    if len(results) > 0:
        print(f"\n🖼️  Creating summary grid...")
        create_summary_grid(results, output_dir, args.prompt)

    print(f"\n{'=' * 60}")
    print(f"✅ All done!")
    print(f"{'=' * 60}")


def create_summary_grid(results, output_dir, prompt, max_images=16):
    """Create a grid showing multiple comparisons"""
    from math import ceil, sqrt

    # Limit to max_images
    results = results[:max_images]
    n_images = len(results)

    # Calculate grid size
    n_cols = min(4, ceil(sqrt(n_images)))
    n_rows = ceil(n_images / n_cols)

    # Load first image to get size
    first_comparison = Image.open(output_dir / "comparisons" / results[0]['comparison'])
    img_width, img_height = first_comparison.size

    # Scale down for grid
    scale = 0.5
    thumb_width = int(img_width * scale)
    thumb_height = int(img_height * scale)

    # Create grid canvas
    grid_width = thumb_width * n_cols
    grid_height = thumb_height * n_rows + 80  # Extra space for title
    grid = Image.new('RGB', (grid_width, grid_height), 'white')

    # Add title
    draw = ImageDraw.Draw(grid)
    try:
        title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
    except:
        title_font = ImageFont.load_default()

    title_text = f"Bridge Inference Results - {prompt}"
    draw.text((20, 20), title_text, fill='black', font=title_font)

    # Add images to grid
    for idx, result in enumerate(results):
        row = idx // n_cols
        col = idx % n_cols

        # Load and resize comparison image
        comp_img = Image.open(output_dir / "comparisons" / result['comparison'])
        comp_img = comp_img.resize((thumb_width, thumb_height), Image.Resampling.LANCZOS)

        # Paste into grid
        x = col * thumb_width
        y = row * thumb_height + 80
        grid.paste(comp_img, (x, y))

    # Save grid
    grid_path = output_dir / "summary_grid.png"
    grid.save(grid_path)
    print(f"   ✅ Summary grid saved: {grid_path}")


if __name__ == "__main__":
    main()



