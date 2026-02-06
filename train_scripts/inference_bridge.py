#!/usr/bin/env python
"""
Bridge Inference Script for Sana Model
Performs image-to-image translation using trained Bridge model
"""

import argparse
import torch
from pathlib import Path
from PIL import Image
from diffusers import SanaPipeline, AutoencoderDC
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from transformers import AutoTokenizer, Gemma2Model
from peft import PeftModel


def parse_args():
    parser = argparse.ArgumentParser(description="Bridge inference script")
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
        "--source_image",
        type=str,
        required=True,
        help="Path to source image",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Text prompt for translation",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="output.png",
        help="Output image path",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=28,
        help="Number of inference steps",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=4.5,
        help="Guidance scale",
    )
    parser.add_argument(
        "--noise_scale",
        type=float,
        default=1.0,
        help="Noise scale for Bridge sampling",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    return parser.parse_args()


def encode_image(vae, image_path, device):
    """Encode image to latent space"""
    from torchvision import transforms

    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')

    transform = transforms.Compose([
        transforms.Resize(1024, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(1024),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    image_tensor = transform(image).unsqueeze(0).to(device)

    # Encode to latents
    with torch.no_grad():
        encoded = vae.encode(image_tensor)

        # Support both older and newer diffusers VAE outputs
        if hasattr(encoded, "latent_dist"):
            latents = encoded.latent_dist.mode()
        elif hasattr(encoded, "latents"):
            latents = encoded.latents
        elif hasattr(encoded, "latent"):
            latents = encoded.latent
        else:
            latents = encoded

        # Apply scaling factor if available, to mirror training / SanaPipeline
        scaling_factor = getattr(getattr(vae, "config", None), "scaling_factor", None)
        if scaling_factor is not None:
            latents = latents * scaling_factor

    return latents


def main():
    args = parse_args()

    print(f"Loading base model from {args.model_path}")
    device = "cuda" if torch.cuda.is_available() else "cpu"

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

    # Load transformer with LoRA
    from diffusers import SanaTransformer2DModel
    transformer = SanaTransformer2DModel.from_pretrained(
        args.model_path,
        subfolder="transformer",
        torch_dtype=torch.bfloat16,
    )

    # Load LoRA weights
    print(f"Loading LoRA weights from {args.lora_path}")
    lora_weights = torch.load(args.lora_path, map_location="cpu")

    # Apply LoRA weights to transformer
    from peft import LoraConfig, get_peft_model
    lora_config = LoraConfig(
        r=128,
        lora_alpha=128,
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
        lora_dropout=0.0,
    )
    transformer = get_peft_model(transformer, lora_config)
    transformer.load_state_dict(lora_weights, strict=False)
    transformer = transformer.to(device)

    print("Models loaded successfully")

    # Encode source image to latents
    print(f"Encoding source image: {args.source_image}")
    source_latents = encode_image(vae, args.source_image, device)

    # Create pipeline
    pipeline = SanaPipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        transformer=transformer,
        scheduler=FlowMatchEulerDiscreteScheduler.from_pretrained(
            args.model_path,
            subfolder="scheduler",
        ),
    )
    pipeline = pipeline.to(device)

    # Set random seed
    generator = torch.Generator(device=device).manual_seed(args.seed)

    # Run inference
    print(f"Running Bridge inference with prompt: {args.prompt}")
    print(f"  Steps: {args.num_inference_steps}")
    print(f"  Guidance scale: {args.guidance_scale}")
    print(f"  Noise scale: {args.noise_scale}")

    with torch.autocast("cuda", dtype=torch.bfloat16):
        output = pipeline(
            prompt=args.prompt,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            latents=source_latents,  # Pass source latents for Bridge
            generator=generator,
        )

    # Save output image
    output_image = output.images[0]
    output_image.save(args.output_path)
    print(f"Output saved to: {args.output_path}")


if __name__ == "__main__":
    main()

