#!/usr/bin/env python
# Bridge Training Script for Sana Model
# Based on ViBT (Vision Bridge Transformer) algorithm

import argparse
import json
import logging
import math
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers import AutoencoderDC, SanaPipeline, SanaTransformer2DModel
from diffusers.optimization import get_scheduler
from diffusers.training_utils import cast_training_params
from diffusers.utils import check_min_version, is_wandb_available
from huggingface_hub import create_repo, upload_folder
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import AutoTokenizer, Gemma2Model
from vibt.scheduler import ViBTScheduler

if is_wandb_available():
    import wandb

check_min_version("0.32.0.dev0")

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Bridge training script for Sana model")

    # Model arguments
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        required=True,
        help="Path to pretrained Sana model",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        help="Revision of pretrained model",
    )

    # Dataset arguments
    parser.add_argument(
        "--train_data_dir",
        type=str,
        required=True,
        help="Path to training data directory containing train.jsonl",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=1024,
        help="Image resolution for training",
    )

    # Training arguments
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=1,
        help="Batch size for training",
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=20000,
        help="Total number of training steps",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before backward pass",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help="Learning rate scheduler",
    )
    parser.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=500,
        help="Number of warmup steps",
    )

    # LoRA arguments
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
        "--lora_dropout",
        type=float,
        default=0.0,
        help="LoRA dropout",
    )

    # Bridge-specific arguments
    parser.add_argument(
        "--noise_scale",
        type=float,
        default=1.0,
        help="Noise scale for Bridge sampling (s parameter)",
    )
    parser.add_argument(
        "--use_stabilized_velocity",
        action="store_true",
        help="Use stabilized velocity matching (recommended)",
    )

    # Output arguments
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for checkpoints",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help="Save checkpoint every X steps",
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=10,
        help="Log every X steps",
    )

    # Validation arguments
    parser.add_argument(
        "--validation_prompt",
        type=str,
        default=None,
        help="Prompt for validation",
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=500,
        help="Run validation every X steps",
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=4,
        help="Number of validation images to generate",
    )

    # Misc arguments
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed",
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="bf16",
        choices=["no", "fp16", "bf16"],
        help="Mixed precision training",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help="Reporting tool (tensorboard, wandb)",
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help="Number of dataloader workers",
    )

    args = parser.parse_args()
    return args


class BridgeDataset(Dataset):
    """
    Dataset for Bridge training with source-target image pairs.
    Expects a JSONL file with format:
    {"src": "path/to/source.png", "tar": "path/to/target.png", "prompt": "description"}
    """

    def __init__(self, data_dir, resolution=1024):
        self.data_dir = Path(data_dir)
        self.resolution = resolution

        # Load JSONL metadata
        jsonl_path = self.data_dir / "train.jsonl"
        self.data = []
        with open(jsonl_path, 'r') as f:
            for line in f:
                self.data.append(json.loads(line))

        logger.info(f"Loaded {len(self.data)} image pairs from {jsonl_path}")

        # Image transforms
        self.transform = transforms.Compose([
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # Load source and target images
        # Handle both absolute and relative paths
        src_rel = item['src']
        tar_rel = item['tar']

        # Remove leading directory name if it matches the data_dir name
        data_dir_name = self.data_dir.name
        if src_rel.startswith(data_dir_name + '/'):
            src_rel = src_rel[len(data_dir_name) + 1:]
        if tar_rel.startswith(data_dir_name + '/'):
            tar_rel = tar_rel[len(data_dir_name) + 1:]

        src_path = self.data_dir / src_rel
        tar_path = self.data_dir / tar_rel
        prompt = item['prompt']

        src_image = Image.open(src_path).convert('RGB')
        tar_image = Image.open(tar_path).convert('RGB')

        # Apply transforms
        src_tensor = self.transform(src_image)
        tar_tensor = self.transform(tar_image)

        return {
            'source_images': src_tensor,
            'target_images': tar_tensor,
            # 'prompts': prompt,
            'prompts': "Convert the style to 3D Chibi Style",
        }


def collate_fn(examples):
    """Collate function for DataLoader"""
    source_images = torch.stack([example['source_images'] for example in examples])
    target_images = torch.stack([example['target_images'] for example in examples])
    prompts = [example['prompts'] for example in examples]

    return {
        'source_images': source_images,
        'target_images': target_images,
        'prompts': prompts,
    }


def encode_images(vae, images):
    """Encode images to latent space"""
    with torch.no_grad():
        encoded = vae.encode(images)

        # Support both older and newer diffusers VAE outputs
        # 1) Distribution-style output with `latent_dist`
        if hasattr(encoded, "latent_dist"):
            latents = encoded.latent_dist.mode()
        # 2) Output wrapper with `latents` tensor
        elif hasattr(encoded, "latents"):
            latents = encoded.latents
        # 3) EncoderOutput with `latent` (Sana / AutoencoderDC pattern)
        elif hasattr(encoded, "latent"):
            latents = encoded.latent
        # 3) Assume tensor-like output
        else:
            latents = encoded

        # Apply scaling factor if defined (matches DreamBooth / SanaPipeline behavior)
        scaling_factor = getattr(getattr(vae, "config", None), "scaling_factor", None)
        if scaling_factor is not None:
            latents = latents * scaling_factor
    return latents


def compute_bridge_loss(
    model_pred,
    target_velocity,
    x0,
    x1,
    t,
    use_stabilized=True,
):
    """
    Compute Bridge training loss with optional stabilized velocity matching.

    Args:
        model_pred: Model prediction (velocity)
        target_velocity: Target velocity u_t = (x1 - x_t) / (1 - t)
        x0: Source latent
        x1: Target latent
        t: Timestep [0, 1]
        use_stabilized: Whether to use stabilized velocity matching
    """
    if not use_stabilized:
        # Standard velocity matching loss
        loss = F.mse_loss(model_pred, target_velocity, reduction="mean")
    else:
        # Stabilized velocity matching (ViBT paper Eq. 4-5)
        # Compute normalization factor α
        batch_size = x0.shape[0]
        D = x0[0].numel()  # Latent dimension

        # ||x1 - x0||^2 for each sample in batch
        diff_norm_sq = torch.sum((x1 - x0).reshape(batch_size, -1) ** 2, dim=1, keepdim=True)

        # α^2 = 1 + [t * D] / [(1-t) * ||x1-x0||^2]
        # Add small epsilon to avoid division by zero
        alpha_sq = 1.0 + (t * D) / ((1 - t) * diff_norm_sq + 1e-8)
        alpha = torch.sqrt(alpha_sq).reshape(-1, 1, 1, 1)

        # Normalize both predictions and targets
        normalized_pred = model_pred / alpha
        normalized_target = target_velocity / alpha

        loss = F.mse_loss(normalized_pred, normalized_target, reduction="mean")

    return loss


def main():
    args = parse_args()

    # Setup accelerator
    logging_dir = Path(args.output_dir) / "logs"
    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir,
        logging_dir=logging_dir,
    )

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )
    if accelerator.is_main_process:
        accelerator.init_trackers("bridge_lora_sana", config=vars(args))
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)

    # Set seed
    if args.seed is not None:
        set_seed(args.seed)

    # Create output directory
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

    # Load models
    logger.info(f"Loading models from {args.pretrained_model_name_or_path}")

    # Load VAE
    vae = AutoencoderDC.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        revision=args.revision,
    )
    vae.requires_grad_(False)
    vae.to(accelerator.device, dtype=torch.float32)

    # Load text encoder
    text_encoder = Gemma2Model.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=args.revision,
    )
    text_encoder.requires_grad_(False)
    text_encoder.to(accelerator.device, dtype=torch.bfloat16)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=args.revision,
    )

    # Load transformer
    transformer = SanaTransformer2DModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="transformer",
        revision=args.revision,
    )

    # Load base SanaPipeline to obtain its scheduler config for ViBT scheduler
    base_pipeline = SanaPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        revision=args.revision,
    )
    base_scheduler = base_pipeline.scheduler
    del base_pipeline

    logger.info("Models loaded successfully")

    # Setup LoRA for transformer
    logger.info(f"Setting up LoRA with rank={args.lora_rank}, alpha={args.lora_alpha}")
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        init_lora_weights="gaussian",
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
        lora_dropout=args.lora_dropout,
    )
    transformer = get_peft_model(transformer, lora_config)
    transformer.print_trainable_parameters()

    # Enable gradient checkpointing
    transformer.enable_gradient_checkpointing()

    # Setup optimizer
    optimizer = torch.optim.AdamW(
        transformer.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=1e-2,
        eps=1e-8,
    )

    # Setup dataset and dataloader
    logger.info(f"Loading dataset from {args.train_data_dir}")
    train_dataset = BridgeDataset(
        data_dir=args.train_data_dir,
        resolution=args.resolution,
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args.dataloader_num_workers,
    )

    # Calculate training steps
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch

    # Setup learning rate scheduler
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )

    # Prepare with accelerator
    transformer, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        transformer, optimizer, train_dataloader, lr_scheduler
    )

    # Calculate total batch size
    total_batch_size = (
        args.train_batch_size
        * accelerator.num_processes
        * args.gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    logger.info(f"  Bridge noise scale = {args.noise_scale}")
    logger.info(f"  Use stabilized velocity = {args.use_stabilized_velocity}")

    # Training loop
    global_step = 0
    first_epoch = 0
    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=global_step,
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )

    for epoch in range(first_epoch, args.num_train_epochs):
        transformer.train()
        train_loss = 0.0

        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(transformer):
                # Get source and target images
                source_images = batch['source_images'].to(accelerator.device)
                target_images = batch['target_images'].to(accelerator.device)
                prompts = batch['prompts']

                # Encode images to latents
                with torch.no_grad():
                    x0 = encode_images(vae, source_images)  # Source latents
                    x1 = encode_images(vae, target_images)  # Target latents

                    # Encode text prompts
                    text_inputs = tokenizer(
                        prompts,
                        padding="max_length",
                        max_length=300,
                        truncation=True,
                        return_tensors="pt",
                    ).to(accelerator.device)

                    encoder_hidden_states = text_encoder(
                        input_ids=text_inputs.input_ids,
                        attention_mask=text_inputs.attention_mask,
                    )[0]

                # Sample timestep t ~ U(0, 1)
                bsz = x0.shape[0]
                t = torch.rand(bsz, device=accelerator.device)

                # Sample noise ε ~ N(0, I)
                noise = torch.randn_like(x0)

                # Construct intermediate state x_t (Brownian Bridge formula)
                # x_t = (1-t)*x0 + t*x1 + sqrt(t*(1-t)) * ε
                t_expanded = t.view(-1, 1, 1, 1)
                x_t = (
                    (1 - t_expanded) * x0
                    + t_expanded * x1
                    + torch.sqrt(t_expanded * (1 - t_expanded)) * noise
                )

                # Compute target velocity: u_t = (x1 - x_t) / (1 - t)
                # Add small epsilon to avoid division by zero
                target_velocity = (x1 - x_t) / (1 - t_expanded + 1e-5)

                # Forward pass: predict velocity
                model_pred = transformer(
                    hidden_states=x_t,
                    timestep=t * 1000,  # Scale to [0, 1000]
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=text_inputs.attention_mask,
                ).sample

                # Compute Bridge loss
                loss = compute_bridge_loss(
                    model_pred=model_pred,
                    target_velocity=target_velocity,
                    x0=x0,
                    x1=x1,
                    t=t.view(-1, 1),
                    use_stabilized=args.use_stabilized_velocity,
                )

                # Backward pass
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(transformer.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Gather loss across all processes
            avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
            train_loss += avg_loss.item() / args.gradient_accumulation_steps

            # Update progress bar
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                # Logging
                if global_step % args.logging_steps == 0:
                    logs = {
                        "loss": train_loss,
                        "lr": lr_scheduler.get_last_lr()[0],
                        "step": global_step,
                        "epoch": epoch,
                    }
                    progress_bar.set_postfix(**logs)
                    accelerator.log(logs, step=global_step)
                    train_loss = 0.0

                # Save checkpoint
                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        os.makedirs(save_path, exist_ok=True)

                        # Save LoRA weights
                        lora_state_dict = get_peft_model_state_dict(transformer)
                        torch.save(lora_state_dict, os.path.join(save_path, "pytorch_lora_weights.bin"))

                        logger.info(f"Saved checkpoint to {save_path}")

                # Run validation
                if args.validation_prompt and global_step % args.validation_steps == 0:
                    logger.info(f"Running validation at step {global_step}")
                    run_validation(
                        transformer=transformer,
                        vae=vae,
                        text_encoder=text_encoder,
                        tokenizer=tokenizer,
                        args=args,
                        accelerator=accelerator,
                        global_step=global_step,
                        scheduler=base_scheduler,
                    )

            if global_step >= args.max_train_steps:
                break

    # Save final checkpoint
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        save_path = os.path.join(args.output_dir, "final_checkpoint")
        os.makedirs(save_path, exist_ok=True)

        # Save LoRA weights
        lora_state_dict = get_peft_model_state_dict(transformer)
        torch.save(lora_state_dict, os.path.join(save_path, "pytorch_lora_weights.bin"))

        logger.info(f"Training completed! Final checkpoint saved to {save_path}")

    accelerator.end_training()


def run_validation(
    transformer,
    vae,
    text_encoder,
    tokenizer,
    args,
    accelerator,
    global_step,
    scheduler,
):
    """Run validation and log images using Bridge inference (data-to-data)"""
    transformer.eval()

    # Wrap the base scheduler with ViBTScheduler (Brownian Bridge sampler)
    vibt_scheduler = ViBTScheduler.from_scheduler(scheduler)
    # Align noise_scale with training argument; keep default shift_gamma, seed from args if provided
    vibt_scheduler.set_parameters(
        noise_scale=args.noise_scale,
        shift_gamma=5.0,
        seed=args.seed,
    )

    # Create pipeline for inference using ViBT scheduler
    pipeline = SanaPipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        transformer=accelerator.unwrap_model(transformer),
        scheduler=vibt_scheduler,
    )
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

    # Load validation dataset to get source images
    val_dataset = BridgeDataset(
        data_dir=args.train_data_dir,
        resolution=args.resolution,
    )

    # Generate validation images
    generator = torch.Generator(device=accelerator.device)
    if args.seed is not None:
        generator = generator.manual_seed(args.seed)

    images = []
    for i in range(args.num_validation_images):
        # Get source image from dataset
        idx = i % len(val_dataset)
        sample = val_dataset[idx]
        source_image = sample['source_images'].unsqueeze(0).to(accelerator.device)

        # Encode source image to latents (Bridge starting point)
        with torch.no_grad():
            source_latents = encode_images(vae, source_image)

        with torch.autocast("cuda", dtype=torch.bfloat16):
            image = pipeline(
                prompt=args.validation_prompt,
                num_inference_steps=28,
                guidance_scale=4.5,
                latents=source_latents,  # ✅ Pass source latents for Bridge
                generator=generator,
            ).images[0]
        images.append(image)

    # Log to trackers
    for tracker in accelerator.trackers:
        if tracker.name == "tensorboard":
            np_images = np.stack([np.asarray(img) for img in images])
            tracker.writer.add_images("validation", np_images, global_step, dataformats="NHWC")
        elif tracker.name == "wandb":
            tracker.log({
                "validation": [
                    wandb.Image(image, caption=f"{i}: {args.validation_prompt}")
                    for i, image in enumerate(images)
                ]
            })

    del pipeline
    torch.cuda.empty_cache()
    transformer.train()


if __name__ == "__main__":
    main()

