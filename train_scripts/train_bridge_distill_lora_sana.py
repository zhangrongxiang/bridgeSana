#!/usr/bin/env python
"""
Bridge Distillation Training Script for Sana Model
Based on Twin-Bridge Distillation algorithm (TwinFlow + ViBT)

This script implements:
1. L_base: RCGM (N=2) consistency loss
2. L_adv: Self-adversarial fake learning (negative time)
3. L_rectify: Path rectification via velocity difference
"""

import argparse
import json
import logging
import math
import os
import contextlib
from pathlib import Path
from copy import deepcopy

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
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict
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

if is_wandb_available():
    import wandb

check_min_version("0.32.0.dev0")

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Bridge distillation training script")

    # Model arguments
    parser.add_argument("--pretrained_model_name_or_path", type=str, required=True)
    parser.add_argument("--revision", type=str, default=None)

    # Dataset arguments
    parser.add_argument("--train_data_dir", type=str, required=True)
    parser.add_argument("--resolution", type=int, default=1024)

    # Training arguments
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument("--max_train_steps", type=int, default=20000)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--lr_scheduler", type=str, default="constant")
    parser.add_argument("--lr_warmup_steps", type=int, default=500)

    # LoRA arguments
    parser.add_argument("--lora_rank", type=int, default=128)
    parser.add_argument("--lora_alpha", type=int, default=128)
    parser.add_argument("--lora_dropout", type=float, default=0.0)

    # Initialization (recommended): start distillation from a trained Bridge LoRA
    # Path can be either a directory containing `pytorch_lora_weights.bin` or the file itself.
    parser.add_argument(
        "--init_lora_path",
        type=str,
        default=None,
        help="Path to a trained Bridge LoRA checkpoint (dir or pytorch_lora_weights.bin).",
    )

    # Bridge-specific arguments
    parser.add_argument("--noise_scale", type=float, default=0.0)
    parser.add_argument("--use_stabilized_velocity", action="store_true")

    # Conditioning (match concat trainer)
    parser.add_argument(
        "--conditioning",
        type=str,
        default="concat_text",
        choices=["concat", "text", "concat_text"],
        help="Conditioning mode: concat (channel concat with x0), text (prompt only), concat_text (both).",
    )
    parser.add_argument(
        "--booting_noise_scale",
        type=float,
        default=0.0,
        help="Std of Gaussian booting noise added to source latents for concat conditioning: x0_cond = x0 + scale * eps. 0 disables.",
    )

    # Distillation-specific arguments
    parser.add_argument("--lambda_base", type=float, default=1.0, help="Weight for RCGM base loss")
    parser.add_argument("--lambda_adv", type=float, default=0.5, help="Weight for adversarial loss")
    parser.add_argument("--lambda_rectify", type=float, default=0.5, help="Weight for rectification loss")
    parser.add_argument("--lambda_corr", type=float, default=1.0, help="Correction strength in rectification")
    parser.add_argument("--rcgm_order", type=int, default=2, help="RCGM estimation order (N)")
    parser.add_argument("--ema_decay", type=float, default=0.99, help="EMA decay rate for teacher model")

    # Output arguments
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--checkpointing_steps", type=int, default=500)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--validation_steps", type=int, default=500)
    parser.add_argument("--validation_prompt", type=str, default=None)
    parser.add_argument("--num_validation_images", type=int, default=4)
    parser.add_argument("--validation_inference_steps", type=int, default=4, help="Number of inference steps for validation (use few steps for distilled models)")

    # Misc arguments
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--mixed_precision", type=str, default="bf16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--report_to", type=str, default="tensorboard")
    parser.add_argument("--dataloader_num_workers", type=int, default=0)

    args = parser.parse_args()
    return args


class BridgeDataset(Dataset):
    """Dataset for Bridge training with source-target image pairs"""

    def __init__(self, data_dir, resolution=1024):
        self.data_dir = Path(data_dir)
        self.resolution = resolution

        jsonl_path = self.data_dir / "train.jsonl"
        self.data = []
        with open(jsonl_path, 'r') as f:
            for line in f:
                self.data.append(json.loads(line))

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
        src_rel = item['src']
        tar_rel = item['tar']

        # Handle path prefix
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

        src_tensor = self.transform(src_image)
        tar_tensor = self.transform(tar_image)

        return {
            'source_images': src_tensor,
            'target_images': tar_tensor,
            'prompts': "Convert the style to 3D Chibi Style",
        }


def collate_fn(examples):
    source_images = torch.stack([example['source_images'] for example in examples])
    target_images = torch.stack([example['target_images'] for example in examples])
    prompts = [example['prompts'] for example in examples]
    return {
        'source_images': source_images,
        'target_images': target_images,
        'prompts': prompts,
    }


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


def decode_latents(vae, latents):
    """Decode latents back to image space in [-1, 1] (approx)."""
    scaling_factor = getattr(getattr(vae, "config", None), "scaling_factor", None)
    if scaling_factor is not None:
        latents = latents / scaling_factor

    decoded = vae.decode(latents)
    if hasattr(decoded, "sample"):
        images = decoded.sample
    elif hasattr(decoded, "samples"):
        images = decoded.samples
    else:
        images = decoded
    return images


def compute_stabilized_velocity_loss(model_pred, target_velocity, x0, x1, t):
    """Compute stabilized velocity matching loss"""
    batch_size = x0.shape[0]
    D = x0[0].numel()
    diff_norm_sq = torch.sum((x1 - x0).reshape(batch_size, -1) ** 2, dim=1, keepdim=True)

    # α^2 = 1 + [(1-t) * D] / [t * ||x1-x0||^2]
    alpha_sq = 1.0 + ((1 - t) * D) / (t * diff_norm_sq + 1e-8)
    alpha = torch.sqrt(alpha_sq).reshape(-1, 1, 1, 1)

    normalized_pred = model_pred / alpha
    normalized_target = target_velocity / alpha

    loss = F.mse_loss(normalized_pred, normalized_target, reduction="mean")
    return loss


def update_ema_model(student_model, teacher_model, decay):
    """Update EMA teacher model"""
    with torch.no_grad():
        for student_param, teacher_param in zip(student_model.parameters(), teacher_model.parameters()):
            teacher_param.data.mul_(decay).add_(student_param.data, alpha=1 - decay)


def sample_concat_condition(
    *,
    transformer,
    cond_proj,
    scheduler,
    latents,
    cond_latents,
    encoder_hidden_states=None,
    encoder_attention_mask=None,
    num_inference_steps=28,
):
    """Minimal denoising loop that injects concat conditioning at every step."""
    scheduler.set_timesteps(num_inference_steps, device=latents.device)
    for t in scheduler.timesteps:
        model_in = cond_proj(torch.cat([latents, cond_latents], dim=1))

        t_scalar = t
        if not torch.is_tensor(t_scalar):
            t_scalar = torch.tensor(t_scalar, device=latents.device)
        timesteps = torch.full(
            (model_in.shape[0],),
            t_scalar.item(),
            device=latents.device,
            dtype=torch.float32,
        )

        model_out = transformer(
            hidden_states=model_in,
            timestep=timesteps,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
        ).sample
        step_out = scheduler.step(model_out, t, latents, return_dict=True)
        if hasattr(step_out, "prev_sample"):
            latents = step_out.prev_sample
        elif isinstance(step_out, dict) and "prev_sample" in step_out:
            latents = step_out["prev_sample"]
        elif isinstance(step_out, (tuple, list)) and len(step_out) > 0:
            latents = step_out[0]
        else:
            raise TypeError(f"Unsupported scheduler.step output type: {type(step_out)}")
    return latents


@torch.no_grad()
def compute_rcgm_target(
    teacher_model,
    cond_proj,
    x_t,
    cond_latents,
    t,
    t_target,
    encoder_hidden_states,
    encoder_attention_mask,
    N=2,
):
    """
    Compute RCGM (Recursive Consistency Gradient Matching) target.

    This implements N-step consistency: the model should predict a velocity
    that matches the accumulated velocity over N future steps.

    Args:
        teacher_model: EMA teacher model
        x_t: Current latent state
        t: Current timestep (reversed time, t=1 at source, t=0 at target)
        t_target: Target timestep (usually 0)
        N: Number of steps for RCGM (default 2)
    """
    # Generate intermediate timesteps
    ts = [t * (1 - i / N) + t_target * (i / N) for i in range(N + 1)]

    accumulated_velocity = torch.zeros_like(x_t)
    current_x = x_t

    for i in range(N):
        t_current = ts[i]
        t_next = ts[i + 1]
        # Negative (moving from t=1 to t=0). Reshape for broadcasting with [B,C,H,W].
        delta_t = (t_next - t_current).view(-1, 1, 1, 1)

        # Predict velocity at current state
        # SanaTransformer expects timesteps as a 1D tensor of shape [batch]
        timesteps = (t_current * 1000).to(device=current_x.device, dtype=torch.float32)

        model_in = current_x
        if cond_proj is not None:
            model_in = cond_proj(torch.cat([current_x, cond_latents], dim=1))

        velocity = teacher_model(
            hidden_states=model_in,
            timestep=timesteps,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
        ).sample

        # Accumulate velocity
        accumulated_velocity = accumulated_velocity + velocity * delta_t

        # Update state for next step
        current_x = current_x + velocity * delta_t

    # Compute RCGM target: (x_final - x_t) / (t_target - t)
    denom = (t_target - t).view(-1, 1, 1, 1)
    rcgm_target = (current_x - x_t) / (denom + 1e-5)

    return rcgm_target


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
        accelerator.init_trackers("bridge_distill_lora_sana", config=vars(args))

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

    # Load models
    logger.info(f"Loading models from {args.pretrained_model_name_or_path}")

    weight_dtype = torch.float32
    if args.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif args.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    vae = AutoencoderDC.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        revision=args.revision,
    )
    vae.requires_grad_(False)
    vae.to(accelerator.device, dtype=torch.float32)

    text_encoder = Gemma2Model.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=args.revision,
    )
    text_encoder.requires_grad_(False)
    text_encoder.to(accelerator.device, dtype=weight_dtype)

    tokenizer = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=args.revision,
    )

    transformer = SanaTransformer2DModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="transformer",
        revision=args.revision,
    )

    # Load base SanaPipeline to obtain its scheduler config for validation sampling
    base_pipeline = SanaPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        revision=args.revision,
    )
    base_scheduler = base_pipeline.scheduler
    del base_pipeline

    logger.info("Models loaded successfully")

    # Channel-concat conditioning adapter: maps [x_t, x0] (2C) -> C
    cond_proj = None
    if args.conditioning in {"concat", "concat_text"}:
        latent_channels = getattr(getattr(vae, "config", None), "latent_channels", 4)
        cond_proj = nn.Conv2d(latent_channels * 2, latent_channels, kernel_size=1, bias=True)
        nn.init.zeros_(cond_proj.weight)
        nn.init.zeros_(cond_proj.bias)
        with torch.no_grad():
            for c in range(latent_channels):
                cond_proj.weight[c, c, 0, 0] = 1.0
        cond_proj.to(accelerator.device)

    # Setup LoRA for student transformer
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
    transformer.enable_gradient_checkpointing()

    # Optionally load trained Bridge LoRA weights before distillation starts
    if args.init_lora_path:
        init_path = Path(args.init_lora_path)
        lora_file = init_path
        if init_path.is_dir():
            lora_file = init_path / "pytorch_lora_weights.bin"

        if not lora_file.exists():
            raise FileNotFoundError(
                f"init_lora_path not found or missing pytorch_lora_weights.bin: {lora_file}"
            )
        if set_peft_model_state_dict is None:
            raise RuntimeError(
                "set_peft_model_state_dict is not available in this PEFT version; "
                "please upgrade peft or load via PeftModel.from_pretrained."
            )

        state = torch.load(lora_file, map_location="cpu")
        set_peft_model_state_dict(transformer, state)
        logger.info(f"Initialized student LoRA from {lora_file}")

        # Also load concat adapter if present
        if cond_proj is not None and init_path.is_dir():
            cond_file = init_path / "cond_proj_weights.bin"
            if cond_file.exists():
                cond_state = torch.load(cond_file, map_location="cpu")
                cond_proj.load_state_dict(cond_state)
                logger.info(f"Initialized cond_proj from {cond_file}")

    # Create EMA teacher model (deepcopy of student)
    logger.info(f"Creating EMA teacher model with decay={args.ema_decay}")
    teacher_transformer = deepcopy(transformer)
    teacher_transformer.requires_grad_(False)
    teacher_transformer.eval()

    teacher_cond_proj = None
    if cond_proj is not None:
        teacher_cond_proj = deepcopy(cond_proj)
        teacher_cond_proj.requires_grad_(False)
        teacher_cond_proj.eval()

    # Setup optimizer
    optimizer = torch.optim.AdamW(
        list(transformer.parameters()) + (list(cond_proj.parameters()) if cond_proj is not None else []),
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
    if cond_proj is not None:
        transformer, teacher_transformer, cond_proj, teacher_cond_proj, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            transformer, teacher_transformer, cond_proj, teacher_cond_proj, optimizer, train_dataloader, lr_scheduler
        )
        teacher_cond_proj.eval()
    else:
        transformer, teacher_transformer, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            transformer, teacher_transformer, optimizer, train_dataloader, lr_scheduler
        )

    total_batch_size = (
        args.train_batch_size
        * accelerator.num_processes
        * args.gradient_accumulation_steps
    )

    logger.info("***** Running Bridge Distillation Training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Total train batch size = {total_batch_size}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    logger.info(f"  Distillation weights: λ_base={args.lambda_base}, λ_adv={args.lambda_adv}, λ_rectify={args.lambda_rectify}")

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
                    x0 = encode_images(vae, source_images)
                    x1 = encode_images(vae, target_images)

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

                # Booting noise for concat conditioning (sample once per batch)
                x0_cond = x0
                if cond_proj is not None and args.booting_noise_scale and args.booting_noise_scale > 0:
                    x0_cond = x0 + args.booting_noise_scale * torch.randn_like(x0)

                # Sample timestep t ~ U(0.01, 0.99) with reversed time
                bsz = x0.shape[0]
                t_min, t_max = 0.01, 0.99
                t_forward = t_min + (t_max - t_min) * torch.rand(bsz, device=accelerator.device)
                t = 1.0 - t_forward  # Reversed time: t ∈ [0.99, 0.01]

                # Sample noise
                noise = torch.randn_like(x0)

                # Construct x_t (Bridge formula with reversed time)
                t_expanded = t.view(-1, 1, 1, 1)
                x_t = (
                    t_expanded * x0
                    + (1 - t_expanded) * x1
                    + args.noise_scale * torch.sqrt(t_expanded * (1 - t_expanded)) * noise
                )

                # Reversed-time convention (matches Sana pretraining / our bridge trainer):
                # t=1 -> x0, t=0 -> x1, x_t = t*x0 + (1-t)*x1 + s*sqrt(t(1-t))*eps
                # Target velocity is v(t, x_t) = (x_t - x1) / t
                target_velocity = (x_t - x1) / (t_expanded + 1e-5)

                # ========== Loss 1: L_base (RCGM Consistency) ==========
                # Forward pass: predict velocity at positive time
                timesteps_pos = t * 1000
                with accelerator.autocast():
                    model_in_pos = x_t
                    if cond_proj is not None:
                        model_in_pos = cond_proj(torch.cat([x_t, x0_cond], dim=1))

                    model_pred_pos = transformer(
                        hidden_states=model_in_pos,
                        timestep=timesteps_pos,
                        encoder_hidden_states=encoder_hidden_states if args.conditioning in {"text", "concat_text"} else None,
                        encoder_attention_mask=text_inputs.attention_mask if args.conditioning in {"text", "concat_text"} else None,
                    ).sample

                # Compute RCGM target using teacher model
                with torch.no_grad():
                    rcgm_target = compute_rcgm_target(
                        teacher_transformer,
                        teacher_cond_proj if teacher_cond_proj is not None else None,
                        x_t,
                        x0_cond,
                        t,
                        torch.zeros_like(t),  # Target timestep = 0
                        encoder_hidden_states if args.conditioning in {"text", "concat_text"} else None,
                        text_inputs.attention_mask if args.conditioning in {"text", "concat_text"} else None,
                        N=args.rcgm_order,
                    )

                # L_base: match RCGM target
                if args.use_stabilized_velocity:
                    loss_base = compute_stabilized_velocity_loss(
                        model_pred_pos, rcgm_target, x0, x1, t.view(-1, 1)
                    )
                else:
                    loss_base = F.mse_loss(model_pred_pos, rcgm_target, reduction="mean")

                # ========== Loss 2: L_adv (Self-Adversarial Fake Learning) ==========
                # Construct fake target x_fake from model prediction
                with torch.no_grad():
                    x_fake = x_t - t_expanded * model_pred_pos.detach()

                # Construct x_{-t}^fake: Bridge from x0 to x_fake at negative time
                x_neg_t_fake = (
                    t_expanded * x0
                    + (1 - t_expanded) * x_fake
                    + args.noise_scale * torch.sqrt(t_expanded * (1 - t_expanded)) * torch.randn_like(x0)
                )

                # Forward pass at negative time (adversarial mode)
                timesteps_neg = -t * 1000  # Negative time
                with accelerator.autocast():
                    model_in_neg = x_neg_t_fake
                    if cond_proj is not None:
                        model_in_neg = cond_proj(torch.cat([x_neg_t_fake, x0_cond], dim=1))

                    model_pred_neg = transformer(
                        hidden_states=model_in_neg,
                        timestep=timesteps_neg,
                        encoder_hidden_states=encoder_hidden_states if args.conditioning in {"text", "concat_text"} else None,
                        encoder_attention_mask=text_inputs.attention_mask if args.conditioning in {"text", "concat_text"} else None,
                    ).sample

                # Target velocity for fake trajectory: v_{-t} = (x_fake - x_{-t}^fake) / (1 - t)
                # Under the same reversed-time convention, the bridge velocity towards endpoint x_fake is
                # v_fake(t, x_t_fake) = (x_t_fake - x_fake) / t
                target_velocity_fake = (x_neg_t_fake - x_fake) / (t_expanded + 1e-5)

                # L_adv: learn to generate fake target
                if args.use_stabilized_velocity:
                    loss_adv = compute_stabilized_velocity_loss(
                        model_pred_neg, target_velocity_fake, x0, x_fake, t.view(-1, 1)
                    )
                else:
                    loss_adv = F.mse_loss(model_pred_neg, target_velocity_fake, reduction="mean")

                # ========== Loss 3: L_rectify (Path Rectification) ==========
                # Compute velocity difference: Δv = v(x_t, -t) - v(x_t, t)
                # This measures the discrepancy between real and fake predictions
                with torch.no_grad():
                    # Predict at negative time for the same x_t
                    with accelerator.autocast():
                        model_in_neg_same = x_t
                        if cond_proj is not None:
                            model_in_neg_same = cond_proj(torch.cat([x_t, x0_cond], dim=1))

                        model_pred_neg_same = transformer(
                            hidden_states=model_in_neg_same,
                            timestep=-t * 1000,
                            encoder_hidden_states=encoder_hidden_states if args.conditioning in {"text", "concat_text"} else None,
                            encoder_attention_mask=text_inputs.attention_mask if args.conditioning in {"text", "concat_text"} else None,
                        ).sample

                    # Velocity difference
                    velocity_diff = model_pred_neg_same - model_pred_pos.detach()

                    # Rectified target: u_real - λ_corr * Δv
                    rectified_target = target_velocity - args.lambda_corr * velocity_diff

                # L_rectify: match rectified target
                if args.use_stabilized_velocity:
                    loss_rectify = compute_stabilized_velocity_loss(
                        model_pred_pos, rectified_target, x0, x1, t.view(-1, 1)
                    )
                else:
                    loss_rectify = F.mse_loss(model_pred_pos, rectified_target, reduction="mean")

                # ========== Total Loss ==========
                loss = (
                    args.lambda_base * loss_base
                    + args.lambda_adv * loss_adv
                    + args.lambda_rectify * loss_rectify
                )

                # Backward pass
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(transformer.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                # Update EMA teacher
                if accelerator.sync_gradients:
                    update_ema_model(
                        accelerator.unwrap_model(transformer),
                        accelerator.unwrap_model(teacher_transformer),
                        args.ema_decay
                    )
                    if cond_proj is not None and teacher_cond_proj is not None:
                        update_ema_model(
                            accelerator.unwrap_model(cond_proj),
                            accelerator.unwrap_model(teacher_cond_proj),
                            args.ema_decay,
                        )

            # Gather loss across all processes (robust for any batch size)
            avg_loss = accelerator.gather(loss.detach().float().reshape(1)).mean()
            train_loss += avg_loss.item() / args.gradient_accumulation_steps

            # Update progress bar
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                # Logging
                if global_step % args.logging_steps == 0:
                    mean_loss = train_loss / args.logging_steps
                    # In distributed runs, gather scalars to log global means.
                    loss_base_val = accelerator.gather(loss_base.detach().float().reshape(1)).mean().item()
                    loss_adv_val = accelerator.gather(loss_adv.detach().float().reshape(1)).mean().item()
                    loss_rectify_val = accelerator.gather(loss_rectify.detach().float().reshape(1)).mean().item()
                    logs = {
                        "loss": mean_loss,
                        "loss_base": loss_base_val,
                        "loss_adv": loss_adv_val,
                        "loss_rectify": loss_rectify_val,
                        "lr": lr_scheduler.get_last_lr()[0],
                        "step": global_step,
                        "epoch": epoch,
                    }
                    progress_bar.set_postfix(**logs)
                    accelerator.log(logs, step=global_step)
                    if accelerator.is_main_process:
                        logger.info(
                            "Step %d: loss=%.4f (base=%.4f, adv=%.4f, rectify=%.4f), lr=%.3e",
                            global_step,
                            mean_loss,
                            loss_base_val,
                            loss_adv_val,
                            loss_rectify_val,
                            logs["lr"],
                        )
                    train_loss = 0.0

                # Run validation (generate images) if configured
                if args.validation_prompt and global_step % args.validation_steps == 0:
                    logger.info(f"Running validation at step {global_step}")
                    run_validation(
                        transformer=transformer,
                        cond_proj=cond_proj,
                        vae=vae,
                        text_encoder=text_encoder,
                        tokenizer=tokenizer,
                        args=args,
                        accelerator=accelerator,
                        global_step=global_step,
                        scheduler=base_scheduler,
                    )

                # Save checkpoint
                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        os.makedirs(save_path, exist_ok=True)

                        # Save LoRA weights
                        lora_state_dict = get_peft_model_state_dict(transformer)
                        torch.save(lora_state_dict, os.path.join(save_path, "pytorch_lora_weights.bin"))

                        if cond_proj is not None:
                            cond_state = accelerator.get_state_dict(cond_proj)
                            torch.save(cond_state, os.path.join(save_path, "cond_proj_weights.bin"))

                        logger.info(f"Saved checkpoint to {save_path}")

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

        if cond_proj is not None:
            cond_state = accelerator.get_state_dict(cond_proj)
            torch.save(cond_state, os.path.join(save_path, "cond_proj_weights.bin"))

        logger.info(f"Training completed! Final checkpoint saved to {save_path}")

    accelerator.end_training()


def run_validation(
    transformer,
    cond_proj,
    vae,
    text_encoder,
    tokenizer,
    args,
    accelerator,
    global_step,
    scheduler,
):
    """Generate validation images starting from source latents and log to trackers."""
    transformer.eval()
    vae.eval()
    text_encoder.eval()

    scheduler_for_val = type(scheduler).from_config(scheduler.config)

    # Try to use ViBT scheduler if available (matches Bridge velocity update semantics)
    try:
        from vibt.scheduler import ViBTScheduler
        from diffusers.schedulers import UniPCMultistepScheduler

        if isinstance(scheduler_for_val, UniPCMultistepScheduler):
            scheduler_for_val = ViBTScheduler.from_scheduler(
                scheduler_for_val,
                noise_scale=args.noise_scale,
                shift_gamma=5.0,
            )
            logger.info(f"Validation using ViBT scheduler with noise_scale={args.noise_scale}")
        else:
            base_unipc = UniPCMultistepScheduler.from_config(scheduler_for_val.config)
            scheduler_for_val = ViBTScheduler.from_scheduler(
                base_unipc,
                noise_scale=args.noise_scale,
                shift_gamma=5.0,
            )
            logger.info(f"Validation using ViBT scheduler with noise_scale={args.noise_scale}")
    except Exception as e:
        logger.info(f"ViBT scheduler not available, using model scheduler: {e}")

    use_concat = args.conditioning in {"concat", "concat_text"} and cond_proj is not None
    if not use_concat:
        pipeline = SanaPipeline(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            transformer=accelerator.unwrap_model(transformer),
            scheduler=scheduler_for_val,
        )
        pipeline = pipeline.to(accelerator.device)
        pipeline.set_progress_bar_config(disable=True)

    val_dataset = BridgeDataset(
        data_dir=args.train_data_dir,
        resolution=args.resolution,
    )

    generator = torch.Generator(device=accelerator.device)
    if args.seed is not None:
        generator = generator.manual_seed(args.seed)

    images = []
    one_step_images = []
    autocast_ctx = (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if accelerator.device.type == "cuda"
        else contextlib.nullcontext()
    )

    # Pre-encode validation prompt for one-step generation
    text_inputs = tokenizer(
        [args.validation_prompt],
        padding="max_length",
        max_length=300,
        truncation=True,
        return_tensors="pt",
    ).to(accelerator.device)
    with torch.no_grad():
        with autocast_ctx:
            encoder_hidden_states = text_encoder(
                input_ids=text_inputs.input_ids,
                attention_mask=text_inputs.attention_mask,
            )[0]

    for i in range(args.num_validation_images):
        idx = i % len(val_dataset)
        sample = val_dataset[idx]
        source_image = sample["source_images"].unsqueeze(0).to(accelerator.device)

        with torch.no_grad():
            source_latents = encode_images(vae, source_image)

        cond_latents = source_latents
        if use_concat and args.booting_noise_scale and args.booting_noise_scale > 0:
            noise = torch.randn(
                source_latents.shape,
                generator=generator,
                device=source_latents.device,
                dtype=source_latents.dtype,
            )
            cond_latents = source_latents + args.booting_noise_scale * noise

        with torch.inference_mode():
            with autocast_ctx:
                # Multi-step sampling (for reference)
                if use_concat:
                    out_latents = sample_concat_condition(
                        transformer=accelerator.unwrap_model(transformer),
                        cond_proj=accelerator.unwrap_model(cond_proj),
                        scheduler=scheduler_for_val,
                        latents=source_latents,
                        cond_latents=cond_latents,
                        encoder_hidden_states=encoder_hidden_states if args.conditioning in {"text", "concat_text"} else None,
                        encoder_attention_mask=text_inputs.attention_mask if args.conditioning in {"text", "concat_text"} else None,
                        num_inference_steps=args.validation_inference_steps,
                    )
                    img_tensor = decode_latents(vae, out_latents).clamp(-1, 1)
                    img = (img_tensor[0].detach().float().cpu() + 1) / 2
                    img = img.permute(1, 2, 0).numpy()
                    img = (img * 255).round().clip(0, 255).astype(np.uint8)
                    image = Image.fromarray(img)
                else:
                    image = pipeline(
                        prompt=args.validation_prompt,
                        num_inference_steps=args.validation_inference_steps,
                        guidance_scale=1.0,
                        latents=source_latents,
                        generator=generator,
                    ).images[0]
                images.append(image)

                # Explicit one-step Bridge estimate (show distilled behavior)
                # Reversed-time: t=1 corresponds to x0. With v(t, x_t)=(x_t-x1)/t, we have x1 ≈ x_t - t*v.
                timesteps_1 = torch.full(
                    (source_latents.shape[0],),
                    1000.0,
                    device=source_latents.device,
                    dtype=torch.float32,
                )
                model_in_1 = source_latents
                if use_concat:
                    model_in_1 = accelerator.unwrap_model(cond_proj)(torch.cat([source_latents, cond_latents], dim=1))

                model_v = accelerator.unwrap_model(transformer)(
                    hidden_states=model_in_1,
                    timestep=timesteps_1,
                    encoder_hidden_states=encoder_hidden_states if args.conditioning in {"text", "concat_text"} else None,
                    encoder_attention_mask=text_inputs.attention_mask if args.conditioning in {"text", "concat_text"} else None,
                ).sample
                x1_hat = source_latents - model_v  # t=1
                img_tensor = decode_latents(vae, x1_hat).clamp(-1, 1)
                img = (img_tensor[0].detach().float().cpu() + 1) / 2
                img = img.permute(1, 2, 0).numpy()
                img = (img * 255).round().clip(0, 255).astype(np.uint8)
                one_step_images.append(Image.fromarray(img))

    for tracker in accelerator.trackers:
        if tracker.name == "tensorboard":
            np_images = np.stack([np.asarray(img) for img in images])
            tracker.writer.add_images("validation", np_images, global_step, dataformats="NHWC")
            np_images_1 = np.stack([np.asarray(img) for img in one_step_images])
            tracker.writer.add_images("validation_one_step", np_images_1, global_step, dataformats="NHWC")
        elif tracker.name == "wandb":
            tracker.log({
                "validation": [
                    wandb.Image(image, caption=f"{i}: {args.validation_prompt}")
                    for i, image in enumerate(images)
                ]
            })
            tracker.log({
                "validation_one_step": [
                    wandb.Image(image, caption=f"{i}: one-step")
                    for i, image in enumerate(one_step_images)
                ]
            })

    if not use_concat:
        del pipeline
    torch.cuda.empty_cache()
    transformer.train()


if __name__ == "__main__":
    main()











