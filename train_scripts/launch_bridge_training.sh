#!/bin/bash

# Bridge Training Launch Script for Sana Model
# Based on ViBT (Vision Bridge Transformer) algorithm

export MODEL_PATH="/cache/SANA1.5_4.8B_1024px_diffusers"
export DATA_DIR="/cache/omnic/3D_Chibi"
export OUTPUT_DIR="/cache/sanaoutput/bridge_3d_chibi"

# Training configuration
export RESOLUTION=1024
export TRAIN_BATCH_SIZE=4
export GRADIENT_ACCUMULATION_STEPS=1
export MAX_TRAIN_STEPS=20000
export LEARNING_RATE=1e-4

# LoRA configuration
export LORA_RANK=128
export LORA_ALPHA=128

# Bridge-specific configuration
export NOISE_SCALE=1.0
export USE_STABILIZED_VELOCITY="--use_stabilized_velocity"

# Logging and checkpointing
export CHECKPOINTING_STEPS=200
export LOGGING_STEPS=5
export VALIDATION_STEPS=50

# Validation prompt (optional)
export VALIDATION_PROMPT="Convert the style to 3D Chibi Style"

# Launch training
accelerate launch train_bridge_lora_sana.py \
  --pretrained_model_name_or_path="${MODEL_PATH}" \
  --train_data_dir="${DATA_DIR}" \
  --output_dir="${OUTPUT_DIR}" \
  --resolution=${RESOLUTION} \
  --train_batch_size=${TRAIN_BATCH_SIZE} \
  --gradient_accumulation_steps=${GRADIENT_ACCUMULATION_STEPS} \
  --max_train_steps=${MAX_TRAIN_STEPS} \
  --learning_rate=${LEARNING_RATE} \
  --lr_scheduler="constant" \
  --lr_warmup_steps=500 \
  --lora_rank=${LORA_RANK} \
  --lora_alpha=${LORA_ALPHA} \
  --lora_dropout=0.0 \
  --noise_scale=${NOISE_SCALE} \
  ${USE_STABILIZED_VELOCITY} \
  --checkpointing_steps=${CHECKPOINTING_STEPS} \
  --logging_steps=${LOGGING_STEPS} \
  --validation_steps=${VALIDATION_STEPS} \
  --validation_prompt="${VALIDATION_PROMPT}" \
  --num_validation_images=4 \
  --mixed_precision="bf16" \
  --report_to="tensorboard" \
  --dataloader_num_workers=4 \
  --seed=42
