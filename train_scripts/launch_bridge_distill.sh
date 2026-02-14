#!/bin/bash

# Bridge Distillation Training Launch Script
# Based on Twin-Bridge Distillation algorithm

export MODEL_PATH="/cache/SANA1.5_4.8B_1024px_diffusers"
export DATA_DIR="/cache/omnic/3D_Chibi"
export OUTPUT_DIR="/cache/sanaoutput/bridge_distill_3d_chibi"

# IMPORTANT: initialize from a trained Bridge LoRA (so the model can do bridge-path inference)
# Can be a directory like /path/to/checkpoint-XXXX or /path/to/final_checkpoint
export INIT_LORA_PATH="/cache/boot2sanaoutput/bridge_3d_chibi_sweep/ns_0.5_boot_0.2/checkpoint-5000"

# Training configuration
export RESOLUTION=1024
export TRAIN_BATCH_SIZE=2
export GRADIENT_ACCUMULATION_STEPS=1
export MAX_TRAIN_STEPS=10000
export LEARNING_RATE=1e-4

# LoRA configuration
export LORA_RANK=128
export LORA_ALPHA=128

# Bridge-specific configuration
export NOISE_SCALE=0.5
export USE_STABILIZED_VELOCITY="--use_stabilized_velocity"

# Conditioning (match concat Bridge training)
# - concat_text: concat(x_t, x0) + text prompt
# - concat: concat only (no text)
# - text: text only (no concat)
export CONDITIONING_MODE="concat_text"
export BOOTING_NOISE_SCALE=0.2

# Distillation weights
export LAMBDA_BASE=1.0      # RCGM consistency loss
export LAMBDA_ADV=0.5       # Self-adversarial loss
export LAMBDA_RECTIFY=0.5   # Path rectification loss
export LAMBDA_CORR=1.0      # Correction strength

# Distillation parameters
export RCGM_ORDER=2         # N=2 for RCGM
export EMA_DECAY=0.99       # EMA teacher decay

# Logging and checkpointing
export CHECKPOINTING_STEPS=500
export LOGGING_STEPS=10

# Validation (visualization)
export VALIDATION_STEPS=100
export VALIDATION_PROMPT="Convert the style to 3D Chibi Style"
export NUM_VALIDATION_IMAGES=4
export VALIDATION_INFERENCE_STEPS=1

# Launch training
accelerate launch train_bridge_distill_lora_sana.py \
  --pretrained_model_name_or_path="${MODEL_PATH}" \
  --init_lora_path="${INIT_LORA_PATH}" \
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
  --conditioning="${CONDITIONING_MODE}" \
  --booting_noise_scale=${BOOTING_NOISE_SCALE} \
  ${USE_STABILIZED_VELOCITY} \
  --lambda_base=${LAMBDA_BASE} \
  --lambda_adv=${LAMBDA_ADV} \
  --lambda_rectify=${LAMBDA_RECTIFY} \
  --lambda_corr=${LAMBDA_CORR} \
  --rcgm_order=${RCGM_ORDER} \
  --ema_decay=${EMA_DECAY} \
  --checkpointing_steps=${CHECKPOINTING_STEPS} \
  --logging_steps=${LOGGING_STEPS} \
  --validation_steps=${VALIDATION_STEPS} \
  --validation_prompt="${VALIDATION_PROMPT}" \
  --num_validation_images=${NUM_VALIDATION_IMAGES} \
  --validation_inference_steps=${VALIDATION_INFERENCE_STEPS} \
  --mixed_precision="bf16" \
  --report_to="tensorboard" \
  --dataloader_num_workers=4 \
  --seed=42
