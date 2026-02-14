#!/bin/bash

# Grid search over noise_scale and booting_noise_scale
# 每一组参数会单独建一个 output 子目录

set -e

# 基本配置（按需修改）
MODEL_PATH="/cache/SANA1.5_4.8B_1024px_diffusers"
DATA_DIR="/cache/omnic/3D_Chibi"
BASE_OUTPUT_DIR="/cache/07boot2sanaoutput/bridge_3d_chibi_sweep"

RESOLUTION=1024
TRAIN_BATCH_SIZE=1
GRADIENT_ACCUMULATION_STEPS=1
MAX_TRAIN_STEPS=5000
LEARNING_RATE=1e-4

LORA_RANK=128
LORA_ALPHA=128
LORA_DROPOUT=0.0

USE_STABILIZED_VELOCITY="--use_stabilized_velocity"
CHECKPOINTING_STEPS=500
LOGGING_STEPS=5
VALIDATION_STEPS=500
VALIDATION_PROMPT="Convert the style to 3D Chibi Style"

MIXED_PRECISION="bf16"
REPORT_TO="tensorboard"
DATALOADER_NUM_WORKERS=4
SEED=42

# 这里设置你想要遍历的取值
NOISE_SCALE_LIST=(0.0 0.5)
BOOTING_NOISE_LIST=(0.0 0.2 0.5)

for noise_scale in "${NOISE_SCALE_LIST[@]}"; do
  for booting_noise_scale in "${BOOTING_NOISE_LIST[@]}"; do
    EXP_NAME="ns_${noise_scale}_boot_${booting_noise_scale}"
    OUTPUT_DIR="${BASE_OUTPUT_DIR}/${EXP_NAME}"

    echo "============================="
    echo "Running experiment: ${EXP_NAME}"
    echo "  noise_scale=${noise_scale}"
    echo "  booting_noise_scale=${booting_noise_scale}"
    echo "  output_dir=${OUTPUT_DIR}"
    echo "============================="

    mkdir -p "${OUTPUT_DIR}"

    accelerate launch train_bridge_lora_sana_concat.py \
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
      --lora_dropout=${LORA_DROPOUT} \
      --noise_scale=${noise_scale} \
      --booting_noise_scale=${booting_noise_scale} \
      ${USE_STABILIZED_VELOCITY} \
      --checkpointing_steps=${CHECKPOINTING_STEPS} \
      --logging_steps=${LOGGING_STEPS} \
      --validation_steps=${VALIDATION_STEPS} \
      --validation_prompt="${VALIDATION_PROMPT}" \
      --num_validation_images=4 \
      --mixed_precision="${MIXED_PRECISION}" \
      --report_to="${REPORT_TO}" \
      --dataloader_num_workers=${DATALOADER_NUM_WORKERS} \
      --seed=${SEED}

  done
done
