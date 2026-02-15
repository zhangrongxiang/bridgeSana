#!/bin/bash

# sCM + LADD Distillation Launch Script for Sana Bridge
# - Trains student LoRA (+ optional cond_proj) to distill from a frozen teacher bridge.
# - Implements algorithm from new-cm-bridge.md (TrigFlow-time sCM + LADD adversarial).
# - IMPORTANT: uses Sana bridge timestep convention internally (0..1000, reversed time).

set -e

# -----------------------
# Paths
# -----------------------
export MODEL_PATH="/cache/SANA1.5_4.8B_1024px_diffusers"
export DATA_DIR="/cache/omnic/3D_Chibi"
export OUTPUT_DIR="/cache/sanaoutput/bridge_scm_ladd_distill_3d_chibi"

# Teacher checkpoint (Bridge LoRA + (optional) cond_proj)
# You can point this to either a directory containing pytorch_lora_weights.bin
# or to the file itself.
export TEACHER_LORA_PATH="/cache/boot2sanaoutput/bridge_3d_chibi_sweep/ns_0.5_boot_0.2/final_checkpoint"
export TEACHER_COND_PROJ_PATH="/cache/boot2sanaoutput/bridge_3d_chibi_sweep/ns_0.5_boot_0.2/final_checkpoint/cond_proj_weights.bin"

# Optional: initialize student from an existing LoRA
# export INIT_STUDENT_LORA_PATH="/cache/bootsanaoutput/bridge_3d_chibi/checkpoint-2000"

# -----------------------
# Training configuration
# -----------------------
export RESOLUTION=1024
export TRAIN_BATCH_SIZE=1
export GRADIENT_ACCUMULATION_STEPS=1
export MAX_TRAIN_STEPS=5000
export LEARNING_RATE=1e-4

# LoRA configuration
export LORA_RANK=128
export LORA_ALPHA=128
export LORA_DROPOUT=0.0

# Conditioning
# - concat_text: concat(latents, x0_cond) + text prompt
# - concat: concat only (no text)
# - text: text only (no concat)
export CONDITIONING_MODE="concat_text"

# ODE-style randomness knobs
# - init_latents_noise_scale: one-time noise added to x0 latents
# - booting_noise_scale: perturb x0_cond used only in concat conditioning
export INIT_LATENTS_NOISE_SCALE=0.0
export BOOTING_NOISE_SCALE=0.2

# sCM + LADD weights
export SCM_LAMBDA=1.0
export ADV_LAMBDA=0.1
export DISC_LR=1e-4

# TrigFlow time sampling (radians)
export T_TRIG_MIN=1e-3
export T_TRIG_MAX=1.5697963   # (pi/2) - 1e-3
export SCM_FD_DT=1e-3
export SCM_WEIGHTING="none"   # none | inv_tan

# Misc
export CHECKPOINTING_STEPS=500
export LOGGING_STEPS=10
export IMAGE_LOGGING_STEPS=10  # set to e.g. 100 to log images every 100 steps
export NUM_LOG_IMAGES=4
export SEED=42

# If you want to force using the rcm conda env, uncomment the next line.
# (This matches your requirement that testing uses conda env named rcm.)
# export LAUNCH_PREFIX="conda run -n rcm"
export LAUNCH_PREFIX=""

# -----------------------
# Launch
# -----------------------
${LAUNCH_PREFIX} accelerate launch train_bridge_scm_ladd_distill_lora_sana.py \
  --pretrained_model_name_or_path="${MODEL_PATH}" \
  --train_data_dir="${DATA_DIR}" \
  --teacher_lora_path="${TEACHER_LORA_PATH}" \
  --teacher_cond_proj_path="${TEACHER_COND_PROJ_PATH}" \
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
  --conditioning="${CONDITIONING_MODE}" \
  --init_latents_noise_scale=${INIT_LATENTS_NOISE_SCALE} \
  --booting_noise_scale=${BOOTING_NOISE_SCALE} \
  --scm_lambda=${SCM_LAMBDA} \
  --adv_lambda=${ADV_LAMBDA} \
  --discriminator_lr=${DISC_LR} \
  --t_trig_min=${T_TRIG_MIN} \
  --t_trig_max=${T_TRIG_MAX} \
  --scm_fd_dt=${SCM_FD_DT} \
  --scm_weighting=${SCM_WEIGHTING} \
  --checkpointing_steps=${CHECKPOINTING_STEPS} \
  --logging_steps=${LOGGING_STEPS} \
  --image_logging_steps=${IMAGE_LOGGING_STEPS} \
  --num_log_images=${NUM_LOG_IMAGES} \
  --mixed_precision="bf16" \
  --report_to="tensorboard" \
  --dataloader_num_workers=4 \
  --seed=${SEED}

# If using INIT_STUDENT_LORA_PATH, add:
#   --init_student_lora_path="${INIT_STUDENT_LORA_PATH}"
