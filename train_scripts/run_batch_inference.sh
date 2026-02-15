#!/bin/bash

# Batch Inference with Visualization
# Usage: bash run_batch_inference.sh

# Configuration
MODEL_PATH="/cache/SANA1.5_4.8B_1024px_diffusers"
LORA_PATH="/cache/new-sanaoutput/bridge_distill_3d_chibi/checkpoint-5000/pytorch_lora_weights.bin"
INPUT_DIR="/cache/omnic/3D_Chibi/src"
OUTPUT_DIR="./600inference_resultsvibt"
PROMPT="Convert the style to 3D Chibi Style"

# Inference parameters
NUM_INFERENCE_STEPS=1
GUIDANCE_SCALE=1.0
SEED=42
MAX_IMAGES=9  # Process first 9 images (set to empty for all)

# Inference backend: direct | concat
INFERENCE_MODE="direct"
# Conditioning: text | concat | concat_text (concat requires cond_proj_weights.bin next to LoRA or provided explicitly)
CONDITIONING="text"
# Optional: explicit path to cond_proj_weights.bin
COND_PROJ_PATH=""

# Booting noise (only meaningful for concat inference)
BOOTING_NOISE_SCALE=0.0

# Scheduler mode: model | vibt
SCHEDULER_MODE="vibt"
# ViBT scheduler params (only used when SCHEDULER_MODE="vibt")
VIBT_NOISE_SCALE=0.0
VIBT_SHIFT_GAMMA=5.0

# Run inference
CMD=(python batch_inference_visualize.py \
  --model_path="${MODEL_PATH}" \
  --lora_path="${LORA_PATH}" \
  --input_dir="${INPUT_DIR}" \
  --output_dir="${OUTPUT_DIR}" \
  --prompt="${PROMPT}" \
  --num_inference_steps=${NUM_INFERENCE_STEPS} \
  --inference_mode="${INFERENCE_MODE}" \
  --conditioning="${CONDITIONING}" \
  --booting_noise_scale=${BOOTING_NOISE_SCALE} \
  --scheduler_mode="${SCHEDULER_MODE}" \
  --vibt_noise_scale=${VIBT_NOISE_SCALE} \
  --vibt_shift_gamma=${VIBT_SHIFT_GAMMA} \
  --guidance_scale=${GUIDANCE_SCALE} \
  --seed=${SEED} \
  --max_images=${MAX_IMAGES} \
  --lora_rank=128 \
  --lora_alpha=128)

if [ -n "${COND_PROJ_PATH}" ]; then
  CMD+=(--cond_proj_path="${COND_PROJ_PATH}")
fi

"${CMD[@]}"

echo ""
echo "✅ Inference complete! Check results in: ${OUTPUT_DIR}"
echo "   - Comparisons: ${OUTPUT_DIR}/comparisons/"
echo "   - Results only: ${OUTPUT_DIR}/results_only/"
echo "   - Summary grid: ${OUTPUT_DIR}/summary_grid.png"
