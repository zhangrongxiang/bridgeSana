#!/bin/bash

# Batch Inference (CM Bridge Timestep Conversion) with Visualization
# Usage:
#   bash run_batch_inference_cm_bridge.sh

set -euo pipefail

# IMPORTANT: activate your env first
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate rcm

# Configuration (copied from run_batch_inference.sh)
MODEL_PATH="/cache/SANA1.5_4.8B_1024px_diffusers"
LORA_PATH="/cache/boot2sanaoutput/bridge_3d_chibi_sweep/ns_0.5_boot_0.2/checkpoint-5000/pytorch_lora_weights.bin"
INPUT_DIR="/cache/omnic/3D_Chibi/src"
OUTPUT_DIR="./600inference_resultsvibt_cm_bridge"
PROMPT="Convert the style to 3D Chibi Style"

# Inference parameters
RESOLUTION=1024
NUM_INFERENCE_STEPS=28
GUIDANCE_SCALE=1.0
SEED=42
MAX_IMAGES=9

# concat_text injection settings
# If empty, the python script will auto-detect `${LORA_DIR}/cond_proj_weights.bin`
COND_PROJ_PATH=""
BOOTING_NOISE_SCALE=0.4

# ViBT scheduler params
VIBT_NOISE_SCALE=0.0
VIBT_SHIFT_GAMMA=5.0

# CM conversion settings
TRIGFLOW_SOURCE_STEPS=128
TRIGFLOW_MAX_T=1.57080

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

CMD=(python "${SCRIPT_DIR}/batch_inference_visualize_cm_bridge.py" \
  --model_path="${MODEL_PATH}" \
  --lora_path="${LORA_PATH}" \
  --input_dir="${INPUT_DIR}" \
  --output_dir="${OUTPUT_DIR}" \
  --prompt="${PROMPT}" \
  --resolution=${RESOLUTION} \
  --num_inference_steps=${NUM_INFERENCE_STEPS} \
  --guidance_scale=${GUIDANCE_SCALE} \
  --seed=${SEED} \
  --max_images=${MAX_IMAGES} \
  --booting_noise_scale=${BOOTING_NOISE_SCALE} \
  --vibt_noise_scale=${VIBT_NOISE_SCALE} \
  --vibt_shift_gamma=${VIBT_SHIFT_GAMMA} \
  --trigflow_source_steps=${TRIGFLOW_SOURCE_STEPS} \
  --trigflow_max_t=${TRIGFLOW_MAX_T} \
  --lora_rank=128 \
  --lora_alpha=128)

if [ -n "${COND_PROJ_PATH}" ]; then
  CMD+=(--cond_proj_path="${COND_PROJ_PATH}")
fi

"${CMD[@]}"

echo ""
echo "✅ CM Bridge inference complete! Check results in: ${OUTPUT_DIR}"
echo "   - Triplets: ${OUTPUT_DIR}/triplets/"
echo "   - Baseline (ViBT native): ${OUTPUT_DIR}/baseline/"
echo "   - CM (TrigFlow→Linear→ViBT): ${OUTPUT_DIR}/cm_bridge/"
