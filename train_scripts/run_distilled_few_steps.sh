#!/bin/bash
set -e

# Few-step inference for distilled student (LoRA + cond_proj)

# -------- Paths --------
MODEL_PATH="/cache/SANA1.5_4.8B_1024px_diffusers"
DISTILL_OUTPUT="/cache/sanaoutput/bridge_scm_ladd_distill_3d_chibi"  # can be output dir or a weights file
CHECKPOINT="latest"  # latest | final | checkpoint-8500 | 8500

INPUT_DIR="/cache/omnic/3D_Chibi/src"
OUTPUT_DIR="./fewstep_distilled_out"

# -------- Inference knobs --------
PROMPT="Convert the style to 3D Chibi Style"
NUM_INFERENCE_STEPS=2
GUIDANCE_SCALE=1.0
SEED=42
MAX_IMAGES=9

# Conditioning
CONDITIONING="concat_text"         # text | concat | concat_text
BOOTING_NOISE_SCALE=0.0             # used only for concat/concat_text (set >0 only if you intentionally want booting noise)
COND_PROJ_PATH=""                   # leave empty to auto-detect next to checkpoint

# Scheduler
SCHEDULER_MODE="vibt"               # model | vibt
VIBT_NOISE_SCALE=0.0                # ODE bridge: keep 0.0
VIBT_SHIFT_GAMMA=5.0

# LoRA config (must match training)
LORA_RANK=128
LORA_ALPHA=128

CMD=(python infer_distilled_few_steps.py \
  --model_path "${MODEL_PATH}" \
  --distill_output "${DISTILL_OUTPUT}" \
  --checkpoint "${CHECKPOINT}" \
  --input_dir "${INPUT_DIR}" \
  --output_dir "${OUTPUT_DIR}" \
  --prompt "${PROMPT}" \
  --num_inference_steps ${NUM_INFERENCE_STEPS} \
  --guidance_scale ${GUIDANCE_SCALE} \
  --seed ${SEED} \
  --max_images ${MAX_IMAGES} \
  --conditioning "${CONDITIONING}" \
  --booting_noise_scale ${BOOTING_NOISE_SCALE} \
  --scheduler_mode "${SCHEDULER_MODE}" \
  --vibt_noise_scale ${VIBT_NOISE_SCALE} \
  --vibt_shift_gamma ${VIBT_SHIFT_GAMMA} \
  --lora_rank ${LORA_RANK} \
  --lora_alpha ${LORA_ALPHA})

if [ -n "${COND_PROJ_PATH}" ]; then
  CMD+=(--cond_proj_path "${COND_PROJ_PATH}")
fi

"${CMD[@]}"

echo ""
echo "✅ Done. See: ${OUTPUT_DIR}"
echo "   - ${OUTPUT_DIR}/comparisons"
echo "   - ${OUTPUT_DIR}/results_only"
