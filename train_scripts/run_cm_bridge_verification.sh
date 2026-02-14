#!/bin/bash
# Run CM-Bridge conversion verification

# Configuration
MODEL_PATH="/cache/SANA1.5_4.8B_1024px_diffusers"
LORA_PATH="/cache/boot2sanaoutput/bridge_3d_chibi_sweep/ns_0.5_boot_0.2/checkpoint-5000/pytorch_lora_weights.bin"
COND_PROJ_PATH="/cache/boot2sanaoutput/bridge_3d_chibi_sweep/ns_0.5_boot_0.2/checkpoint-5000/cond_proj_weights.bin"
SOURCE_IMAGE="/cache/omnic/3D_Chibi/src/041.png"
OUTPUT_DIR="./cm_bridge_verification"

# Optional parameters
PROMPT="Convert the style to 3D Chibi Style"
NUM_STEPS=28
SIGMA_DATA=1.0
SEED=42

echo "=================================================="
echo "CM-Bridge Conversion Verification"
echo "=================================================="
echo ""
echo "Model: $MODEL_PATH"
echo "LoRA: $LORA_PATH"
echo "Concat Adapter: $COND_PROJ_PATH"
echo "Source: $SOURCE_IMAGE"
echo "Output: $OUTPUT_DIR"
echo ""

python verify_cm_bridge_conversion.py \
    --model_path "$MODEL_PATH" \
    --lora_path "$LORA_PATH" \
    --cond_proj_path "$COND_PROJ_PATH" \
    --source_image "$SOURCE_IMAGE" \
    --prompt "$PROMPT" \
    --output_dir "$OUTPUT_DIR" \
    --num_steps $NUM_STEPS \
    --sigma_data $SIGMA_DATA \
    --seed $SEED

echo ""
echo "=================================================="
echo "Verification complete! Check results in:"
echo "$OUTPUT_DIR"
echo "=================================================="
