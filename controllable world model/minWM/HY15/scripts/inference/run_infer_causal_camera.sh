set -e
PROJECT_ROOT="$(cd "$(dirname "$0")/../../.."; pwd)"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT/HY15:$PROJECT_ROOT/shared:$PYTHONPATH"
export NCCL_DEBUG=WARN

# Download: huggingface-cli download MIN-Lab/minMW --local-dir ./ckpts
TRANSFORMER_DIR="${TRANSFORMER_DIR:-./ckpts/HY15/Action2V/dmd}"
EXAMPLE_JSON="${EXAMPLE_JSON:-./assets/example.json}"
OUTPUT_DIR="${OUTPUT_DIR:-./outputs/eval_causal_camera}"

# Detect available GPUs
NUM_GPUS=1

echo "=== Causal Camera Inference ==="
echo "  GPUs: $NUM_GPUS"
echo "  JSON: $EXAMPLE_JSON"
echo "  Output: $OUTPUT_DIR"
echo ""

torchrun \
    --nproc_per_node="$NUM_GPUS" \
    --master_port "${MASTER_PORT:-29502}" \
    HY15/hy15_inference.py \
    --mode ar_rollout \
    --use_camera \
    --transformer_dir "$TRANSFORMER_DIR" \
    ${MODEL_PATH:+--model_path "$MODEL_PATH"} \
    --example_json "$EXAMPLE_JSON" \
    --output_dir "$OUTPUT_DIR" \
    --num_inference_steps 4 \
    --shift 5.0 \
    --guidance_scale 1.0 \
    --fps 16 \
    --chunk_latent_frames 4 \
    --stabilization_level 1
