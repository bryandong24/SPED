set -e
PROJECT_ROOT="$(cd "$(dirname "$0")/../../.."; pwd)"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT/HY15:$PROJECT_ROOT/shared:$PYTHONPATH"
export NCCL_DEBUG=WARN

# Download: huggingface-cli download MIN-Lab/minMW --local-dir ./ckpts
TRANSFORMER_DIR="${TRANSFORMER_DIR:-./ckpts/HY15/Action2V/ar_diffusion_tf}"
EXAMPLE_JSON="${EXAMPLE_JSON:-./assets/example.json}"
OUTPUT_DIR="${OUTPUT_DIR:-./outputs/eval_ar_camera}"

NUM_GPUS=1

echo "=== AR Diffusion Camera Inference ==="
echo "  GPUs: $NUM_GPUS"
echo "  JSON: $EXAMPLE_JSON"
echo "  Output: $OUTPUT_DIR"
echo "  Trajectory: read from JSON per sample"
echo ""

torchrun \
    --nproc_per_node="$NUM_GPUS" \
    --master_port "${MASTER_PORT:-29504}" \
    HY15/hy15_inference.py \
    --mode ar_rollout \
    --use_camera \
    --transformer_dir "$TRANSFORMER_DIR" \
    ${MODEL_PATH:+--model_path "$MODEL_PATH"} \
    --example_json "$EXAMPLE_JSON" \
    --output_dir "$OUTPUT_DIR" \
    --num_inference_steps 50 \
    --shift 5.0 \
    --guidance_scale 6.0 \
    --fps 16 \
    --chunk_latent_frames 4 \
    --stabilization_level 1
