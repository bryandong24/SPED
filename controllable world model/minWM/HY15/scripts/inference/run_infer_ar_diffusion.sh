set -e
PROJECT_ROOT="$(cd "$(dirname "$0")/../../.."; pwd)"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT/HY15:$PROJECT_ROOT/shared:$PYTHONPATH"
export NCCL_DEBUG=WARN

# Download: huggingface-cli download MIN-Lab/minMW --local-dir ./ckpts
TRANSFORMER_DIR="${TRANSFORMER_DIR:-./ckpts/HY15/TI2V/ar_diffusion_tf}"
EXAMPLE_JSON="${EXAMPLE_JSON:-./assets/example.json}"
OUTPUT_DIR="${OUTPUT_DIR:-./outputs/eval_ar_rollout}"

# model_path is auto-detected from HF cache if not set

NUM_GPUS=1

torchrun \
    --nproc_per_node="$NUM_GPUS" \
    HY15/hy15_inference.py \
    --mode ar_rollout \
    --transformer_dir "$TRANSFORMER_DIR" \
    ${MODEL_PATH:+--model_path "$MODEL_PATH"} \
    --example_json "$EXAMPLE_JSON" \
    --output_dir "$OUTPUT_DIR" \
    --num_inference_steps 50 \
    --shift 5.0 \
    --guidance_scale 6.0 \
    --fps 16 \
    --stabilization_level 1
