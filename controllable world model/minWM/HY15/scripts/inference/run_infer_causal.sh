set -e
PROJECT_ROOT="$(cd "$(dirname "$0")/../../.."; pwd)"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT/HY15:$PROJECT_ROOT/shared:$PYTHONPATH"
export NCCL_DEBUG=WARN

# Download: huggingface-cli download MIN-Lab/minMW --local-dir ./ckpts
# Use causal_cd or causal_ode or dmd depending on the stage
TRANSFORMER_DIR="${TRANSFORMER_DIR:-./ckpts/HY15/TI2V/causal_cd}"
EXAMPLE_JSON="${EXAMPLE_JSON:-./assets/example.json}"
OUTPUT_DIR="${OUTPUT_DIR:-./outputs/eval_arfs_rollout}"

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
    --num_inference_steps 4 \
    --shift 5.0 \
    --guidance_scale 1.0 \
    --fps 16 \
    --stabilization_level 1
