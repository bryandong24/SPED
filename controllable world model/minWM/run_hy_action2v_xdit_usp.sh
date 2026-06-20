#!/usr/bin/env bash
# HunyuanVideo-1.5 Action2V DMD inference with xDiT USP sequence parallelism.
# Default: one 8-GPU single-prompt group using pure Ring sequence parallelism.
set -eo pipefail

DIR="$(cd "$(dirname "$0")"; pwd)"
cd "$DIR"

NUM_GPUS="${NUM_GPUS:-8}"
SP_SIZE="${SP_SIZE:-$NUM_GPUS}"
ULYSSES_DEGREE="${ULYSSES_DEGREE:-1}"
RING_DEGREE="${RING_DEGREE:-8}"
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29572}"

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  CUDA_VISIBLE_DEVICES="$(seq -s, 0 "$((NUM_GPUS - 1))")"
  export CUDA_VISIBLE_DEVICES
fi

export PATH="$DIR/.venv/bin:$PATH"
export PYTHONPATH="$DIR/HY15:$DIR/shared:$DIR/third_party/xDiT:$PYTHONPATH"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export NCCL_NET="${NCCL_NET:-Socket}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

TRANSFORMER_DIR="${TRANSFORMER_DIR:-./ckpts/HY15/Action2V/dmd}"
EXAMPLE_JSON="${EXAMPLE_JSON:-./assets/example_smoke.json}"
OUTPUT_DIR="${OUTPUT_DIR:-./outputs/hy_action2v_xdit_u${ULYSSES_DEGREE}_r${RING_DEGREE}}"
SEED="${SEED:-0}"
PROFILE_TIMING="${PROFILE_TIMING:-1}"
OVERWRITE="${OVERWRITE:-1}"
XDIT_ATTENTION_BACKEND="${XDIT_ATTENTION_BACKEND:-auto}"
VAE_DECODE_MODE="${VAE_DECODE_MODE:-tile_parallel}"
CONDITIONING_CACHE_SIZE="${CONDITIONING_CACHE_SIZE:-8}"
SERVE_STDIN="${SERVE_STDIN:-0}"
MAX_REQUESTS="${MAX_REQUESTS:-0}"

cmd=(
  torchrun
  --nnodes "$NNODES"
  --node_rank "$NODE_RANK"
  --nproc_per_node "$NUM_GPUS"
  --master_addr "$MASTER_ADDR"
  --master_port "$MASTER_PORT"
  HY15/hy15_inference.py
  --mode ar_rollout
  --parallel_mode xdit_usp
  --sp_size "$SP_SIZE"
  --ulysses_degree "$ULYSSES_DEGREE"
  --ring_degree "$RING_DEGREE"
  --xdit_attention_backend "$XDIT_ATTENTION_BACKEND"
  --vae_decode_mode "$VAE_DECODE_MODE"
  --conditioning_cache_size "$CONDITIONING_CACHE_SIZE"
  --seed "$SEED"
  --use_camera
  --transformer_dir "$TRANSFORMER_DIR"
  --example_json "$EXAMPLE_JSON"
  --output_dir "$OUTPUT_DIR"
  --num_inference_steps 4
  --shift 5.0
  --guidance_scale 1.0
  --fps 16
  --chunk_latent_frames 4
  --stabilization_level 1
)

if [[ -n "${MODEL_PATH:-}" ]]; then
  cmd+=(--model_path "$MODEL_PATH")
fi

case "$PROFILE_TIMING" in
  1|true|TRUE|yes|YES) cmd+=(--profile_timing) ;;
esac

case "$OVERWRITE" in
  1|true|TRUE|yes|YES) cmd+=(--overwrite) ;;
esac

case "$SERVE_STDIN" in
  1|true|TRUE|yes|YES) cmd+=(--serve_stdin --max_requests "$MAX_REQUESTS") ;;
esac

cmd+=("$@")

echo "=== HY Action2V xDiT USP inference ==="
echo "  NUM_GPUS=$NUM_GPUS SP_SIZE=$SP_SIZE ULYSSES_DEGREE=$ULYSSES_DEGREE RING_DEGREE=$RING_DEGREE"
echo "  NNODES=$NNODES NODE_RANK=$NODE_RANK CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "  XDIT_ATTENTION_BACKEND=$XDIT_ATTENTION_BACKEND VAE_DECODE_MODE=$VAE_DECODE_MODE"
echo "  PROFILE_TIMING=$PROFILE_TIMING OVERWRITE=$OVERWRITE SERVE_STDIN=$SERVE_STDIN"
echo "  transformer=$TRANSFORMER_DIR"
echo "  examples=$EXAMPLE_JSON"
echo "  output=$OUTPUT_DIR"
echo ""

"${cmd[@]}"
