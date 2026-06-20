#!/usr/bin/env bash
# HY15 Action2V (4-step DMD) — NATIVE sequence-parallel (Ulysses all_to_all), N GPUs.
# Node fix: this A3 VM's NCCL defaults to the gIB RDMA net plugin (no RDMA device here),
# which breaks all_to_all. Override with NCCL_NET=Socket so the NET proxy works while
# the actual GPU<->GPU transfers use NVLink P2P.
set -eo pipefail
DIR="$(cd "$(dirname "$0")"; pwd)"; cd "$DIR"
NUM_GPUS="${NUM_GPUS:-4}"; SP_SIZE="${SP_SIZE:-$NUM_GPUS}"
export PATH="$DIR/.venv/bin:$PATH"
export PYTHONPATH="$DIR/HY15:$DIR/shared:$PYTHONPATH"
export NCCL_NET=Socket   # MUST override the node's base NCCL_NET=gIB (no RDMA dev here)
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export HY_SDPA_BACKEND="${HY_SDPA_BACKEND:-default}"  # AR rollout: flash(default) beats cuDNN
export TOKENIZERS_PARALLELISM=false
[ -z "${CUDA_VISIBLE_DEVICES:-}" ] && export CUDA_VISIBLE_DEVICES="$(seq -s, 0 $((NUM_GPUS-1)))"
TRANSFORMER_DIR="${TRANSFORMER_DIR:-./ckpts/HY15/Action2V/dmd}"
EXAMPLE_JSON="${EXAMPLE_JSON:-./assets/example_smoke.json}"
OUTPUT_DIR="${OUTPUT_DIR:-./outputs/hy_action2v_sp${SP_SIZE}_${HY_SDPA_BACKEND}}"
echo "[sp] NUM_GPUS=$NUM_GPUS SP_SIZE=$SP_SIZE SDPA=$HY_SDPA_BACKEND NCCL_NET=$NCCL_NET GPUS=$CUDA_VISIBLE_DEVICES"
torchrun --nnodes=1 --nproc_per_node="$NUM_GPUS" --master_port "${MASTER_PORT:-29572}" \
  HY15/hy15_inference.py --mode ar_rollout --use_camera \
  --parallel_mode sp --sp_size "$SP_SIZE" \
  --transformer_dir "$TRANSFORMER_DIR" --example_json "$EXAMPLE_JSON" \
  --output_dir "$OUTPUT_DIR" --overwrite --profile_timing \
  --num_inference_steps 4 --shift 5.0 --guidance_scale 1.0 \
  --fps 16 --chunk_latent_frames 4 --stabilization_level 1 --seed 0 "$@"
