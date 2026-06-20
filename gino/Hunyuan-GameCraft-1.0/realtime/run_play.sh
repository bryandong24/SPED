#!/bin/bash
# Launch the live GameCraft web demo (N-way sequence parallel).
# Usage: GPUS=0,1,2,3 H=384 W=672 WEBPORT=8080 bash realtime/run_play.sh
set -e
cd "$(dirname "$0")/.."          # repo root
export PYTHONPATH="$(pwd):$PYTHONPATH"
export MODEL_BASE="weights/stdmodels"

GPUS="${GPUS:-0,1,2,3}"
export CUDA_VISIBLE_DEVICES="$GPUS"
NUM_GPU=$(awk -F',' '{print NF}' <<< "$GPUS")
H="${H:-384}"; W="${W:-672}"
export PORT="${WEBPORT:-8080}"
export DEFAULT_ACTION="${DEFAULT_ACTION:-w}"
export DEFAULT_SPEED="${DEFAULT_SPEED:-0.2}"

export DISABLE_SP=0
export CPU_OFFLOAD=0
# GCP A3-Ultra: avoid the gIB NCCL plugin for single-node SP
export NCCL_NET=Socket
export NCCL_IB_DISABLE=1
unset NCCL_TUNER_CONFIG_PATH NCCL_NET_GDR_LEVEL LD_LIBRARY_PATH 2>/dev/null || true

PY="$(pwd)/.venv/bin/python"
echo "[play] GPUS=$GPUS ($NUM_GPU-way SP) size=${H}x${W} web=http://0.0.0.0:$PORT"
exec "$PY" -m torch.distributed.run --nnodes=1 --nproc_per_node=$NUM_GPU --master_port "${MPORT:-29650}" \
    realtime/web_play.py \
    --image-path "asset/village.png" \
    --prompt "A charming medieval village with cobblestone streets, thatched-roof houses, and vibrant flower gardens under a bright blue sky." \
    --add-neg-prompt "overexposed, low quality, deformation, a poor composition, bad hands, bad teeth, bad eyes, bad limbs, distortion, blurring, text, subtitles, static, picture, black border." \
    --ckpt "weights/gamecraft_models/mp_rank_00_model_states_distill.pt" \
    --video-size "$H" "$W" \
    --cfg-scale 1.0 \
    --image-start \
    --action-list w \
    --action-speed-list 0.2 \
    --seed 250160 \
    --infer-steps "${STEPS:-8}" \
    --flow-shift-eval-video 5.0 \
    --use-fp8 \
    --save-path "/tmp/gc_play"
