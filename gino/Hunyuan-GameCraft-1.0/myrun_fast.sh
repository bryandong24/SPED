#!/bin/bash
# GameCraft fast path: multi-GPU sequence-parallel + lower resolution + 8-step distill + fp8.
# Usage: GPUS=2,3,5,6 H=384 W=672 bash myrun_fast.sh
set -e
cd "$(dirname "$0")"
export PYTHONPATH="$(pwd):$PYTHONPATH"
export MODEL_BASE="weights/stdmodels"

GPUS="${GPUS:-2,3,5,6}"
export CUDA_VISIBLE_DEVICES="$GPUS"
NUM_GPU=$(awk -F',' '{print NF}' <<< "$GPUS")
H="${H:-384}"; W="${W:-672}"
STEPS="${STEPS:-8}"
OUT="${OUT:-./results_fast_${H}x${W}_sp${NUM_GPU}}"

# Sequence parallel ON across the visible GPUs.
export DISABLE_SP=0
export CPU_OFFLOAD=0

# This is a GCP A3-Ultra node with NCCL_NET=gIB (GPUDirect) set globally, which the
# venv NCCL can't load -> "network gIB not found". For single-node intra-node SP we only
# need NVLink/P2P + SHM, so force standard transport and drop the gIB-specific config.
export NCCL_NET=Socket
export NCCL_IB_DISABLE=1
unset NCCL_TUNER_CONFIG_PATH NCCL_NET_GDR_LEVEL LD_LIBRARY_PATH 2>/dev/null || true

checkpoint_path="weights/gamecraft_models/mp_rank_00_model_states_distill.pt"
PY="$(pwd)/.venv/bin/python"

echo "[fast] GPUS=$GPUS NUM_GPU=$NUM_GPU size=${H}x${W} steps=$STEPS out=$OUT"
"$PY" -m torch.distributed.run --nnodes=1 --nproc_per_node=$NUM_GPU --master_port "${PORT:-29620}" \
    hymm_sp/sample_fast.py \
    --image-path "asset/village.png" \
    --prompt "A charming medieval village with cobblestone streets, thatched-roof houses, and vibrant flower gardens under a bright blue sky." \
    --add-neg-prompt "overexposed, low quality, deformation, a poor composition, bad hands, bad teeth, bad eyes, bad limbs, distortion, blurring, text, subtitles, static, picture, black border." \
    --ckpt "$checkpoint_path" \
    --video-size "$H" "$W" \
    --cfg-scale 1.0 \
    --image-start \
    --action-list w s d a \
    --action-speed-list 0.2 0.2 0.2 0.2 \
    --seed 250160 \
    --infer-steps "$STEPS" \
    --flow-shift-eval-video 5.0 \
    --use-fp8 \
    --save-path "$OUT"
echo "DONE -> $OUT"
