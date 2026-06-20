#!/bin/bash
# Deterministic multi-GPU SP swap run (fresh from image, fixed swap point) for A/B science.
# Env: GPUS H W SWAP_PROMPT SWAP_AT SWAP_HIST SWAP_GT SWAP_MASK OUT PORT
set -e
cd "$(dirname "$0")"
export PYTHONPATH="$(pwd):$PYTHONPATH"
export MODEL_BASE="weights/stdmodels"
GPUS="${GPUS:-0,1,2,3}"; export CUDA_VISIBLE_DEVICES="$GPUS"
NUM_GPU=$(awk -F',' '{print NF}' <<< "$GPUS")
H="${H:-256}"; W="${W:-448}"
export DISABLE_SP=0 CPU_OFFLOAD=0
export NCCL_NET=Socket NCCL_IB_DISABLE=1
unset NCCL_TUNER_CONFIG_PATH NCCL_NET_GDR_LEVEL LD_LIBRARY_PATH 2>/dev/null || true
export SWAP_PROMPT="${SWAP_PROMPT:-a ruined abandoned medieval village at night, heavy snow falling, dark eerie blue moonlight, broken crumbling houses}"
export SWAP_AT="${SWAP_AT:-3}"
export SWAP_HIST="${SWAP_HIST:-1.0}" SWAP_GT="${SWAP_GT:-1.0}" SWAP_MASK="${SWAP_MASK:-1.0}"
OUT="${OUT:-./res_swapfast}"
PY="$(pwd)/.venv/bin/python"
echo "[swap-fast] GPUS=$GPUS size=${H}x${W} swap@${SWAP_AT} levers(h=$SWAP_HIST g=$SWAP_GT m=$SWAP_MASK) -> $OUT"
"$PY" -m torch.distributed.run --nnodes=1 --nproc_per_node=$NUM_GPU --master_port "${PORT:-29660}" \
    hymm_sp/sample_swap.py \
    --image-path "asset/village.png" \
    --prompt "A charming medieval village with cobblestone streets, thatched-roof houses, vibrant flower gardens, bright blue sky, warm golden sunlight." \
    --add-neg-prompt "overexposed, low quality, deformation, a poor composition, bad hands, bad teeth, bad eyes, bad limbs, distortion, blurring, text, subtitles, static, picture, black border." \
    --ckpt "weights/gamecraft_models/mp_rank_00_model_states_distill.pt" \
    --video-size "$H" "$W" \
    --cfg-scale 1.0 --image-start \
    --action-list w w w w w w w w w \
    --action-speed-list 0.2 0.2 0.2 0.2 0.2 0.2 0.2 0.2 0.2 \
    --seed 250160 --infer-steps 8 --flow-shift-eval-video 5.0 --use-fp8 \
    --save-path "$OUT"
echo "DONE -> $OUT"
