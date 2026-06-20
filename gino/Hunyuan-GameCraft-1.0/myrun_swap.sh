#!/bin/bash
# Mid-rollout prompt-swap test. Usage: bash myrun_swap.sh {distill|std}
# Env knobs:
#   SWAP_PROMPT  new prompt to switch to
#   SWAP_AT      action index to switch at (0-based)
#   SWAP_RESET_REF=1  freeze the style anchor after the swap
#   GPU          which GPU (default 3)
set -e
cd "$(dirname "$0")"
MODE="${1:-distill}"
export PYTHONPATH="$(pwd):$PYTHONPATH"
export MODEL_BASE="weights/stdmodels"
export DISABLE_SP=1
export CPU_OFFLOAD=0
export CUDA_VISIBLE_DEVICES="${GPU:-3}"
PY="$(pwd)/.venv/bin/python"

# Default transition mirrors our Self-Forcing arc: village day -> ruined night.
export SWAP_PROMPT="${SWAP_PROMPT:-A ruined medieval village at night, crumbling houses, snow falling, eerie moonlight, abandoned and dark.}"
export SWAP_AT="${SWAP_AT:-3}"

if [ "$MODE" = "std" ]; then
    CKPT="weights/gamecraft_models/mp_rank_00_model_states.pt"
    STEPS=50; CFG=2.0; OUT="./results_swap_std"
else
    CKPT="weights/gamecraft_models/mp_rank_00_model_states_distill.pt"
    STEPS=8;  CFG=1.0; OUT="./results_swap_distill"
fi

# 6 actions so there is a runway before and after the swap at idx=3.
"$PY" -m torch.distributed.run --nnodes=1 --nproc_per_node=1 --master_port 29611 \
    hymm_sp/sample_swap.py \
    --image-path "asset/village.png" \
    --prompt "A charming medieval village with cobblestone streets, thatched-roof houses, and vibrant flower gardens under a bright blue sky." \
    --add-neg-prompt "overexposed, low quality, deformation, a poor composition, bad hands, bad teeth, bad eyes, bad limbs, distortion, blurring, text, subtitles, static, picture, black border." \
    --ckpt "$CKPT" \
    --video-size 704 1216 \
    --cfg-scale "$CFG" \
    --image-start \
    --action-list w w s s d d \
    --action-speed-list 0.2 0.2 0.2 0.2 0.2 0.2 \
    --seed 250160 \
    --infer-steps "$STEPS" \
    --flow-shift-eval-video 5.0 \
    --use-fp8 \
    --save-path "$OUT"
echo "DONE -> $OUT (swap to '$SWAP_PROMPT' at action $SWAP_AT)"
