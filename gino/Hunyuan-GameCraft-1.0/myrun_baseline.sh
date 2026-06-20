#!/bin/bash
# Baseline smoke test. Usage: bash myrun_baseline.sh {distill|std}
# Single free GPU, no sequence-parallel. Uses our dedicated .venv.
set -e
cd "$(dirname "$0")"
MODE="${1:-distill}"
export PYTHONPATH="$(pwd):$PYTHONPATH"
export MODEL_BASE="weights/stdmodels"
export DISABLE_SP=1
export CPU_OFFLOAD=0
GPU="${GPU:-3}"            # a free H100
export CUDA_VISIBLE_DEVICES="$GPU"
PY="$(pwd)/.venv/bin/python"

if [ "$MODE" = "std" ]; then
    CKPT="weights/gamecraft_models/mp_rank_00_model_states.pt"
    STEPS=50; CFG=2.0; OUT="./results_baseline_std"
else
    CKPT="weights/gamecraft_models/mp_rank_00_model_states_distill.pt"
    STEPS=8;  CFG=1.0; OUT="./results_baseline_distill"
fi

"$PY" -m torch.distributed.run --nnodes=1 --nproc_per_node=1 --master_port 29610 \
    hymm_sp/sample_batch.py \
    --image-path "asset/village.png" \
    --prompt "A charming medieval village with cobblestone streets, thatched-roof houses, and vibrant flower gardens under a bright blue sky." \
    --add-neg-prompt "overexposed, low quality, deformation, a poor composition, bad hands, bad teeth, bad eyes, bad limbs, distortion, blurring, text, subtitles, static, picture, black border." \
    --ckpt "$CKPT" \
    --video-size 704 1216 \
    --cfg-scale "$CFG" \
    --image-start \
    --action-list w s d a \
    --action-speed-list 0.2 0.2 0.2 0.2 \
    --seed 250160 \
    --infer-steps "$STEPS" \
    --flow-shift-eval-video 5.0 \
    --use-fp8 \
    --save-path "$OUT"
echo "DONE -> $OUT"
