#!/usr/bin/env bash
# minWM — HunyuanVideo-1.5 Action2V (4-step DMD, camera/ProPE control), single-GPU.
# Direct `python` (not torchrun): HY uses a gloo single-process group, so NCCL is never
# touched. HY's ar_rollout path uses torch SDPA attention — no flash-attn needed.
set -eo pipefail
DIR="$(cd "$(dirname "$0")"; pwd)"
cd "$DIR"
export PATH="$DIR/.venv/bin:$PATH"
export PYTHONPATH="$DIR/HY15:$DIR/shared:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

TRANSFORMER_DIR="${TRANSFORMER_DIR:-./ckpts/HY15/Action2V/dmd}"
EXAMPLE_JSON="${EXAMPLE_JSON:-./assets/example.json}"
OUTPUT_DIR="${OUTPUT_DIR:-./outputs/hy_action2v}"

echo "transformer=$TRANSFORMER_DIR example=$EXAMPLE_JSON out=$OUTPUT_DIR gpu=$CUDA_VISIBLE_DEVICES"
python HY15/hy15_inference.py \
  --mode ar_rollout \
  --use_camera \
  --transformer_dir "$TRANSFORMER_DIR" \
  --example_json "$EXAMPLE_JSON" \
  --output_dir "$OUTPUT_DIR" \
  --num_inference_steps 4 --shift 5.0 --guidance_scale 1.0 \
  --fps 16 --chunk_latent_frames 4 --stabilization_level 1
echo "Done. Videos in: $OUTPUT_DIR"
