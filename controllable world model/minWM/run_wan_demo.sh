#!/usr/bin/env bash
# minWM — Wan Action2V (4-step DMD) inference, single-GPU.
#
# Why this exists instead of Wan21/scripts/inference/run_infer_causal_camera.sh:
#   * Runs via direct `python` (NOT torchrun): on a single GPU torchrun still sets
#     LOCAL_RANK, which makes wan_inference.py init an NCCL process group. NCCL fails
#     in this environment ("Failed to initialize any NET plugin"). Plain python takes
#     the non-distributed branch and never touches NCCL.
#   * Relies on the SDPA attention fallback (see RUN_NOTES.md) — no flash-attn build.
#
# Override any of the env vars below. Defaults run all 30 demo prompts with the
# matching camera trajectories.
set -eo pipefail
DIR="$(cd "$(dirname "$0")"; pwd)"
cd "$DIR"
export PATH="$DIR/.venv/bin:$PATH"
export PYTHONPATH="$DIR/HY15:$DIR/Wan21:$DIR/shared:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

CONFIG_PATH="${CONFIG_PATH:-Wan21/configs/causal_forcing_dmd_camera.yaml}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-./ckpts/Wan21/Action2V/dmd/model.pt}"
DATA_PATH="${DATA_PATH:-Wan21/prompts/demos.txt}"
TRAJECTORY_PATH="${TRAJECTORY_PATH:-Wan21/prompts/trajectories.txt}"
OUTPUT_FOLDER="${OUTPUT_FOLDER:-./outputs/wan_action2v}"

echo "config=$CONFIG_PATH ckpt=$CHECKPOINT_PATH data=$DATA_PATH traj=$TRAJECTORY_PATH out=$OUTPUT_FOLDER gpu=$CUDA_VISIBLE_DEVICES"
python Wan21/wan_inference.py \
  --config_path "$CONFIG_PATH" \
  --output_folder "$OUTPUT_FOLDER" \
  --checkpoint_path "$CHECKPOINT_PATH" \
  --data_path "$DATA_PATH" \
  --sp_size 1 \
  --trajectory_path "$TRAJECTORY_PATH"
echo "Done. Videos in: $OUTPUT_FOLDER"
