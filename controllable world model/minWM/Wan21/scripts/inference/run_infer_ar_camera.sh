#!/bin/bash
set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/../../.."; pwd)"
cd "$PROJECT_ROOT"

export NCCL_DEBUG=WARN

# ===== Paths =====
CONFIG_PATH="${CONFIG_PATH:-Wan21/configs/ar_camera_tf.yaml}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-./ckpts/Wan21/Action2V/ar_diffusion_tf/model.pt}"
DATA_PATH="${DATA_PATH:-Wan21/prompts/demos.txt}"
OUTPUT_FOLDER="${OUTPUT_FOLDER:-output/ar_camera}"
SP_SIZE="${SP_SIZE:-1}"

# ===== Camera Trajectory =====
TRAJECTORY="${TRAJECTORY:-w*19}"
TRAJECTORY_PATH="${TRAJECTORY_PATH:-}"

# Build trajectory argument
if [ -n "$TRAJECTORY_PATH" ]; then
  TRAJ_ARGS="--trajectory_path $TRAJECTORY_PATH"
else
  TRAJ_ARGS="--trajectory $TRAJECTORY"
fi

NUM_GPUS_PER_NODE=1
NNODES=${WORLD_SIZE:-1}
NODE_RANK=${RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-"localhost"}
MASTER_PORT=${MASTER_PORT:-29621}

echo "=== Inference: AR Camera Control ==="
echo "  Config:     $CONFIG_PATH"
echo "  Checkpoint: $CHECKPOINT_PATH"
echo "  Output:     $OUTPUT_FOLDER"

export SP_SIZE=$SP_SIZE
torchrun \
  --master_addr=$MASTER_ADDR \
  --master_port=$MASTER_PORT \
  --nproc_per_node=$NUM_GPUS_PER_NODE \
  --nnodes=$NNODES \
  --node_rank=$NODE_RANK \
  Wan21/wan_inference.py \
  --config_path "$CONFIG_PATH" \
  --output_folder "$OUTPUT_FOLDER" \
  --checkpoint_path "$CHECKPOINT_PATH" \
  --data_path "$DATA_PATH" \
  --sp_size $SP_SIZE \
  $TRAJ_ARGS
