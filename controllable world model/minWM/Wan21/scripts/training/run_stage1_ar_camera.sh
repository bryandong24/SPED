#!/bin/bash
set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/../../.."; pwd)"
cd "$PROJECT_ROOT"

export WANDB_MODE=offline
export NCCL_DEBUG=WARN

NUM_GPUS_PER_NODE=8
NNODES=${WORLD_SIZE:-1}
NODE_RANK=${RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-"localhost"}
MASTER_PORT=${MASTER_PORT:-29601}
TOTAL_GPUS=$((NUM_GPUS_PER_NODE * NNODES))

echo "=== Stage 1: AR Camera Control with Teacher Forcing ==="
echo "  NNODES: $NNODES, NODE_RANK: $NODE_RANK, TOTAL_GPUS: $TOTAL_GPUS"

torchrun \
  --master_addr=$MASTER_ADDR \
  --master_port=$MASTER_PORT \
  --nproc_per_node=$NUM_GPUS_PER_NODE \
  --nnodes=$NNODES \
  --node_rank=$NODE_RANK \
  Wan21/wan_train.py \
  --config_path Wan21/configs/ar_camera_tf.yaml \
  --logdir logs/ar_camera_tf \
  --sp_size 1
