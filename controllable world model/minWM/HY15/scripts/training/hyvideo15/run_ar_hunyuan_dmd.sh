#!/bin/bash
set -e

# Self-Forcing DMD Distillation Training Script for HY1.5 TI2V

PROJECT_ROOT="$(cd "$(dirname "$0")"; while [ "$PWD" != "/" ] && [ ! -f "requirements.txt" ]; do cd ..; done; pwd)"
cd "$PROJECT_ROOT"

export TORCH_NCCL_TRACE_BUFFER_SIZE=1048576
export TORCH_NCCL_DUMP_ON_TIMEOUT=1
export TORCH_NCCL_DEBUG_INFO_TEMP_FILE=/tmp/nccl_trace_rank_

export NCCL_DEBUG=INFO
export WANDB_BASE_URL="https://api.wandb.ai"
# export WANDB_MODE=online
export WANDB_MODE=offline
export TOKENIZERS_PARALLELISM=false
# export NCCL_P2P_DISABLE=1
export TORCH_NCCL_ENABLE_MONITORING=0

# CFG negative prompt embeddings (readme §4.2.1 (2) → ./dataset/others/HY/TI2V/)
export NEG_PROMPT_PT="${NEG_PROMPT_PT:-./dataset/others/HY/TI2V/hunyuan_neg_prompt.pt}"
export NEG_BYT5_PT="${NEG_BYT5_PT:-./dataset/others/HY/TI2V/hunyuan_neg_byt5_prompt.pt}"

# ===== Paths (relative to project root, see readme §2.1 / §4.2.1 / §4.2.2 Stage 2 / Stage 3) =====
# HY1.5 base transformer (readme §2.1)
TRANSFORMER_DIR="${TRANSFORMER_DIR:-./ckpts/HunyuanVideo-1.5/transformer/480p_i2v}"
echo "Using transformer directory: $TRANSFORMER_DIR"

# ==================== Model Paths ====================
GENERATOR_MODEL_PATH="$TRANSFORMER_DIR"
REAL_SCORE_MODEL_PATH="$TRANSFORMER_DIR"
FAKE_SCORE_MODEL_PATH="$TRANSFORMER_DIR"

# Student (generator) init = Stage 2 产物（readme §4.2.2 Stage 3 (1)，causal_ode or causal_cd）
AR_ACTION_LOAD_FROM_DIR="${AR_ACTION_LOAD_FROM_DIR:-./ckpts/HY15/TI2V/causal_ode/diffusion_pytorch_model.safetensors}"
# Teacher (real score) = Phase 1 多步 bidirectional 产物（readme §4.2.1 落点）
TEACHER_MODEL_PATH="${TEACHER_MODEL_PATH:-./ckpts/HY15/TI2V/bidirectional/diffusion_pytorch_model.safetensors}"

# ==================== Dataset Paths ====================
# SFT latents (readme §4.2.1 (2)) — Stage 3 self rollout 仅消耗条件部分
JSON_PATH="${JSON_PATH:-./dataset/HY15/TI2V/train_index.json}"
# Dummy data root（与 JSON 同根即可，DMD 训练只用 JSON_PATH 里的条件）
DUMMY_DATA_DIR="${DUMMY_DATA_DIR:-./dataset/HY15/TI2V}"
# Training output（与 §3.2 quickstart 推理路径对齐 → ./ckpts/HY15/TI2V/dmd）
OUTPUT_DIR="${OUTPUT_DIR:-./ckpts/HY15/TI2V/dmd}"
RESUME_CKPT="${RESUME_CKPT:-}"

# ===== Multi-node configuration =====
NUM_GPUS_PER_NODE=8
NNODES=${WORLD_SIZE:-1}
NODE_RANK=${RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-"localhost"}
MASTER_PORT=${MASTER_PORT:-29503}

TOTAL_GPUS=$((NUM_GPUS_PER_NODE * NNODES))

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

echo "=== Multi-node config ==="
echo "  NNODES: $NNODES"
echo "  NODE_RANK: $NODE_RANK"
echo "  MASTER_ADDR: $MASTER_ADDR"
echo "  MASTER_PORT: $MASTER_PORT"
echo "  TOTAL_GPUS: $TOTAL_GPUS"
echo "========================="

# ==================== Training Arguments ====================
training_args=(
  --json_path "$JSON_PATH"
  --data_path "$DUMMY_DATA_DIR"
  --num_latent_t 20
  --output_dir "$OUTPUT_DIR"
  --max_train_steps 4000
  --train_batch_size 1
  --train_sp_batch_size 1
  --gradient_accumulation_steps 1
  --num_height 480
  --num_width 832
  --num_frames 20
  --enable_gradient_checkpointing "full"
  --log_visualization
  --causal
  --i2v_rate 1.0
  --window_frames 20
  --seed 1000
  --lr_scheduler "constant"
  --lr_warmup_steps 10
  --lr_num_cycles 1
  --lr_power 1.0
  --min_lr_ratio 0.5
  --scale_lr False
  --training_state_checkpointing_steps 0
  --weight_only_checkpointing_steps 0
)

# ==================== Parallel Arguments ====================
parallel_args=(
  --num_gpus $TOTAL_GPUS
  --sp_size 2
  --tp_size 1
  --hsdp_replicate_dim 1
  --hsdp_shard_dim $TOTAL_GPUS
)

# ==================== Model Arguments ====================
model_args=(
  --cls_name "HunyuanTransformer3DARActionModel"
  --pretrained_model_name_or_path "$GENERATOR_MODEL_PATH"
  --real_score_model_path "$REAL_SCORE_MODEL_PATH"
  --fake_score_model_path "$FAKE_SCORE_MODEL_PATH"
  --ar_action_load_from_dir "$AR_ACTION_LOAD_FROM_DIR"
  --teacher_model_path "$TEACHER_MODEL_PATH"
)

if [ -n "$RESUME_CKPT" ]; then
  model_args+=(
    --resume_from_checkpoint "$RESUME_CKPT"
  )
fi

# ==================== Dataset Arguments ====================
dataset_args=(
  --dataloader_num_workers 4
)

# ==================== Validation Arguments ====================
validation_args=(
  --validation_steps 200
  --validation_sampling_steps "4"
  --validation_guidance_scale "6.0"
)

# ==================== Optimizer Arguments ====================
optimizer_args=(
  --learning_rate 1e-5
  --mixed_precision "bf16"
  --checkpointing_steps 100
  --weight_decay 0.01
  --max_grad_norm 1.0
  --betas "0.0,0.999"
  --gradient_accumulation_steps 1
)

# ==================== Miscellaneous Arguments ====================
miscellaneous_args=(
  --inference_mode False
  --checkpoints_total_limit 3
  --training_cfg_rate 0.0
  --dit_precision "bf16"
  --ema_start_step 100
  --ema_decay 0.99
  --use_ema True
)

# ==================== DMD Arguments ====================
dmd_args=(
  --dmd_denoising_steps '1000,750,500,250'
  --min_timestep_ratio 0.00
  --max_timestep_ratio 1.00
  --dfake_gen_update_ratio 5
  --cfg_scale 5.0  # alternative: 3.5
  --fake_score_learning_rate 8e-6
  --fake_score_betas '0.0,0.999'
  --warp_denoising_step
)

# ==================== Self-Forcing Arguments ====================
self_forcing_args=(
  --num_frame_per_block 4
  --independent_first_frame False
  --same_step_across_blocks True
  --last_step_only False
  --context_noise 0
  --enable_gradient_masking
  --gradient_mask_last_n_frames 20
  --flow_shift 5.0
  --solver cm
)

# ==================== Run Training ====================
LOG_DIR="logs/debug_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

echo "Redirecting output to $LOG_DIR/train.log"

export PYTHONPATH="$PROJECT_ROOT/HY15:$PROJECT_ROOT/shared:$PYTHONPATH"
torchrun \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    --nproc_per_node=$NUM_GPUS_PER_NODE \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    HY15/trainer/pipelines/ar_hunyuan_dmd_distill_entry.py \
    "${parallel_args[@]}" \
    "${model_args[@]}" \
    "${dataset_args[@]}" \
    "${training_args[@]}" \
    "${optimizer_args[@]}" \
    "${validation_args[@]}" \
    "${miscellaneous_args[@]}" \
    "${dmd_args[@]}" \
    "${self_forcing_args[@]}" \
    2>&1 | tee "$LOG_DIR/train.log"
