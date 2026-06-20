set -e
PROJECT_ROOT="$(cd "$(dirname "$0")"; while [ "$PWD" != "/" ] && [ ! -f "requirements.txt" ]; do cd ..; done; pwd)"
cd "$PROJECT_ROOT"

export WANDB_BASE_URL="https://api.wandb.ai"
# export WANDB_MODE=online
export WANDB_MODE=offline
export TOKENIZERS_PARALLELISM=false
export NCCL_DEBUG=WARN

# ===== Paths (relative to project root, see readme §2.1 / §4.1.1) =====
# HY1.5 base transformer (readme §2.1)
TRANSFORMER_DIR="${TRANSFORMER_DIR:-./ckpts/HunyuanVideo-1.5/transformer/480p_i2v}"

# Camera-annotated SFT latents index (readme §4.1.1 (2))
PREENCODED_DIR="${PREENCODED_DIR:-${CAMERA_JSON_PATH:-./dataset/HY15/Action2V/train_index.json}}"
# Training output（与 §4.1.2 Stage 1 (1) 预下载落点对齐 → ./ckpts/HY15/Action2V/bidirectional）
OUTPUT_DIR="${OUTPUT_DIR:-./ckpts/HY15/Action2V/bidirectional}"

# CFG negative prompt embeddings (readme §4.1.1 (2) → ./dataset/others/HY/Action2V/)
export NEG_PROMPT_PT="${NEG_PROMPT_PT:-./dataset/others/HY/Action2V/hunyuan_neg_prompt.pt}"
export NEG_BYT5_PT="${NEG_BYT5_PT:-./dataset/others/HY/Action2V/hunyuan_neg_byt5_prompt.pt}"

NUM_GPUS_PER_NODE=8
NNODES=${WORLD_SIZE:-1}
NODE_RANK=${RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-"localhost"}
MASTER_PORT=${MASTER_PORT:-29613}
TOTAL_GPUS=$((NUM_GPUS_PER_NODE * NNODES))
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

training_args=(
  --json_path "$PREENCODED_DIR"
  --use_discrete_action ${USE_DISCRETE_ACTION:-False}
  --i2v_rate 0.0
  --train_time_shift 3.0
  --window_frames 20
  --output_dir "$OUTPUT_DIR"
  --max_train_steps 20000
  --train_batch_size 1
  --train_sp_batch_size 1
  --gradient_accumulation_steps 1
  --enable_gradient_checkpointing_type "full"
  --seed 42
  --weighting_scheme "logit_normal"
  --logit_mean 0.0
  --logit_std 1.0
)

parallel_args=(
  --num_gpus $TOTAL_GPUS
  --sp_size 8
  --tp_size 1
  --hsdp_replicate_dim 1
  --hsdp_shard_dim $TOTAL_GPUS
)

model_args=(
  --cls_name "HunyuanTransformer3DARActionProPEModel"
  --pretrained_model_name_or_path "$TRANSFORMER_DIR"
)

if [ -n "$AR_ACTION_LOAD_FROM_DIR" ]; then
  model_args+=(--ar_action_load_from_dir "$AR_ACTION_LOAD_FROM_DIR")
fi

dataset_args=(
  --dataloader_num_workers 1
)

validation_args=(
  --validation_steps 1000
  --validation_sampling_steps "50"
  --validation_guidance_scale "6.0"
  --log_validation
)

optimizer_args=(
  --learning_rate 2e-5
  --mixed_precision "bf16"
  --checkpointing_steps 1000
  --weight_decay 1e-4
  --max_grad_norm 1.0
)

if [ -n "$RESUME_CKPT" ]; then
  optimizer_args+=(--resume_from_checkpoint "$RESUME_CKPT")
fi

miscellaneous_args=(
  --inference_mode False
  --checkpoints_total_limit 1
  --training_cfg_rate 0.1
  --multi_phased_distill_schedule "4000-1"
  --not_apply_cfg_solver
  --dit_precision "fp32"
  --num_euler_timesteps 50
  --ema_start_step 0
)

export PYTHONPATH="$PROJECT_ROOT/HY15:$PROJECT_ROOT/shared:$PYTHONPATH"
torchrun \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    --nproc_per_node=$NUM_GPUS_PER_NODE \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    HY15/trainer/pipelines_camera/bidir_camera_training_entry.py \
    "${parallel_args[@]}" \
    "${model_args[@]}" \
    "${dataset_args[@]}" \
    "${training_args[@]}" \
    "${optimizer_args[@]}" \
    "${validation_args[@]}" \
    "${miscellaneous_args[@]}"
