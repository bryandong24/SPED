set -e
PROJECT_ROOT="$(cd "$(dirname "$0")"; while [ "$PWD" != "/" ] && [ ! -f "requirements.txt" ]; do cd ..; done; pwd)"
cd "$PROJECT_ROOT"

export WANDB_BASE_URL="https://api.wandb.ai"
# export WANDB_MODE=online
export WANDB_MODE=offline
export TOKENIZERS_PARALLELISM=false
export NCCL_DEBUG=WARN

# ===== Paths (relative to project root, see readme §2.1 / §4.2.1 / §4.2.2 Stage 1 / Stage 2(b)) =====
# HY1.5 base transformer (readme §2.1)
TRANSFORMER_DIR="${TRANSFORMER_DIR:-./ckpts/HunyuanVideo-1.5/transformer/480p_i2v}"

# Student / teacher init = Stage 1 product (readme §4.2.2 Stage 2(b) (1) → 同 Stage 2(a))
AR_ACTION_LOAD_FROM_DIR="${AR_ACTION_LOAD_FROM_DIR:-./ckpts/HY15/TI2V/ar_diffusion_tf/diffusion_pytorch_model.safetensors}"

# CD 走 SFT latents（readme §4.2.2 Stage 2(b) (2)，复用 §4.2.1 编码产物）
PREENCODED_DIR="${PREENCODED_DIR:-./dataset/HY15/TI2V/train_index.json}"
# Training output（与 §4.2.2 Stage 3 (1) 预下载落点对齐 → ./ckpts/HY15/TI2V/causal_cd）
OUTPUT_DIR="${OUTPUT_DIR:-./ckpts/HY15/TI2V/causal_cd}"
RESUME_CKPT="${RESUME_CKPT:-}"

# CFG negative prompt embeddings (readme §4.2.1 (2) → ./dataset/others/HY/TI2V/)
export NEG_PROMPT_PT="${NEG_PROMPT_PT:-./dataset/others/HY/TI2V/hunyuan_neg_prompt.pt}"
export NEG_BYT5_PT="${NEG_BYT5_PT:-./dataset/others/HY/TI2V/hunyuan_neg_byt5_prompt.pt}"

NUM_GPUS_PER_NODE=8
NNODES=${WORLD_SIZE:-1}
NODE_RANK=${RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-"localhost"}
MASTER_PORT=${MASTER_PORT:-29612}
TOTAL_GPUS=$((NUM_GPUS_PER_NODE * NNODES))
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

echo "=== CD Training config ==="
echo "  NNODES: $NNODES  NODE_RANK: $NODE_RANK  TOTAL_GPUS: $TOTAL_GPUS"
echo "=========================="

training_args=(
  --json_path "$PREENCODED_DIR"
  --causal
  --i2v_rate 0.0
  --train_time_shift 3.0
  --window_frames 20
  --output_dir "$OUTPUT_DIR"
  --max_train_steps 200000
  --train_batch_size 1
  --train_sp_batch_size 1
  --gradient_accumulation_steps 1
  --num_latent_t 9
  --num_height 480
  --num_width 832
  --num_frames 77
  --enable_gradient_checkpointing_type "full"
  --seed 3208
  # CD timestep schedule: same as inference (N steps with shift)
  --cd_num_steps 50
  --ode_shift 5.0
  --cfg-scale 5.0  # alternative: 3.5
)

parallel_args=(
  --num_gpus $TOTAL_GPUS
  --sp_size 2
  --tp_size 1
  --hsdp_replicate_dim 1
  --hsdp_shard_dim $TOTAL_GPUS
)

model_args=(
  --cls_name "HunyuanTransformer3DARActionModel"
  --load_from_dir "$TRANSFORMER_DIR"
  --model_path "$TRANSFORMER_DIR"
  --pretrained_model_name_or_path "$TRANSFORMER_DIR"
  --ar_action_load_from_dir "$AR_ACTION_LOAD_FROM_DIR"
)

dataset_args=(
  --dataloader_num_workers 1
  --training_cfg_rate 0.0
  --i2v_rate 0.0
)

validation_args=(
  --log_validation
  --validation_steps 300
  --validation_sampling_steps "20" # TODO: 1. CD/SF 应该有自己的 validation; 2. frames=21时mask有bug: chunk_num=21//4=5只覆盖前20帧, 第21帧token只能attend to text, vision self-attn全0, 末尾denoise退化("最后一秒炸了"不是数据问题); 3. 根因: ar_causal_cd_pipeline.py:207加载validation_samples未做window_frames裁剪和//4*4对齐
  --validation_guidance_scale "6.0"
)

optimizer_args=(
  --learning_rate 1e-5
  --mixed_precision "bf16"
  --checkpointing_steps 500
  --weight_decay 1e-4
  --max_grad_norm 1.0
)

if [ -n "$RESUME_CKPT" ]; then
  optimizer_args+=(--resume_from_checkpoint "$RESUME_CKPT")
fi

miscellaneous_args=(
  --inference_mode False
  --checkpoints_total_limit 3
  --dit_precision "fp32"
)

export PYTHONPATH="$PROJECT_ROOT/HY15:$PROJECT_ROOT/shared:$PYTHONPATH"
torchrun \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    --nproc_per_node=$NUM_GPUS_PER_NODE \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    HY15/trainer/pipelines/ar_causal_cd_entry.py \
    "${parallel_args[@]}" \
    "${model_args[@]}" \
    "${dataset_args[@]}" \
    "${training_args[@]}" \
    "${optimizer_args[@]}" \
    "${validation_args[@]}" \
    "${miscellaneous_args[@]}"
