set -e
# Auto-find project root (look for requirements.txt in parent dirs)
PROJECT_ROOT="$(cd "$(dirname "$0")"; while [ "$PWD" != "/" ] && [ ! -f "requirements.txt" ]; do cd ..; done; pwd)"
cd "$PROJECT_ROOT"


export WANDB_BASE_URL="https://api.wandb.ai"
export WANDB_MODE=offline
export TOKENIZERS_PARALLELISM=false
export NCCL_DEBUG=WARN

# ===== Model paths =====
# HY1.5 base transformer (per ckpts_download.md §1.1)
TRANSFORMER_DIR="${TRANSFORMER_DIR:-./ckpts/HunyuanVideo-1.5/transformer/480p_i2v}"

# Stage 1 AR Diffusion TF ckpt (readme §4.1.2 Stage 1 product)
AR_ACTION_LOAD_FROM_DIR="${AR_ACTION_LOAD_FROM_DIR:-./ckpts/HY15/Action2V/ar_diffusion_tf/diffusion_pytorch_model.safetensors}"

# ===== Data paths =====
# SFT 编码后的 train_index.json（来自 §4.1.1 数据准备）
PREENCODED_DIR="${PREENCODED_DIR:-./dataset/HY15/Action2V/train_index.json}"
OUTPUT_DIR="${OUTPUT_DIR:-./outputs/ode_sampling_camera}"
# ODE latents 落点（与 SFT latents 平级，避免覆盖；下游训练会再跑 create_train_index.py）
ODE_OUTPUT_PATH="${ODE_OUTPUT_PATH:-./dataset/HY15/Action2V_ode/latents}"

# ===== Neg prompt =====
export NEG_PROMPT_PT="${NEG_PROMPT_PT:-./dataset/others/HY/Action2V/hunyuan_neg_prompt.pt}"
export NEG_BYT5_PT="${NEG_BYT5_PT:-./dataset/others/HY/Action2V/hunyuan_neg_byt5_prompt.pt}"

# ===== Multi-node configuration =====
NUM_GPUS_PER_NODE=8
NNODES=${WORLD_SIZE:-1}
NODE_RANK=${RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-"localhost"}
MASTER_PORT=${MASTER_PORT:-29611}

TOTAL_GPUS=$((NUM_GPUS_PER_NODE * NNODES))

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

echo "=== Multi-node config ==="
echo "  NNODES: $NNODES"
echo "  NODE_RANK: $NODE_RANK"
echo "  MASTER_ADDR: $MASTER_ADDR"
echo "  MASTER_PORT: $MASTER_PORT"
echo "  TOTAL_GPUS: $TOTAL_GPUS"
echo "  TRANSFORMER_DIR: $TRANSFORMER_DIR"
echo "  AR_CKPT: $AR_ACTION_LOAD_FROM_DIR"
echo "  ODE_OUTPUT: $ODE_OUTPUT_PATH"
echo "========================="

training_args=(
  --json_path "$PREENCODED_DIR"
  --causal
  --i2v_rate 0.0
  --train_time_shift 3.0
  --window_frames 20
  --output_dir "$OUTPUT_DIR"
  --train_batch_size 1
  --train_sp_batch_size 1
  --gradient_accumulation_steps 4
  --num_latent_t 16
  --num_height 480
  --num_width 832
  --num_frames 77
  --enable_gradient_checkpointing_type "full"
  --seed 3208
  --weighting_scheme "logit_normal"
  --logit_mean 0.0
  --logit_std 1.0
  --ode-shift 5.0
  --ode-output-path "$ODE_OUTPUT_PATH"
  --ode-sampling-steps 48
)

parallel_args=(
  --num_gpus $TOTAL_GPUS
  --sp_size 1
  --tp_size 1
  --hsdp_replicate_dim 1
  --hsdp_shard_dim $TOTAL_GPUS
)

model_args=(
  --cls_name "HunyuanTransformer3DARActionProPEModel"
  --load_from_dir "$TRANSFORMER_DIR"
  --model_path "$TRANSFORMER_DIR"
  --pretrained_model_name_or_path "$TRANSFORMER_DIR"
)

if [ -n "$AR_ACTION_LOAD_FROM_DIR" ]; then
  model_args+=(
    --ar_action_load_from_dir "$AR_ACTION_LOAD_FROM_DIR"
  )
fi

dataset_args=(
  --dataloader_num_workers 1
)

validation_args=(
  --validation_steps 9999
  --validation_sampling_steps "50"
  --validation_guidance_scale "6.0"
)

optimizer_args=(
  --learning_rate 1e-5
  --mixed_precision "bf16"
  --checkpointing_steps 9999
  --weight_decay 1e-4
  --max_grad_norm 1.0
)

miscellaneous_args=(
  --inference_mode False
  --checkpoints_total_limit 3
  --training_cfg_rate 0.0
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
    HY15/trainer/pipelines_camera/ar_ode_sampling_entry.py \
    "${parallel_args[@]}" \
    "${model_args[@]}" \
    "${dataset_args[@]}" \
    "${training_args[@]}" \
    "${optimizer_args[@]}" \
    "${validation_args[@]}" \
    "${miscellaneous_args[@]}"
