# SPDX-License-Identifier: Apache-2.0
"""
Self-Forcing DMD distillation pipeline for HY-WorldPlay AR models.

This pipeline implements the self-forcing DMD (Diffusion Model Distillation) approach
adapted from FastVideo for HY-WorldPlay. Key features:

1. Generator is an AR model with KV cache support
2. Real score and fake score teachers are bidirectional models
3. Self-forcing training with alternating updates
4. No real video data supervision - only conditions (text, image, camera)

Implementation references:
- Input processing: trainer/training/ar_hunyuan_training_pipeline.py
- AR inference: hyvideo/pipelines/worldplay_video_pipeline.py
- Self-forcing logic: fastvideo/training/self_forcing_distillation_pipeline.py
"""
import math
import os
from typing import Any, Optional

import torch
import torch.distributed as dist
import torch.nn.functional as F
from einops import rearrange
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm.auto import tqdm
import dataclasses

from trainer.dataset.ti2v_dataset import build_ti2v_dataloader
from trainer.distributed import (
    cleanup_dist_env_and_memory,
    get_local_torch_device,
    get_sp_group,
    get_sp_parallel_rank,
    get_sp_world_size,
    get_world_group,
)
from trainer.forward_context import set_forward_context
from trainer.logger import init_logger
from trainer.pipelines import ComposedPipelineBase, TrainingBatch
from trainer.pipelines.ar_hunyuan_training_pipeline import TrainingPipeline
from trainer.training.dmd_utils_hy import (
    clone_training_batch,
    get_sp_frame_indices,
    parse_denoising_steps,
    select_memory_frames,
)
from hyvideo.schedulers.scheduling_flow_match_discrete import (
    FlowMatchDiscreteScheduler,
)
from trainer.training.training_utils import (
    clip_grad_norm_while_handling_failing_dtensor_cases,
    compute_density_for_timestep_sampling,
    get_scheduler,
    get_sigmas,
    load_checkpoint,
    save_checkpoint,
)
from trainer.training.muon import get_muon_optimizer
from trainer.training.ema import EMA
from trainer.utils import set_random_seed
from trainer.training.activation_checkpoint import (
    apply_activation_checkpointing)
import wandb

from algorithms.dmd import apply_cfg, compute_kl_gradient, dmd_generator_loss, dmd_critic_loss
from algorithms.self_forcing import create_gradient_mask, apply_gradient_mask

logger = init_logger(__name__)


class ARHunyuanDMDDistillationPipeline(TrainingPipeline):
    """
    Self-Forcing DMD distillation pipeline for HY-WorldPlay AR models.

    This pipeline implements the self-forcing DMD approach where:
    1. Generator (AR model) and critic (fake_score) are trained in alternating steps
    2. Generator loss uses DMD-style loss with the critic as fake score
    3. Critic loss trains the fake score model to distinguish real vs fake
    4. No real video data is used - only conditions (text, image, camera poses)
    """

    def __init__(
        self,
        model_path: str,
        trainer_args: Any,
        required_config_modules: list[str] | None = None,
        loaded_modules: dict[str, torch.nn.Module] | None = None,
    ) -> None:
        super().__init__(model_path, trainer_args, required_config_modules, loaded_modules)

        # DMD-specific attributes
        self.denoising_step_list: list[int] = []
        self._kv_cache: list[dict[str, Any]] | None = None
        self.dfake_gen_update_ratio: int = 5
        self.num_frame_per_block: int = 3
        self.independent_first_frame: bool = False
        self.same_step_across_blocks: bool = False
        self.last_step_only: bool = False
        self.context_noise: int = 0
        self.enable_gradient_masking: bool = False
        self.gradient_mask_last_n_frames: int = 21

        # Flow shift for HY-WorldPlay
        self.flow_shift: float = 5.0

        # Scheduler (unified: used for both training noise and inference denoising)
        self.infer_scheduler: FlowMatchDiscreteScheduler | None = None

        # Teacher models
        self.real_score_transformer: torch.nn.Module | None = None
        self.fake_score_transformer: torch.nn.Module | None = None
        self.fake_score_optimizer: torch.optim.Optimizer | None = None
        self.fake_score_lr_scheduler: Any | None = None

        # Training state
        self.current_trainstep: int = 0

        # EMA for generator (will be initialized at ema_start_step)
        self.generator_ema: EMA | None = None
        

    def _load_teacher_models(self) -> None:
        """Load real_score_transformer and fake_score_transformer."""

        # Load real score teacher (bidirectional model)
        teacher_model_path = self.training_args.teacher_model_path
        real_score_path = self.training_args.real_score_model_path
        if real_score_path:
            logger.info("Loading real score teacher from: %s", real_score_path)
            self.real_score_transformer = self._load_teacher_model(real_score_path,teacher_model_path)
            self.real_score_transformer.eval()
            self.real_score_transformer.cpu()
            logger.info("Real score teacher offloaded to CPU (saves ~26GB GPU memory)")
        else:
            logger.warning("Real score model path not provided, skipping real score teacher")

        # Load fake score teacher (bidirectional model, critic)
        fake_score_path = self.training_args.fake_score_model_path
        if fake_score_path:
            logger.info("Loading fake score teacher from: %s", fake_score_path)
            self.fake_score_transformer = self._load_teacher_model(fake_score_path,teacher_model_path)
            self.fake_score_transformer.train()
        else:
            raise ValueError("Fake score model path must be provided for DMD distillation")

    def _load_teacher_model(self, model_path: str, teacher_model_path: str | None   ) -> torch.nn.Module:
        """Load a teacher model (bidirectional) from checkpoint using the same loader as main transformer."""
        import glob
        import os
        from trainer.models.loader.fsdp_load import maybe_load_fsdp_model
        from trainer.models.registry import ModelRegistry
        from trainer.utils import PRECISION_TO_TYPE

        # Import the transformer class
        from trainer.models.hyvideo.transformer.ar_action_hunyuanvideo_1_5_transformer import (
            ARHunyuanVideo_1_5_DiffusionTransformer,
        )

        # Get the model class from registry (same as main transformer)
        cls_name = self.trainer_args.cls_name
        model_cls, _ = ModelRegistry.resolve_model_cls(cls_name)

        # Find safetensors files
        safetensors_list = glob.glob(os.path.join(str(model_path), "*.safetensors"))
        if not safetensors_list:
            raise ValueError(f"No safetensors files found in {model_path}")

        logger.info("Loading teacher model from %s safetensors files in %s",
                    len(safetensors_list), model_path)

        # Get dtype from trainer_args
        default_dtype = PRECISION_TO_TYPE[self.trainer_args.pipeline_config.dit_precision]

        # Load using the same FSDP loader as main transformer
        model = maybe_load_fsdp_model(
            load_from_dir=model_path,
            ar_action_load_from_dir=teacher_model_path,
            cls_name=cls_name,
            model_cls=model_cls,
            init_params={},
            weight_dir_list=safetensors_list,
            device=self.device,
            hsdp_replicate_dim=self.trainer_args.hsdp_replicate_dim,
            hsdp_shard_dim=self.trainer_args.hsdp_shard_dim,
            cpu_offload=self.trainer_args.dit_cpu_offload,
            pin_cpu_memory=getattr(self.trainer_args, 'pin_cpu_memory', False),
            fsdp_inference=self.trainer_args.use_fsdp_inference,
            param_dtype=default_dtype,
            reduce_dtype=torch.float32,
            output_dtype=None,
            training_mode=True,  # Teacher models may need to be trainable for fake_score
        )

        total_params = sum(p.numel() for p in model.parameters())
        logger.info("Loaded teacher model with %.2fB parameters", total_params / 1e9)

        # Set to correct dtype if needed
        dtypes = set(param.dtype for param in model.parameters())
        if len(dtypes) > 1:
            model = model.to(default_dtype)

        logger.info("Successfully loaded teacher model from: %s", model_path)
        return model

    def set_trainable_all(self):
        # Only train DiT
        self.transformer.requires_grad_(True)
        self.real_score_transformer.requires_grad_(False)
        self.fake_score_transformer.requires_grad_(True)

    def initialize_training_pipeline(self, training_args: Any) -> None:
        """Initialize the self-forcing DMD training pipeline."""
        # Call parent initialization first to set self.device and self.transformer
        # super().initialize_training_pipeline(training_args)

        self.device = get_local_torch_device()
        self.training_args = training_args
        world_group = get_world_group()
        self.world_size = world_group.world_size
        self.global_rank = world_group.rank
        self.sp_group = get_sp_group()
        self.rank_in_sp_group = self.sp_group.rank_in_group
        self.sp_world_size = self.sp_group.world_size
        self.local_rank = world_group.local_rank

        # Validate Sequence Parallel configuration
        if self.sp_world_size > 1:
            frames_per_gpu = training_args.num_latent_t // self.sp_world_size
            if frames_per_gpu == 0:
                raise ValueError(
                    f"Sequence Parallel mismatch: num_latent_t={training_args.num_latent_t} "
                    f"with sp_size={self.sp_world_size} gives {frames_per_gpu} frames per GPU. "
                    f"Set num_latent_t >= sp_size (recommended: {training_args.window_frames})"
                )
            logger.info(f"SP validation passed: {frames_per_gpu} frames per GPU")

        self.transformer = self.get_module("transformer")
        self.seed = training_args.seed
        self.set_schemas()

        # add the causal option
        self.causal = training_args.causal
         
        # Set random seeds for deterministic training
        assert self.seed is not None, "seed must be set"
        set_random_seed(self.seed)
        self.transformer.train()
        self.set_trainable()
        if training_args.enable_gradient_checkpointing_type is not None:
            self.transformer = apply_activation_checkpointing(
                self.transformer,
                checkpointing_type=training_args.
                enable_gradient_checkpointing_type)
        params_to_optimize = self.transformer.parameters()
        params_to_optimize = list(
            filter(lambda p: p.requires_grad, params_to_optimize))
        betas = [float(x) for x in training_args.betas.split(",")]
        self.optimizer = get_muon_optimizer(
            model=self.transformer,
            lr=training_args.learning_rate,                      # Learning rate
            weight_decay=training_args.weight_decay,  # Weight decay
            adamw_betas=tuple(betas),   # AdamW betas for 1D parameters
            adamw_eps=1e-8,        # AdamW epsilon
        )


        self.init_steps = 0
        logger.info("optimizer: %s", self.optimizer)

        self.lr_scheduler = get_scheduler(
            training_args.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=training_args.lr_warmup_steps,
            num_training_steps=training_args.max_train_steps,
            num_cycles=training_args.lr_num_cycles,
            power=training_args.lr_power,
            min_lr_ratio=training_args.min_lr_ratio,
            last_epoch=self.init_steps - 1,
        )
        self._load_teacher_models()
        self.set_trainable_all()
        if training_args.enable_gradient_checkpointing_type is not None:
            self.fake_score_transformer = apply_activation_checkpointing(
                self.fake_score_transformer,
                checkpointing_type=training_args.
                enable_gradient_checkpointing_type)
        # DMD-specific parameters
        self.denoising_step_list = parse_denoising_steps(training_args.dmd_denoising_steps)
        self.num_inference_steps = len(self.denoising_step_list)  # timesteps now from scheduler, not denoising_step_list
        self.dfake_gen_update_ratio = training_args.dfake_gen_update_ratio
        self.num_frame_per_block = training_args.num_frame_per_block
        self.independent_first_frame = training_args.independent_first_frame
        self.same_step_across_blocks = training_args.same_step_across_blocks
        self.last_step_only = training_args.last_step_only
        self.context_noise = training_args.context_noise
        self.enable_gradient_masking = training_args.enable_gradient_masking
        self.gradient_mask_last_n_frames = training_args.gradient_mask_last_n_frames
        self.flow_shift = training_args.flow_shift

        # Timestep clamping parameters
        num_train_timesteps = 1000
        self.num_train_timesteps = num_train_timesteps
        self.min_timestep = int(getattr(training_args, 'min_timestep_ratio', 0.02) * num_train_timesteps)
        self.max_timestep = int(getattr(training_args, 'max_timestep_ratio', 0.98) * num_train_timesteps)

        # Real score guidance scale for CFG (reference: FastVideo Line 244)
        self.cfg_scale = getattr(training_args, 'cfg_scale', 5.0)
        assert training_args.training_cfg_rate == 0.0, \
            "Distillation student must see real prompts; set --training_cfg_rate 0.0"

        # Load neg prompt embeddings for CFG uncond (aligned with CD pipeline)
        neg_prompt_path = os.environ.get("NEG_PROMPT_PT", "")
        neg_byt5_path = os.environ.get("NEG_BYT5_PT", "")
        if neg_prompt_path and neg_byt5_path:
            self.neg_prompt_pt = torch.load(neg_prompt_path, map_location=self.device, weights_only=True)
            self.neg_byt5_pt = torch.load(neg_byt5_path, map_location=self.device, weights_only=True)
            logger.info("Loaded neg prompt embeddings for CFG uncond")
        else:
            self.neg_prompt_pt = None
            self.neg_byt5_pt = None
            logger.warning("NEG_PROMPT_PT / NEG_BYT5_PT not set; falling back to zeros for CFG uncond")

        # EMA will be initialized at ema_start_step in the training loop
        ema_decay = getattr(training_args, 'ema_decay', 0.0)
        if ema_decay > 0.0:
            self.generator_ema = None  # Will be created at ema_start_step
            logger.info("EMA will be initialized at step %s with decay=%s",
                        training_args.ema_start_step, ema_decay)
        else:
            self.generator_ema = None
            logger.info("EMA disabled (ema_decay <= 0.0)")

        # Initialize scheduler (unified for training noise and inference denoising)
        self.infer_scheduler = FlowMatchDiscreteScheduler(
            shift=self.flow_shift,
            reverse=True,
            solver=training_args.solver,
        )

        # Initialize fake score optimizer
        fake_score_lr = training_args.fake_score_learning_rate
        if fake_score_lr == 0.0:
            fake_score_lr = training_args.learning_rate

        betas = [float(x) for x in training_args.fake_score_betas.split(",")]
        self.fake_score_optimizer = get_muon_optimizer(
            model=self.fake_score_transformer,
            lr=fake_score_lr,
            weight_decay=training_args.weight_decay,
            adamw_betas=tuple(betas),
            adamw_eps=1e-8,
        )

        # Initialize fake score LR scheduler
        fake_score_lr_scheduler = training_args.fake_score_lr_scheduler
        lr_scheduler = training_args.lr_scheduler
        if fake_score_lr_scheduler == "constant":
            self.fake_score_lr_scheduler = get_scheduler(
                "constant",
                optimizer=self.fake_score_optimizer,
                num_warmup_steps=0,
                num_training_steps=training_args.max_train_steps,
            )
        else:
            self.fake_score_lr_scheduler = get_scheduler(
                lr_scheduler if lr_scheduler else "constant",
                optimizer=self.fake_score_optimizer,
                num_warmup_steps=training_args.lr_warmup_steps,
                num_training_steps=training_args.max_train_steps,
                num_cycles=training_args.lr_num_cycles,
                power=training_args.lr_power,
                min_lr_ratio=training_args.min_lr_ratio,
                last_epoch=-1,
            )

        self.train_dataset, self.train_dataloader = build_ti2v_dataloader(
            json_path=training_args.json_path,
            causal=training_args.causal,
            window_frames=training_args.window_frames,
            batch_size=training_args.train_batch_size,
            num_data_workers=training_args.dataloader_num_workers,
            drop_last=True,
            drop_first_row=False,
            seed=self.seed,
            training_cfg_rate=training_args.training_cfg_rate,
            i2v_rate=training_args.i2v_rate,
        )

        # Ensure shared_state["max_frames"] is correctly initialized
        if hasattr(self.train_dataset, 'shared_state'):
            current_max_frames = self.train_dataset.shared_state.get("max_frames")
            expected_max_frames = training_args.window_frames
            if current_max_frames != expected_max_frames:
                logger.warning("shared_state['max_frames'] is %s, expected %s. Resetting...",
                               current_max_frames, expected_max_frames)
                self.train_dataset.shared_state["max_frames"] = expected_max_frames

        self.num_update_steps_per_epoch = math.ceil(
            len(self.train_dataloader) /
            training_args.gradient_accumulation_steps * training_args.sp_size /
            training_args.train_sp_batch_size)
        self.num_train_epochs = math.ceil(training_args.max_train_steps /
                                          self.num_update_steps_per_epoch)

        # TODO(will): is there a cleaner way to track epochs?
        self.current_epoch = 0

        if self.global_rank == 0:
            project = training_args.tracker_project_name or "trainer"
            wandb_config = dataclasses.asdict(training_args)
            wandb.login(key=training_args.wandb_key)
            wandb.init(
                config=wandb_config,
                name=training_args.wandb_run_name,
                entity=training_args.wandb_entity,
                project=project,
            )
       
        
        logger.info("DMD denoising steps: %s", self.denoising_step_list)
        logger.info("DMD generator update ratio: %s", self.dfake_gen_update_ratio)
        logger.info("DMD num frames per block: %s", self.num_frame_per_block)

    # ==================== Helper Methods ====================

    def timestep_transform(self, t: torch.Tensor, shift: float = 1.0, num_timesteps: float = 1000.0) -> torch.Tensor:
        """Apply timestep transformation used in HY-WorldPlay."""
        t = t / num_timesteps
        t = shift * t / (1 + (shift - 1) * t)
        t = t * num_timesteps
        return t

    def merge_tensor_by_mask(self, tensor_1, tensor_2, mask, dim):
        """Merge two tensors using a mask."""
        assert tensor_1.shape == tensor_2.shape, f"Shape mismatch: {tensor_1.shape} vs {tensor_2.shape}"

        masked_indices = torch.nonzero(mask).squeeze(1)

        # Validate indices are within bounds
        if len(masked_indices) > 0:
            max_idx = masked_indices.max().item()
            valid_range = tensor_1.shape[dim] - 1
            assert max_idx <= valid_range, f"Index out of bounds: {max_idx} > {valid_range}"

        tmp = tensor_1.clone()

        if dim == 0:
            tmp[masked_indices] = tensor_2[masked_indices]
        elif dim == 1:
            tmp[:, masked_indices] = tensor_2[:, masked_indices]
        elif dim == 2:
            tmp[:, :, masked_indices] = tensor_2[:, :, masked_indices]

        return tmp

    def _prepare_cond_latents(self, task_type: str, cond_latents: torch.Tensor,
                               latents: torch.Tensor, multitask_mask: torch.Tensor) -> torch.Tensor:
        """
        Prepare conditional latents and mask for multitask training.

        Args:
            task_type: Type of task ("i2v" or "t2v").
            cond_latents: Conditional latents tensor.
            latents: Main latents tensor.
            multitask_mask: Multitask mask tensor.

        Returns:
            Concatenated conditional latents.
        """
        if cond_latents is not None and task_type == 'i2v':
            latents_concat = cond_latents.repeat(1, 1, latents.shape[2], 1, 1)
            latents_concat[:, :, 1:, :, :] = 0.0
        else:
            latents_concat = torch.zeros(
                latents.shape[0], latents.shape[1], latents.shape[2],
                latents.shape[3], latents.shape[4]
            ).to(latents.device)

        mask_zeros = torch.zeros(
            latents.shape[0], 1, latents.shape[2],
            latents.shape[3], latents.shape[4]
        )
        mask_ones = torch.ones(
            latents.shape[0], 1, latents.shape[2],
            latents.shape[3], latents.shape[4]
        )

        mask_concat = self.merge_tensor_by_mask(
            mask_zeros.cpu(), mask_ones.cpu(), mask=multitask_mask.cpu(), dim=2
        ).to(device=latents.device)

        cond_latents_result = torch.concat([latents_concat, mask_concat], dim=1)
        return cond_latents_result.to(latents.dtype)

    def get_task_mask(self, task_type: str, latent_target_length: int) -> torch.Tensor:
        """Get task mask for multitask training."""
        if latent_target_length <= 0:
            import traceback
            tb = ''.join(traceback.format_stack())
            print(f"FATAL: latent_target_length={latent_target_length} at step {self.current_trainstep}", flush=True)
            print(f"Caller stack:\n{tb}", flush=True)
            raise ValueError(f"latent_target_length must be > 0, got {latent_target_length}")

        if task_type == "t2v":
            mask = torch.zeros(latent_target_length, device=self.device)
        elif task_type == "i2v":
            mask = torch.zeros(latent_target_length, device=self.device)
            mask[0] = 1.0
        else:
            raise ValueError(f"{task_type} is not supported!")

        return mask

    # ==================== KV Cache Methods ====================

    def init_kv_cache(self):
        """Initialize KV cache — all None, populated by forward_txt and context rerun."""
        self._kv_cache = []
        transformer_num_layers = len(self.transformer.double_blocks)
        for _ in range(transformer_num_layers):
            self._kv_cache.append(
                {"k_vision": None, "v_vision": None, "k_txt": None, "v_txt": None}
            )


    # ==================== Input Processing Methods ====================

    def _get_next_batch(self, training_batch: TrainingBatch) -> TrainingBatch:
        """
        Load conditions and generate random latents (DMD core: no real video data).

        This method is adapted from ar_hunyuan_mem_training_pipeline.py but generates
        random latents instead of loading from dataset, as DMD doesn't need real video.
        """
        batch = next(self.train_loader_iter, None)
        if batch is None:
            self.current_epoch += 1
            logger.info("Starting epoch %s", self.current_epoch)
            self.train_loader_iter = iter(self.train_dataloader)
            batch = next(self.train_loader_iter)

        prompt_embed = batch["prompt_embed"]

        training_batch.prompt_embed = prompt_embed.to(self.device, dtype=torch.bfloat16)
        training_batch.image_cond = batch['image_cond'].to(self.device, dtype=torch.bfloat16)
        training_batch.vision_states = batch['vision_states'].to(self.device, dtype=torch.bfloat16)
        training_batch.prompt_mask = batch['prompt_mask'].to(self.device, dtype=torch.bfloat16)
        training_batch.byt5_text_states = batch['byt5_text_states'].to(self.device, dtype=torch.bfloat16)
        training_batch.byt5_text_mask = batch['byt5_text_mask'].to(self.device, dtype=torch.bfloat16)

        # DMD Core: Generate random latents instead of using real video
        batch_size = prompt_embed.shape[0]
        latents = torch.randn(
            batch_size, 32, self.training_args.num_latent_t,
            self.training_args.num_height // 16,
            self.training_args.num_width // 16,
            generator=self.noise_gen_cuda,
            device=self.device,
            dtype=torch.bfloat16,
        )
        training_batch.latents = latents

        return training_batch

    def _prepare_ar_dit_inputs(self, training_batch: TrainingBatch) -> TrainingBatch:
        """
        Prepare AR DIT inputs with timestep transformation.

        Adapted from ar_hunyuan_mem_training_pipeline.py.
        """
        latents = training_batch.latents
        batch_size = latents.shape[0]
        latent_t = latents.shape[2]

        # Generate noise
        noise = torch.randn(
            latents.shape,
            generator=self.noise_gen_cuda,
            device=latents.device,
            dtype=latents.dtype,
        )

        # Sample one timestep per batch item, shared across all frames
        u = torch.rand(batch_size, device=latents.device, generator=self.noise_gen_cuda)
        indices = (u * self.num_train_timesteps).long()
        indices = (self.num_train_timesteps - self.timestep_transform(
            indices.float(), self.flow_shift)).long()
        indices = indices.clamp(0, self.num_train_timesteps - 1)

        # Get sigmas from infer_scheduler (unified scheduler)
        sigmas = self.infer_scheduler.get_sigma(indices)
        sigmas = sigmas.reshape(batch_size, 1, 1, 1, 1)  # [B, 1, 1, 1, 1] broadcast over T,H,W

        # Expand timestep to all frames: [B] -> [B, T]
        timesteps = indices.float().unsqueeze(1).expand(batch_size, latent_t).to(device=self.device, dtype=torch.bfloat16)

        # SP broadcast
        if self.sp_world_size > 1:
            self.sp_group.broadcast(timesteps, src=0)

        # Add noise
        noisy_model_input = (1.0 - sigmas) * training_batch.latents + sigmas * noise

        training_batch.noisy_model_input = noisy_model_input
        training_batch.timesteps = timesteps  # Keep as [B, T]
        training_batch.sigmas = sigmas
        training_batch.noise = noise
        training_batch.raw_latent_shape = training_batch.latents.shape

        return training_batch

    def _build_input_kwargs(self, training_batch: TrainingBatch) -> TrainingBatch:
        """
        Build model input kwargs following HY-WorldPlay's format.

        Adapted from ar_hunyuan_mem_training_pipeline.py.
        """
        # Defensive check at entry
        if training_batch.noisy_model_input.shape[2] == 0:
            print(f"[Rank {self.global_rank}] FATAL: noisy_model_input.shape[2]=0 at _build_input_kwargs entry!", flush=True)
            print(f"  noisy_model_input.shape={training_batch.noisy_model_input.shape}", flush=True)
            print(f"  raw_latent_shape={training_batch.raw_latent_shape}", flush=True)
            raise RuntimeError(f"noisy_model_input has 0 frames at _build_input_kwargs entry")

        extra_kwargs = {
            "byt5_text_states": training_batch.byt5_text_states,
            "byt5_text_mask": training_batch.byt5_text_mask,
        }

        multitask_mask = self.get_task_mask("i2v", training_batch.noisy_model_input.shape[2])

        cond_latents = self._prepare_cond_latents(
            "i2v", training_batch.image_cond, training_batch.noisy_model_input, multitask_mask
        )

        latents_concat = torch.concat([training_batch.noisy_model_input, cond_latents], dim=1).to(torch.bfloat16)

        training_batch.input_kwargs = {
            "hidden_states": latents_concat,
            "timestep": training_batch.timesteps.flatten().to(self.device, dtype=torch.bfloat16),
            "timestep_txt": torch.tensor(0).unsqueeze(0).to(self.device, dtype=torch.bfloat16),
            "text_states": training_batch.prompt_embed,
            "text_states_2": None,
            "encoder_attention_mask": training_batch.prompt_mask,
            "timestep_r": None,
            "vision_states": training_batch.vision_states,
            "mask_type": "i2v",
            "guidance": None,
            "extra_kwargs": extra_kwargs,
            "return_dict": False,
        }

        return training_batch

    # ==================== Self-Forcing Generation Methods ====================

    def _cache_text_embeddings(self, training_batch: TrainingBatch):
        """Cache text embeddings in KV cache via forward_txt."""
        extra_kwargs = {
            "byt5_text_states": training_batch.byt5_text_states,
            "byt5_text_mask": training_batch.byt5_text_mask,
        }
        timestep_txt = torch.tensor([0]).to(self.device, dtype=torch.bfloat16)

        with torch.no_grad():
            self._kv_cache = self.transformer(
                bi_inference=False,
                ar_txt_inference=True,
                ar_vision_inference=False,
                timestep_txt=timestep_txt,
                text_states=training_batch.prompt_embed,
                encoder_attention_mask=training_batch.prompt_mask,
                vision_states=training_batch.vision_states,
                mask_type="i2v",
                extra_kwargs=extra_kwargs,
                kv_cache=self._kv_cache,
                cache_txt=True,
            )

    def _generator_multi_step_simulation_forward(
        self, training_batch: TrainingBatch
    ) -> torch.Tensor:
        """
        Multi-step simulation forward for self-forcing DMD.

        Core AR loop: for each chunk, denoise via forward_vision(cache_vision=False),
        then context-rerun via forward_vision(cache_vision=True) to update KV cache.
        Gradient flows only through the last denoising step of each chunk.
        """
        latents_shape = training_batch.raw_latent_shape  # [B, C, T, H, W]
        batch_size, _, num_frames, _, _ = latents_shape
        num_blocks = num_frames // self.num_frame_per_block

        # 1. Initial noise — must be identical across SP ranks so that
        #    sequence-parallel token splits correspond to the same video.
        noise = torch.randn(
            batch_size, 32, num_frames,
            latents_shape[3], latents_shape[4],
            device=self.device, dtype=torch.bfloat16,
        )
        if self.sp_world_size > 1:
            self.sp_group.broadcast(noise, src=0)
        latents = noise.clone()

        # 2. Initialize KV cache (all None)
        self.init_kv_cache()

        # 3. Cache text KV
        self._cache_text_embeddings(training_batch)

        # 5. Prepare condition latents
        multitask_mask = self.get_task_mask("i2v", num_frames)
        cond_latents = self._prepare_cond_latents(
            "i2v", training_batch.image_cond, noise, multitask_mask
        ).to(torch.bfloat16)

        solver = self.infer_scheduler.config.solver

        # For CM solver, video_out stores x̂_0 (with grad) separately from working latents
        video_out = latents.clone() if solver == "cm" else None

        # For CM: sample k_i once for all blocks (shared grad step)
        if solver == "cm":
            self.infer_scheduler.set_timesteps(self.num_inference_steps, device=self.device)
            _timesteps_for_ki = self.infer_scheduler.timesteps
            k_i = torch.randint(0, len(_timesteps_for_ki), (1,),
                                generator=self.noise_random_generator).to(self.device)
            if self.sp_world_size > 1:
                self.sp_group.broadcast(k_i, src=0)
            k_i = k_i.item()
            self._last_ki = k_i  # store for _generator_forward

        # 6. Per-chunk AR loop
        for block_idx in range(num_blocks):
            start_frame = block_idx * self.num_frame_per_block
            end_frame = start_frame + self.num_frame_per_block

            cond_block = cond_latents[:, :, start_frame:end_frame]

            # Per-chunk set_timesteps (aligned with inference _ar_rollout_inner)
            self.infer_scheduler.set_timesteps(self.num_inference_steps, device=self.device)
            timesteps = self.infer_scheduler.timesteps

            if solver == "euler":
                # ---- Euler: original logic, grad only on last step ----
                for step_idx, t in enumerate(timesteps):
                    is_last_step = (step_idx == len(timesteps) - 1)

                    timestep_block = torch.full(
                        (self.num_frame_per_block,), t,
                        device=self.device, dtype=torch.bfloat16,
                    )

                    latent_model_input = latents[:, :, start_frame:end_frame]
                    latents_concat = torch.concat([latent_model_input, cond_block], dim=1)

                    forward_kwargs = dict(
                        bi_inference=False,
                        ar_txt_inference=False,
                        ar_vision_inference=True,
                        hidden_states=latents_concat,
                        timestep=timestep_block,
                        timestep_r=None,
                        mask_type="i2v",
                        return_dict=False,
                        kv_cache=self._kv_cache,
                        cache_vision=False,
                        rope_temporal_size=end_frame,
                        start_rope_start_idx=start_frame,
                    )

                    if is_last_step:
                        pred_noise = self.transformer(**forward_kwargs)[0]
                    else:
                        with torch.no_grad():
                            pred_noise = self.transformer(**forward_kwargs)[0]

                    latent_model_input = self.infer_scheduler.step(
                        pred_noise, t, latent_model_input, return_dict=False
                    )[0]
                    latents[:, :, start_frame:end_frame] = latent_model_input

            elif solver == "cm":
                # ---- CM: use shared k_i (sampled once before block loop) ----

                for step_idx, t in enumerate(timesteps):
                    timestep_block = torch.full(
                        (self.num_frame_per_block,), t,
                        device=self.device, dtype=torch.bfloat16,
                    )

                    latent_model_input = latents[:, :, start_frame:end_frame]
                    latents_concat = torch.concat([latent_model_input, cond_block], dim=1)

                    forward_kwargs = dict(
                        bi_inference=False,
                        ar_txt_inference=False,
                        ar_vision_inference=True,
                        hidden_states=latents_concat,
                        timestep=timestep_block,
                        timestep_r=None,
                        mask_type="i2v",
                        return_dict=False,
                        kv_cache=self._kv_cache,
                        cache_vision=False,
                        rope_temporal_size=end_frame,
                        start_rope_start_idx=start_frame,
                    )

                    if step_idx == k_i:
                        # Selected step: WITH grad, predict x̂_0 directly
                        pred_noise = self.transformer(**forward_kwargs)[0]
                        sigma_t = self.infer_scheduler.sigmas[self.infer_scheduler.step_index
                                                              if self.infer_scheduler.step_index is not None
                                                              else self.infer_scheduler.index_for_timestep(t)]
                        x_0_hat = latent_model_input.float() - sigma_t * pred_noise.float()
                        video_out[:, :, start_frame:end_frame] = x_0_hat.to(latents.dtype)
                        self._last_gen_sigma = float(sigma_t)

                        # Fork: full CM step (with re-noising), detach for KV cache path
                        latent_model_input = self.infer_scheduler.step(
                            pred_noise, t, latent_model_input, return_dict=False
                        )[0].detach()
                        if self.sp_world_size > 1:
                            self.sp_group.broadcast(latent_model_input, src=0)
                    else:
                        # Non-selected steps: no_grad, normal CM step
                        with torch.no_grad():
                            pred_noise = self.transformer(**forward_kwargs)[0]
                        latent_model_input = self.infer_scheduler.step(
                            pred_noise, t, latent_model_input, return_dict=False
                        )[0]
                        if self.sp_world_size > 1:
                            self.sp_group.broadcast(latent_model_input, src=0)

                    # latents is the working tensor for denoising (always detached for CM)
                    latents[:, :, start_frame:end_frame] = latent_model_input

            # 6.2 Context rerun (update KV cache with denoised chunk)
            denoised_chunk = latents[:, :, start_frame:end_frame].detach()
            denoised_input = torch.concat([denoised_chunk, cond_block], dim=1)
            context_timestep = torch.full(
                (self.num_frame_per_block,),
                self.context_noise,
                device=self.device, dtype=torch.bfloat16,
            )

            with torch.no_grad():
                new_kv = self.transformer(
                    bi_inference=False,
                    ar_txt_inference=False,
                    ar_vision_inference=True,
                    hidden_states=denoised_input,
                    timestep=context_timestep,
                    timestep_r=None,
                    mask_type="i2v",
                    return_dict=False,
                    kv_cache=self._kv_cache,
                    cache_vision=True,
                    rope_temporal_size=end_frame,
                    start_rope_start_idx=start_frame,
                )

            # Append vision KV
            for i in range(len(self._kv_cache)):
                if self._kv_cache[i]["k_vision"] is None:
                    self._kv_cache[i]["k_vision"] = new_kv[i]["k_vision"]
                    self._kv_cache[i]["v_vision"] = new_kv[i]["v_vision"]
                else:
                    self._kv_cache[i]["k_vision"] = torch.cat(
                        [self._kv_cache[i]["k_vision"], new_kv[i]["k_vision"]], dim=2
                    )
                    self._kv_cache[i]["v_vision"] = torch.cat(
                        [self._kv_cache[i]["v_vision"], new_kv[i]["v_vision"]], dim=2
                    )

        # 7. Select output
        result = video_out if solver == "cm" else latents

        # 8. Gradient masking
        if self.enable_gradient_masking:
            gradient_mask = create_gradient_mask(
                total_frames=num_frames,
                last_n_frames=self.gradient_mask_last_n_frames,
                shape=result.shape,
                frame_dim=2,  # HY15 layout: [B, C, T, H, W], frame at dim=2
                device=self.device,
            )
            result = apply_gradient_mask(result, gradient_mask)

        return result

    # ==================== DMD Loss Methods ====================

    def _generator_forward(
        self, generator_pred_video: torch.Tensor, training_batch: TrainingBatch
    ) -> torch.Tensor:
        """
        Compute DMD loss using real and fake score teachers with CFG.

        DMD Loss: loss = 0.5 * MSE(original, original - grad.detach())
        grad = (fake_pred - real_pred) / |original - real_pred|

        Reference: FastVideo distillation_pipeline.py Line 612-687
        """

        original_latent = generator_pred_video

        with torch.no_grad():
            # Renoise: uniform in [denoising_step_list[k_i+1], denoising_step_list[k_i]], then shift
            # Same procedure as _fake_score_forward — both scorers see identically-constructed input.
            batch_size = generator_pred_video.shape[0]
            hi = self.denoising_step_list[self._last_ki]
            lo = self.denoising_step_list[self._last_ki + 1] if self._last_ki + 1 < len(self.denoising_step_list) else 0
            u = torch.rand(batch_size, device=self.device, generator=self.noise_gen_cuda)
            if self.sp_world_size > 1:
                self.sp_group.broadcast(u, src=0)
            raw_t = lo + u * (hi - lo)
            # raw_t is in step space [lo, hi]; train_index = 1000 - raw_t directly
            timestep = (self.num_train_timesteps - raw_t).long()
            timestep = timestep.clamp(self.min_timestep, self.max_timestep)
            sigma = self.infer_scheduler.get_sigma(timestep)
            # print(f"[Rank {self.global_rank}] GENERATOR_FORWARD k_i={self._last_ki} gen_sigma={self._last_gen_sigma:.4f} renoise_range=[{lo},{hi}] renoise_sigma={sigma.tolist()} step={self.current_trainstep}", flush=True)

            # Add noise to generator_pred_video
            noise = torch.randn(generator_pred_video.shape, dtype=generator_pred_video.dtype, device=generator_pred_video.device, generator=self.noise_gen_cuda)
            sigma_bcast = sigma.reshape(batch_size, 1, 1, 1, 1)  # [B, 1, 1, 1, 1]
            noisy_latent = (1 - sigma_bcast) * generator_pred_video + sigma_bcast * noise

            # Build teacher input for noisy latent
            multitask_mask = self.get_task_mask("i2v", noisy_latent.shape[2])
            cond_latents = self._prepare_cond_latents(
                "i2v", training_batch.image_cond, noisy_latent, multitask_mask
            )
            latents_concat = torch.concat([noisy_latent, cond_latents], dim=1).to(torch.bfloat16)

            # Fake score prediction (bidirectional model, being trained)
            num_frames = noisy_latent.shape[2]
            timestep_expanded = timestep.unsqueeze(1).expand(batch_size, num_frames).flatten()
            teacher_kwargs = {
                "hidden_states": latents_concat,
                "timestep": timestep_expanded.to(self.device, dtype=torch.bfloat16),
                "timestep_txt": torch.tensor(0).unsqueeze(0).to(self.device, dtype=torch.bfloat16),
                "text_states": training_batch.prompt_embed,
                "text_states_2": None,
                "encoder_attention_mask": training_batch.prompt_mask,
                "timestep_r": None,
                "vision_states": training_batch.vision_states,
                "mask_type": "i2v",
                "guidance": None,
                "extra_kwargs": {
                    "byt5_text_states": training_batch.byt5_text_states,
                    "byt5_text_mask": training_batch.byt5_text_mask,
                },
                "return_dict": False,
            }

            fake_score_pred_noise = self.fake_score_transformer(**teacher_kwargs)[0]
            # model output [B, C, T, H, W] -> permute -> [B, T, C, H, W]
            fake_score_pred_noise = fake_score_pred_noise.permute(0, 2, 1, 3, 4)
            _nl = noisy_latent.permute(0, 2, 1, 3, 4)  # [B, T, C, H, W]
            faker_score_pred_video = self.infer_scheduler.pred_noise_to_pred_video(
                fake_score_pred_noise.flatten(0, 1),
                _nl.flatten(0, 1),
                timestep_expanded
            ).unflatten(0, fake_score_pred_noise.shape[:2]).permute(0, 2, 1, 3, 4)  # back [B, C, T, H, W]

            # Move real score teacher to GPU for inference
            self.real_score_transformer.to(self.device)

            # Real score cond forward
            real_score_pred_noise_cond = self.real_score_transformer(**teacher_kwargs)[0]
            real_score_pred_noise_cond = real_score_pred_noise_cond.permute(0, 2, 1, 3, 4)
            pred_real_video_cond = self.infer_scheduler.pred_noise_to_pred_video(
                real_score_pred_noise_cond.flatten(0, 1),
                _nl.flatten(0, 1),
                timestep_expanded
            ).unflatten(0, real_score_pred_noise_cond.shape[:2]).permute(0, 2, 1, 3, 4)

            # Real score uncond forward — use real neg prompt if available, else zeros
            B = training_batch.prompt_embed.shape[0]
            if self.neg_prompt_pt is not None:
                uncond_text = self.neg_prompt_pt['negative_prompt_embeds'][0].to(self.device, dtype=torch.bfloat16).unsqueeze(0).expand(B, -1, -1)
                uncond_mask = self.neg_prompt_pt['negative_prompt_mask'][0].to(self.device, dtype=torch.bfloat16).unsqueeze(0).expand(B, -1)
                uncond_byt5 = self.neg_byt5_pt['byt5_text_states'][0].to(self.device, dtype=torch.bfloat16).unsqueeze(0).expand(B, -1, -1)
                uncond_byt5_mask = self.neg_byt5_pt['byt5_text_mask'][0].to(self.device, dtype=torch.bfloat16).unsqueeze(0).expand(B, -1)
            else:
                uncond_text = torch.zeros_like(training_batch.prompt_embed)
                uncond_mask = training_batch.prompt_mask
                uncond_byt5 = torch.zeros_like(training_batch.byt5_text_states)
                uncond_byt5_mask = training_batch.byt5_text_mask
            teacher_kwargs_uncond = {
                "hidden_states": latents_concat,
                "timestep": timestep_expanded.to(self.device, dtype=torch.bfloat16),
                "timestep_txt": torch.tensor(0).unsqueeze(0).to(self.device, dtype=torch.bfloat16),
                "text_states": uncond_text,
                "text_states_2": None,
                "encoder_attention_mask": uncond_mask,
                "timestep_r": None,
                "vision_states": training_batch.vision_states,
                "mask_type": "i2v",
                "guidance": None,
                "extra_kwargs": {
                    "byt5_text_states": uncond_byt5,
                    "byt5_text_mask": uncond_byt5_mask,
                },
                "return_dict": False,
            }

            real_score_pred_noise_uncond = self.real_score_transformer(**teacher_kwargs_uncond)[0]
            real_score_pred_noise_uncond = real_score_pred_noise_uncond.permute(0, 2, 1, 3, 4)
            pred_real_video_uncond = self.infer_scheduler.pred_noise_to_pred_video(
                real_score_pred_noise_uncond.flatten(0, 1),
                _nl.flatten(0, 1),
                timestep_expanded
            ).unflatten(0, real_score_pred_noise_uncond.shape[:2]).permute(0, 2, 1, 3, 4)

            # Apply CFG: pred = uncond + scale * (cond - uncond)
            real_score_pred_video = apply_cfg(pred_real_video_cond, pred_real_video_uncond, self.cfg_scale)

            # Move real score teacher back to CPU to free GPU memory
            self.real_score_transformer.cpu()
            torch.cuda.empty_cache()

            # Compute gradient (reference: FastVideo Line 672-674)
            grad = compute_kl_gradient(
                fake_x0=faker_score_pred_video,
                real_x0=real_score_pred_video,
                generator_output=original_latent,
                normalize_mode="global",  # HY15 uses global scalar normalizer
            )

        # Compute DMD loss (reference: FastVideo Line 676)
        loss = dmd_generator_loss(
            generator_output=original_latent,
            kl_gradient=grad,
        )

        return loss

    def _fake_score_forward(self, training_batch: TrainingBatch) -> torch.Tensor:
        """
        Compute fake score loss using flow matching.

        The critic learns to predict the flow from noise to generator output.
        """
        # Get generator prediction under no_grad: prevents FSDP2 from registering
        # post_backward_hooks during exit-step forward passes. Using .detach() alone
        # is insufficient — hooks are registered at forward time, not backward time,
        # and never fire when backward is cut, causing FSDP2 state leak → OOM → NCCL hang.
        # Reference: FastVideo distillation_pipeline.py line 689-695
        with torch.no_grad():
            generator_pred = self._generator_multi_step_simulation_forward(training_batch)

        # Sample timesteps for flow matching
        batch_size = generator_pred.shape[0]
        num_frames = generator_pred.shape[2]

        # Renoise timestep: uniform in [denoising_step_list[k_i+1], denoising_step_list[k_i]], then shift
        hi = self.denoising_step_list[self._last_ki]  # e.g. 500
        lo = self.denoising_step_list[self._last_ki + 1] if self._last_ki + 1 < len(self.denoising_step_list) else 0
        u = torch.rand(batch_size, device=self.device, generator=self.noise_gen_cuda)
        if self.sp_world_size > 1:
            self.sp_group.broadcast(u, src=0)
        # Uniform in [lo, hi], then convert to train index via 1000 - val
        raw_t = lo + u * (hi - lo)  # in [lo, hi] step space
        # train_index = 1000 - raw_t directly (no extra shift needed)
        timestep = (self.num_train_timesteps - raw_t).long()
        timestep = timestep.clamp(self.min_timestep, self.max_timestep)

        # Get noise
        noise = torch.randn(generator_pred.shape, dtype=generator_pred.dtype, device=generator_pred.device, generator=self.noise_gen_cuda)

        sigma = self.infer_scheduler.get_sigma(timestep).reshape(batch_size, 1, 1, 1, 1)  # [B, 1, 1, 1, 1]
        # print(f"[Rank {self.global_rank}] FAKE_SCORE_FORWARD k_i={self._last_ki} gen_sigma={self._last_gen_sigma:.4f} renoise_range=[{lo},{hi}] renoise_sigma={sigma.flatten().tolist()} step={self.current_trainstep}", flush=True)
        noisy = (1 - sigma) * generator_pred + sigma * noise  # [B, C, T, H, W]

        # Build fake score input (5D, same convention as _dmd_forward)
        multitask_mask = self.get_task_mask("i2v", num_frames)
        cond_latents = self._prepare_cond_latents(
            "i2v", training_batch.image_cond, generator_pred, multitask_mask
        )
        latents_concat = torch.concat([noisy, cond_latents], dim=1).to(torch.bfloat16)  # [B, C+C_cond, T, H, W]

        fake_kwargs = {
            "hidden_states": latents_concat,
            "timestep": timestep.unsqueeze(1).expand(batch_size, num_frames).flatten().to(self.device, dtype=torch.bfloat16),
            "timestep_txt": torch.tensor(0).unsqueeze(0).to(self.device, dtype=torch.bfloat16),
            "text_states": training_batch.prompt_embed,
            "text_states_2": None,
            "encoder_attention_mask": training_batch.prompt_mask,
            "timestep_r": None,
            "vision_states": training_batch.vision_states,
            "mask_type": "i2v",
            "guidance": None,
            "extra_kwargs": {
                "byt5_text_states": training_batch.byt5_text_states,
                "byt5_text_mask": training_batch.byt5_text_mask,
            },
            "return_dict": False,
        }

        # Fake score prediction
        fake_pred = self.fake_score_transformer(**fake_kwargs)[0]  # [B, C, T, H, W]

        # Flow matching loss: MSE between predicted noise and target
        # noisy = (1-sigma)*original + sigma*noise  =>  target velocity = noise - original
        _orig = generator_pred.permute(0, 2, 1, 3, 4).flatten(0, 1)  # [B*T, C, H, W]
        _noise = noise.permute(0, 2, 1, 3, 4).flatten(0, 1)
        fake_pred_flat = fake_pred.permute(0, 2, 1, 3, 4).flatten(0, 1)
        loss = dmd_critic_loss(critic_flow_pred=fake_pred_flat, noise=_noise, clean=_orig)

        return loss

    # ==================== Training Methods ====================

    def train_one_step(self, training_batch: TrainingBatch) -> TrainingBatch:
        """
        Self-forcing training step with alternating updates and gradient accumulation.

        Generator trains every N steps, fake_score trains N-1 steps.
        Reference: FastVideo self_forcing_distillation_pipeline.py Line 560-682
        """
        gradient_accumulation_steps = getattr(self.training_args, 'gradient_accumulation_steps', 1)
        train_generator = (self.current_trainstep % self.dfake_gen_update_ratio == 0)

        generator_loss = 0.0
        fake_score_loss = 0.0

        # Prepare batches for gradient accumulation
        batches = []
        for _ in range(gradient_accumulation_steps):
            fresh_batch = TrainingBatch()
            batch = self._get_next_batch(fresh_batch)
            batch = self._prepare_ar_dit_inputs(batch)
            batch = self._build_input_kwargs(batch)
            cloned_batch = clone_training_batch(batch)
            batches.append(cloned_batch)

        if train_generator:
            logger.debug("Training generator at step %s", self.current_trainstep)
            self.optimizer.zero_grad()
            self.transformer.train()
            total_generator_loss = 0.0

            for batch in batches:
                generator_pred = self._generator_multi_step_simulation_forward(batch)
                dmd_loss = self._generator_forward(generator_pred, batch)
                (dmd_loss / gradient_accumulation_steps).backward()
                total_generator_loss += dmd_loss.detach().item()

            grad_norm = clip_grad_norm_while_handling_failing_dtensor_cases(
                [p for p in self.transformer.parameters() if p.requires_grad],
                self.training_args.max_grad_norm,
            )

            self.optimizer.step()
            self.lr_scheduler.step()

            if self.generator_ema is not None:
                self.generator_ema.update(self.transformer)

            avg_generator_loss = torch.tensor(total_generator_loss / gradient_accumulation_steps, device=self.device)
            world_group = get_world_group()
            world_group.all_reduce(avg_generator_loss, op=torch.distributed.ReduceOp.AVG)
            generator_loss = avg_generator_loss.item()
        else:
            logger.debug("Training critic at step %s", self.current_trainstep)
            self.fake_score_optimizer.zero_grad()
            self.fake_score_transformer.train()
            total_critic_loss = 0.0

            for batch in batches:
                critic_loss = self._fake_score_forward(batch)
                (critic_loss / gradient_accumulation_steps).backward()
                total_critic_loss += critic_loss.detach().item()

            grad_norm = clip_grad_norm_while_handling_failing_dtensor_cases(
                [p for p in self.fake_score_transformer.parameters() if p.requires_grad],
                self.training_args.max_grad_norm,
            )

            self.fake_score_optimizer.step()
            self.fake_score_lr_scheduler.step()

            avg_critic_loss = torch.tensor(total_critic_loss / gradient_accumulation_steps, device=self.device)
            world_group = get_world_group()
            world_group.all_reduce(avg_critic_loss, op=torch.distributed.ReduceOp.AVG)
            fake_score_loss = avg_critic_loss.item()

        self.current_trainstep += 1

        training_batch.generator_loss = generator_loss
        training_batch.fake_score_loss = fake_score_loss
        training_batch.total_loss = generator_loss + fake_score_loss
        training_batch.grad_norm = grad_norm.item()

        return training_batch

    # ==================== Main Training Loop ====================

    def train(self) -> None:
        """Main training loop for self-forcing DMD."""
        assert self.seed is not None, "seed must be set"
        set_random_seed(self.seed + self.global_rank)
        logger.info('Rank: %s, starting self-forcing DMD training', self.global_rank)

        if not self.post_init_called:
            self.post_init()

        # Initialize random generators
        self.noise_random_generator = torch.Generator(device="cpu").manual_seed(self.seed)
        self.noise_gen_cuda = torch.Generator(device="cuda").manual_seed(self.seed)

        # Resume from checkpoint if needed
        if self.training_args.resume_from_checkpoint:
            self._resume_from_checkpoint()
            self.current_trainstep = self.init_steps
        else:
            self.current_trainstep = 0

        self.train_loader_iter = iter(self.train_dataloader)

        # Progress bar
        progress_bar = tqdm(
            range(0, self.training_args.max_train_steps),
            initial=0,
            desc="Steps",
            disable=self.local_rank > 0,
        )

        for step in range(1, self.training_args.max_train_steps + 1):
            # Update max_frames for curriculum learning (gradually increase during training)
            if hasattr(self.train_dataset, 'update_max_frames'):
                self.train_dataset.update_max_frames(step)

            import time
            start_time = time.perf_counter()

            # === EMA creation logic (reference: FastVideo Line 857-865) ===
            ema_start_step = getattr(self.training_args, 'ema_start_step', 100)
            ema_decay = getattr(self.training_args, 'ema_decay', 0.0)
            if (step >= ema_start_step) and (self.generator_ema is None) and (ema_decay > 0):
                self.generator_ema = EMA(self.transformer, decay=ema_decay, mode="local_shard")
                logger.info("Created generator EMA at step %s with decay=%s", step, ema_decay)

            # Train one step
            training_batch = TrainingBatch()
            training_batch = self.train_one_step(training_batch)

            # Log metrics
            if self.global_rank == 0:
                import wandb
                wandb.log({
                    "train_generator_loss": training_batch.generator_loss,
                    "train_fake_score_loss": training_batch.fake_score_loss,
                    "train_total_loss": training_batch.total_loss,
                    "learning_rate": self.lr_scheduler.get_last_lr()[0],
                    "fake_score_learning_rate": self.fake_score_lr_scheduler.get_last_lr()[0],
                    "grad_norm": training_batch.grad_norm,
                }, step=step)

            progress_bar.set_postfix({
                "gen_loss": f"{training_batch.generator_loss:.4f}",
                "fake_loss": f"{training_batch.fake_score_loss:.4f}",
                "grad_norm": f"{training_batch.grad_norm:.2f}",
            })
            progress_bar.update(1)

            # Save checkpoint
            if step % self.training_args.checkpointing_steps == 0:
                # Save student (online) weights
                save_checkpoint(
                    self.transformer, self.global_rank,
                    self.training_args.output_dir, step,
                    self.optimizer, self.train_dataloader,
                    self.lr_scheduler, self.noise_random_generator,
                )

                # Save EMA weights via apply_to_model → save_checkpoint (FSDP-safe)
                if self.generator_ema is not None:
                    import os
                    ema_save_dir = os.path.join(self.training_args.output_dir, f"ema_checkpoint-{step}")
                    with self.generator_ema.apply_to_model(self.transformer):
                        save_checkpoint(
                            self.transformer, self.global_rank,
                            ema_save_dir, step,
                            None, None, None, None,
                        )
                    logger.info("EMA checkpoint saved to %s", ema_save_dir)

                self.transformer.train()
                self.sp_group.barrier()

        # Save final checkpoint (student)
        save_checkpoint(
            self.transformer, self.global_rank,
            self.training_args.output_dir,
            self.training_args.max_train_steps,
            self.optimizer, self.train_dataloader,
            self.lr_scheduler, self.noise_random_generator,
        )

        # Save final EMA checkpoint
        if self.generator_ema is not None:
            import os
            ema_save_dir = os.path.join(self.training_args.output_dir,
                                        f"ema_checkpoint-{self.training_args.max_train_steps}")
            with self.generator_ema.apply_to_model(self.transformer):
                save_checkpoint(
                    self.transformer, self.global_rank,
                    ema_save_dir, self.training_args.max_train_steps,
                    None, None, None, None,
                )
            logger.info("Final EMA checkpoint saved to %s", ema_save_dir)

        if get_sp_group():
            cleanup_dist_env_and_memory()
