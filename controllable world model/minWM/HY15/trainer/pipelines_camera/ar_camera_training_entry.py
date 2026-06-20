# SPDX-License-Identifier: Apache-2.0
import dataclasses
import math
import os
import time
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Iterator
from typing import Any

import imageio
import numpy as np
import torch
import torch.distributed as dist
import torchvision
from hyvideo.schedulers.scheduling_flow_match_discrete import FlowMatchDiscreteScheduler as LocalFlowMatchScheduler
from einops import rearrange
from torch.utils.data import DataLoader
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm.auto import tqdm


from trainer.distributed.parallel_state import (get_sp_parallel_rank,
                                                  get_sp_world_size)

import trainer.envs as envs
from trainer.configs.sample import SamplingParam
from trainer.dataset_camera import build_camera_plucker_dataloader
from trainer.dataset_camera.dataloader.schema import pyarrow_schema_t2v, pyarrow_schema_i2v
from trainer.dataset_camera.action_utils import discretize_poses_to_actions
from trainer.dataset_camera.validation_dataset import ValidationDataset
from trainer.distributed import (cleanup_dist_env_and_memory,
                                   get_local_torch_device, get_sp_group,
                                   get_world_group)
from trainer.trainer_args import TrainerArgs, TrainingArgs, WorkloadType
from trainer.forward_context import set_forward_context
from trainer.logger import init_logger
from trainer.pipelines import (ComposedPipelineBase, ForwardBatch,
                                 TrainingBatch)
from trainer.training.activation_checkpoint import (
    apply_activation_checkpointing)
from trainer.training.training_utils import (
    clip_grad_norm_while_handling_failing_dtensor_cases,
    compute_density_for_timestep_sampling, get_scheduler, get_sigmas,
    load_checkpoint, normalize_dit_input, save_checkpoint,
)
from trainer.utils import set_random_seed, shallow_asdict
# import muon optimizer
from trainer.training.muon import get_muon_optimizer

import wandb  # isort: skip

logger = init_logger(__name__)


def _get_trainable_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def merge_tensor_by_mask(tensor_1, tensor_2, mask, dim):
    assert tensor_1.shape == tensor_2.shape
    # Mask is a 0/1 vector. Choose tensor_2 when the value is 1; otherwise, tensor_1
    masked_indices = torch.nonzero(mask).squeeze(1)
    tmp = tensor_1.clone()
    if dim == 0:
        tmp[masked_indices] = tensor_2[masked_indices]
    elif dim == 1:
        tmp[:, masked_indices] = tensor_2[:, masked_indices]
    elif dim == 2:
        tmp[:, :, masked_indices] = tensor_2[:, :, masked_indices]
    return tmp

class ARCameraTrainingPipeline(ComposedPipelineBase):
    """
    A pipeline for training a model. All training pipelines should inherit from this class.
    All reusable components and code should be implemented in this class.
    """
    _required_config_modules = ["scheduler", "transformer", "vae"]
    validation_pipeline: ComposedPipelineBase
    train_dataloader: StatefulDataLoader
    train_loader_iter: Iterator[dict[str, Any]]
    current_epoch: int = 0

    def __init__(
            self,
            model_path: str,
            trainer_args: TrainingArgs,
            required_config_modules: list[str] | None = None,
            loaded_modules: dict[str, torch.nn.Module] | None = None) -> None:
        trainer_args.inference_mode = False
        self.lora_training = trainer_args.lora_training
        if self.lora_training and trainer_args.lora_rank is None:
            raise ValueError("lora rank must be set when using lora training")

        set_random_seed(trainer_args.seed)  # for lora param init
        super().__init__(model_path, trainer_args, required_config_modules,
                         loaded_modules)  # type: ignore
    
    def initialize_pipeline(self, trainer_args: TrainerArgs):
        pass

    def create_training_stages(self, training_args: TrainingArgs):
        pass

    def initialize_validation_pipeline(self, training_args: TrainingArgs):
        pass


    def create_pipeline_stages(self, trainer_args: TrainerArgs):
        raise RuntimeError(
            "create_pipeline_stages should not be called for training pipeline")

    def set_schemas(self) -> None:
        if self.training_args.workload_type == WorkloadType.I2V:
            self.train_dataset_schema = pyarrow_schema_i2v
        else:
            self.train_dataset_schema = pyarrow_schema_t2v

    def initialize_training_pipeline(self, training_args: TrainingArgs):
        logger.info("Initializing training pipeline...")
        self.device = get_local_torch_device()
        self.training_args = training_args
        world_group = get_world_group()
        self.world_size = world_group.world_size
        self.global_rank = world_group.rank
        self.sp_group = get_sp_group()
        self.rank_in_sp_group = self.sp_group.rank_in_group
        self.sp_world_size = self.sp_group.world_size
        self.local_rank = world_group.local_rank
        self.transformer = self.get_module("transformer")
        self.seed = training_args.seed
        self.task_type = training_args.workload_type.value  # "i2v" or "t2v"
        self.set_schemas()
        # self.action = training_args.action  # Removed: camera control
        # add the causal option
        self.causal = training_args.causal
        self.training_args.use_teacher_forcing = self.causal

        if not self.causal:
            self.transformer.set_attn_mode('flash')
            logger.info("Non-causal training: set attn_mode='flash' (bidirectional)")

        assert hasattr(self.transformer.double_blocks[0], 'img_attn_prope_proj'), \
            "ProPE (img_attn_prope_proj) missing — should be created during FSDP loading."
        logger.info("ProPE projection layers verified.")
        if getattr(training_args, 'use_discrete_action', False):
            if not hasattr(self.transformer, 'action_in'):
                logger.warning(
                    "use_discrete_action=True but action_in missing. "
                    "FSDP should have created it. Adding as zero-init fallback.")
                self.transformer.add_discrete_action_parameters()
            else:
                logger.info("WorldPlay: action_in loaded from checkpoint.")
        else:
            if hasattr(self.transformer, 'action_in'):
                logger.warning(
                    "use_discrete_action=False but action_in found (from WorldPlay ckpt). Removing.")
                del self.transformer.action_in
        self.train_time_shift = training_args.train_time_shift

        # Set random seeds for deterministic training
        assert self.seed is not None, "seed must be set"
        set_random_seed(self.seed)
        self.transformer.train()
        if training_args.enable_gradient_checkpointing_type is not None:
            self.transformer = apply_activation_checkpointing(
                self.transformer,
                checkpointing_type=training_args.
                enable_gradient_checkpointing_type)

        self.set_trainable()
        params_to_optimize = self.transformer.parameters()
        params_to_optimize = list(
            filter(lambda p: p.requires_grad, params_to_optimize))
        betas = [float(x) for x in training_args.betas.split(",")]
        self.optimizer = get_muon_optimizer(
            model=self.transformer,
            lr=training_args.learning_rate,                      # Learning rate
            weight_decay=training_args.weight_decay,  # Weight decay
            adamw_betas=betas,   # AdamW betas for 1D parameters
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

        self.train_dataset, self.train_dataloader = build_camera_plucker_dataloader(
            json_path=training_args.json_path,
            causal=training_args.causal,
            window_frames=training_args.window_frames,
            batch_size=training_args.train_batch_size,
            num_data_workers=training_args.dataloader_num_workers,
            drop_last=True,
            drop_first_row=False,
            seed=self.seed,
            cfg_rate=training_args.training_cfg_rate,
            i2v_rate=training_args.i2v_rate,
            task_type=self.task_type,
        )

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

        # Offload VAE to CPU to save GPU memory during training
        # VAE is only used for reading config (latents_mean/std) during training
        self.get_module("vae").cpu()

        if training_args.log_validation:
            import json
            val_json = getattr(training_args, "validation_dataset_file", None) or training_args.json_path
            with open(val_json) as f:
                index = json.load(f)
            self.validation_samples = []
            wf = training_args.window_frames
            for i in range(min(8, len(index))):
                data = torch.load(index[i]["latent_path"], map_location="cpu", weights_only=True)
                latent = data["latent"]
                if latent.dim() == 5:
                    t_crop = min(wf, latent.shape[2])
                    data["latent"] = latent[:, :, :t_crop, ...]
                self.validation_samples.append(data)
            logger.info("Loaded %d fixed validation samples from %s (cropped to T=%d)", len(self.validation_samples), val_json, wf)

    def initialize_validation_pipeline(self, training_args: TrainingArgs):
        pass

    def _prepare_training(self, training_batch: TrainingBatch) -> TrainingBatch:
        self.transformer.train()
        self.optimizer.zero_grad()
        training_batch.total_loss = 0.0
        return training_batch

    def _get_next_batch(self, training_batch: TrainingBatch) -> TrainingBatch:
        batch = next(self.train_loader_iter, None)  # type: ignore
        if batch is None:
            self.current_epoch += 1
            logger.info("Starting epoch %s", self.current_epoch)
            # Reset iterator for next epoch
            self.train_loader_iter = iter(self.train_dataloader)
            # Get first batch of new epoch
            batch = next(self.train_loader_iter)

        latents = batch["latent"]
        prompt_embed = batch["prompt_embed"]

        # Removed: camera control (w2c, intrinsic, action) processing
        video_path = batch.get('video_path', batch.get('path'))
        image_cond = batch.get('image_cond')
        vision_states = batch.get('vision_states')
        prompt_mask = batch.get('prompt_mask')
        byt5_text_states = batch.get('byt5_text_states')
        byt5_text_mask = batch.get('byt5_text_mask')
        # add an indicator for memory training
        select_window_out_flag = batch.get('select_window_out_flag', 0)
        i2v_mask = batch.get('i2v_mask')
        viewmats = batch.get('viewmats')
        Ks = batch.get('Ks')
        action = batch.get('action')

        if self.global_rank == 0 and training_batch.current_timestep == 1:
            logger.info("First batch shapes: " + ", ".join(
                f"{k}: {v.shape}" for k, v in batch.items() if hasattr(v, 'shape')))
            if viewmats is not None:
                logger.info(f"viewmats shape: {viewmats.shape}, sample frame [0,0]:\n{viewmats[0, 0]}")
            if Ks is not None:
                logger.info(f"Ks shape: {Ks.shape}, sample frame [0,0]:\n{Ks[0, 0]}")
            if viewmats is not None:
                logger.info(f"viewmats shape: {viewmats.shape}, Ks shape: {Ks.shape}")
                logger.info(f"viewmats[0,0] (first frame w2c 4x4):\n{viewmats[0, 0]}")
                logger.info(f"Ks[0,0] (first frame intrinsics 3x3):\n{Ks[0, 0]}")

        training_batch.latents = latents.to(get_local_torch_device(),
                                            dtype=torch.bfloat16)
        training_batch.prompt_embed = prompt_embed.to(
            get_local_torch_device(), dtype=torch.bfloat16)
        training_batch.video_path = video_path[0] if isinstance(video_path, list) else video_path

        training_batch.image_cond = image_cond.to(
            get_local_torch_device(), dtype=torch.bfloat16)
        training_batch.vision_states = vision_states.to(
            get_local_torch_device(), dtype=torch.bfloat16)
        training_batch.prompt_mask = prompt_mask.to(
            get_local_torch_device(), dtype=torch.bfloat16)
        training_batch.byt5_text_states = byt5_text_states.to(
            get_local_torch_device(), dtype=torch.bfloat16)
        training_batch.byt5_text_mask = byt5_text_mask.to(
            get_local_torch_device(), dtype=torch.bfloat16)
        training_batch.select_window_out_flag = select_window_out_flag[0] if isinstance(select_window_out_flag, (list, torch.Tensor)) else select_window_out_flag
        training_batch.i2v_mask = i2v_mask.to(
            get_local_torch_device(), dtype=torch.bfloat16)    # i2v mask only works for memory training
        if viewmats is not None:
            training_batch.viewmats = viewmats.to(
                get_local_torch_device(), dtype=torch.bfloat16)
            training_batch.Ks = Ks.to(
                get_local_torch_device(), dtype=torch.bfloat16)
        if action is not None:
            training_batch.action = action.to(
                get_local_torch_device(), dtype=torch.bfloat16)

        return training_batch

    def timestep_transform(self, t, shift=1.0, num_timesteps=1000.0):
        t = t / num_timesteps
        t = shift * t / (1 + (shift - 1) * t)
        t = t * num_timesteps
        return t

    def _prepare_ar_dit_inputs(self,
                            training_batch: TrainingBatch) -> TrainingBatch:
        latents = training_batch.latents
        batch_size = latents.shape[0]
        latent_t = latents.shape[2]
        latent_h = latents.shape[3]
        latent_w = latents.shape[4]
        noise = torch.randn(latents.shape,
                            generator=self.noise_gen_cuda,
                            device=latents.device,
                            dtype=latents.dtype)

        u = compute_density_for_timestep_sampling(
            weighting_scheme=self.training_args.weighting_scheme,
            batch_size=batch_size,
            generator=self.noise_random_generator,
            logit_mean=self.training_args.logit_mean,
            logit_std=self.training_args.logit_std,
            mode_scale=self.training_args.mode_scale,
        )
        indices = (u * self.noise_scheduler.config.num_train_timesteps).long()
        indices = (self.noise_scheduler.config.num_train_timesteps - self.timestep_transform(indices, self.train_time_shift)).long()
        # Repeat the same timestep for all T frames: [B] -> [B*T]
        indices = indices.unsqueeze(-1).repeat_interleave(latent_t, dim=-1).reshape(-1)
        
        timesteps = self.noise_scheduler.timesteps[indices].to(device=self.device)
        if self.training_args.sp_size > 1:
            # Make sure that the timesteps are the same across all sp processes.
            sp_group = get_sp_group()
            sp_group.broadcast(timesteps, src=0)


        sigmas = get_sigmas(
            self.noise_scheduler,
            latents.device,
            timesteps,
            n_dim=latents.ndim,
            dtype=latents.dtype,
        )
        sigmas = rearrange(sigmas, '(B D) C T H W -> B C (D T) H W', D=latent_t)
        noisy_model_input = (1.0 -
                             sigmas) * training_batch.latents + sigmas * noise
        training_batch.noisy_model_input = noisy_model_input
        training_batch.timesteps = timesteps
        training_batch.sigmas = sigmas
        training_batch.noise = noise
        training_batch.raw_latent_shape = training_batch.latents.shape

        # Teacher forcing: prepare clean latents and their timesteps
        if self.training_args.use_teacher_forcing:
            if self.training_args.noise_augmentation_max_timestep > 0:
                # Add small noise to clean latents for robustness
                aug_u = compute_density_for_timestep_sampling(
                    weighting_scheme="uniform",
                    batch_size=batch_size * latent_t,
                    generator=self.noise_random_generator,
                )
                aug_indices = (aug_u * self.training_args.noise_augmentation_max_timestep).long()
                aug_indices = aug_indices.clamp(0, self.noise_scheduler.config.num_train_timesteps - 1)
                aug_timesteps_flat = self.noise_scheduler.timesteps[aug_indices].to(device=self.device)
                aug_sigmas = get_sigmas(
                    self.noise_scheduler, latents.device, aug_timesteps_flat,
                    n_dim=latents.ndim, dtype=latents.dtype,
                )
                aug_sigmas = rearrange(aug_sigmas, '(B D) C T H W -> B C (D T) H W', D=latent_t)
                clean_x = (1.0 - aug_sigmas) * training_batch.latents + aug_sigmas * noise
                training_batch.clean_x = clean_x
                training_batch.aug_timesteps = aug_timesteps_flat
            else:
                # Use exact ground truth as clean tokens
                training_batch.clean_x = training_batch.latents.clone()
                training_batch.aug_timesteps = torch.zeros_like(timesteps)

        return training_batch

    def _sp_consistency_check(self, training_batch: TrainingBatch, step: int) -> None:
        """
        Verify that all ranks in the same SP group hold identical input tensors.

        Runs every ``SP_CONSISTENCY_CHECK_INTERVAL`` steps (env var, default 50).
        Set to 0 to disable.  For each tensor we all-gather a scalar fingerprint
        (sum-of-abs in float64) from every SP rank and compare.  Cost: one
        all-gather of a 1-element tensor per checked field — negligible.
        """
        if self.sp_world_size <= 1:
            return
        interval = int(os.environ.get("SP_CONSISTENCY_CHECK_INTERVAL", "50"))
        if interval <= 0 or step % interval != 0:
            return

        sp_group = get_sp_group()
        rank_in_sp = self.rank_in_sp_group

        def _fingerprint(t: torch.Tensor) -> torch.Tensor:
            return t.detach().double().abs().sum().unsqueeze(0).to(self.device)

        # ── Tensors that MUST be identical across all SP ranks ──
        check_tensors: dict[str, torch.Tensor | None] = {
            # Core training inputs
            "latents": training_batch.latents,
            "noise": training_batch.noise,
            "timesteps": training_batch.timesteps,
            "noisy_model_input": training_batch.noisy_model_input,
            "sigmas": training_batch.sigmas,
            # Text embeddings (affected by CFG dropout in dataset)
            "prompt_embed": training_batch.prompt_embed,
            "prompt_mask": getattr(training_batch, 'prompt_mask', None),
            "byt5_text_states": getattr(training_batch, 'byt5_text_states', None),
            "byt5_text_mask": getattr(training_batch, 'byt5_text_mask', None),
            # Image conditioning
            "image_cond": getattr(training_batch, 'image_cond', None),
            "vision_states": getattr(training_batch, 'vision_states', None),
            # Loss mask
            "i2v_mask": getattr(training_batch, 'i2v_mask', None),
            # Camera matrices for PRoPE
            "viewmats": getattr(training_batch, 'viewmats', None),
            "Ks": getattr(training_batch, 'Ks', None),
        }
        if self.training_args.use_teacher_forcing:
            check_tensors["clean_x"] = getattr(training_batch, 'clean_x', None)
            check_tensors["aug_timesteps"] = getattr(training_batch, 'aug_timesteps', None)

        # ── Also check noise_random_generator state (CPU generator) ──
        if hasattr(self, 'noise_random_generator') and self.noise_random_generator is not None:
            gen_state = self.noise_random_generator.get_state()
            state_hash = gen_state.to(torch.int8).float().sum().unsqueeze(0).to(self.device)
            check_tensors["noise_rng_state"] = state_hash

        # ── Also check noise_gen_cuda state (CUDA generator) ──
        if hasattr(self, 'noise_gen_cuda') and self.noise_gen_cuda is not None:
            cuda_state = self.noise_gen_cuda.get_state()
            cuda_hash = cuda_state.to(torch.int8).float().sum().unsqueeze(0).to(self.device)
            check_tensors["noise_gen_cuda_state"] = cuda_hash

        mismatches = []
        for name, tensor in check_tensors.items():
            if tensor is None:
                continue
            local_fp = _fingerprint(tensor)
            gathered = [torch.zeros_like(local_fp) for _ in range(self.sp_world_size)]
            dist.all_gather(gathered, local_fp, group=sp_group.device_group)

            ref = gathered[0].item()
            for r in range(1, len(gathered)):
                val = gathered[r].item()
                # Relative tolerance for floats; exact match for int-like (rng state)
                if name in ("noise_rng_state", "noise_gen_cuda_state"):
                    if val != ref:
                        mismatches.append(
                            f"  {name}: rank0={ref}, rank{r}={val} [DIVERGED]"
                        )
                else:
                    rel_diff = abs(val - ref) / max(abs(ref), 1e-12)
                    if rel_diff > 1e-5:
                        mismatches.append(
                            f"  {name}: rank0={ref:.6e}, rank{r}={val:.6e}, "
                            f"rel_diff={rel_diff:.2e}"
                        )

        if mismatches:
            msg = (
                f"\n{'='*60}\n"
                f"[SP CONSISTENCY CHECK FAILED] step={step}, "
                f"sp_rank={rank_in_sp}, global_rank={self.global_rank}\n"
                + "\n".join(mismatches)
                + f"\n{'='*60}"
            )
            logger.error(msg)
            raise RuntimeError(msg)
        else:
            if rank_in_sp == 0:
                n_checked = sum(1 for t in check_tensors.values() if t is not None)
                logger.info(
                    "[SP CHECK PASSED] step=%d, %d tensors consistent across %d SP ranks",
                    step, n_checked, self.sp_world_size,
                )

    def _build_attention_metadata(
            self, training_batch: TrainingBatch) -> TrainingBatch:
        training_batch.attn_metadata = None
        return training_batch

    def _build_rope_idx(self,
                        training_batch: TrainingBatch) -> TrainingBatch:
        rank_in_sp_group = get_sp_parallel_rank()
        per_sp_seq_length = training_batch.latents.shape[2] * training_batch.per_seq_length

        training_batch.current_start = per_sp_seq_length * rank_in_sp_group
        training_batch.current_end = per_sp_seq_length * rank_in_sp_group + per_sp_seq_length
        return training_batch

    def _prepare_cond_latents(self, task_type, cond_latents, latents, multitask_mask):
        """
        Prepare conditional latents and mask for multitask training.

        Args:
            task_type: Type of task ("i2v" or "t2v").
            cond_latents: Conditional latents tensor.
            latents: Main latents tensor.
            multitask_mask: Multitask mask tensor.

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                - latents_concat: Concatenated conditional latents.
                - mask_concat: Concatenated mask tensor.
        """
        latents_concat = None
        mask_concat = None

        if cond_latents is not None and task_type == 'i2v':
            latents_concat = cond_latents.repeat(1, 1, latents.shape[2], 1, 1)
            latents_concat[:, :, 1:, :, :] = 0.0
        else:
            latents_concat = torch.zeros(latents.shape[0], latents.shape[1], latents.shape[2], latents.shape[3],
                                         latents.shape[4]).to(latents.device)

        mask_zeros = torch.zeros(latents.shape[0], 1, latents.shape[2], latents.shape[3], latents.shape[4])
        mask_ones = torch.ones(latents.shape[0], 1, latents.shape[2], latents.shape[3], latents.shape[4])
        mask_concat = merge_tensor_by_mask(mask_zeros.cpu(), mask_ones.cpu(), mask=multitask_mask.cpu(), dim=2).to(
            device=latents.device)

        cond_latents = torch.concat([latents_concat, mask_concat], dim=1)

        return cond_latents

    def get_task_mask(self, task_type, latent_target_length):
        if task_type == "t2v":
            mask = torch.zeros(latent_target_length)
        elif task_type == "i2v":
            mask = torch.zeros(latent_target_length)
            mask[0] = 1.0
        else:
            raise ValueError(f"{task_type} is not supported !")
        return mask

    def _build_input_kwargs(self,
                            training_batch: TrainingBatch) -> TrainingBatch:
        extra_kwargs = {
            "byt5_text_states": training_batch.byt5_text_states,
            "byt5_text_mask": training_batch.byt5_text_mask,
        }

        multitask_mask = self.get_task_mask(self.task_type, training_batch.noisy_model_input.shape[2]).to(self.device)
        cond_latents = self._prepare_cond_latents(
            self.task_type, training_batch.image_cond, training_batch.noisy_model_input, multitask_mask
        )

        latents_concat = torch.concat([training_batch.noisy_model_input, cond_latents], dim=1)

        # Prepare clean_x with same conditioning for teacher forcing
        clean_x_concat = None
        aug_timesteps = None
        if self.training_args.use_teacher_forcing and training_batch.clean_x is not None:
            clean_cond_latents = self._prepare_cond_latents(
                self.task_type, training_batch.image_cond, training_batch.clean_x, multitask_mask
            )
            clean_x_concat = torch.concat([training_batch.clean_x, clean_cond_latents], dim=1)
            aug_timesteps = training_batch.aug_timesteps.to(
                get_local_torch_device(), dtype=torch.bfloat16
            )

        training_batch.input_kwargs = {
            "hidden_states":
            latents_concat,
            "timestep":
            training_batch.timesteps.to(get_local_torch_device(),
                                        dtype=torch.bfloat16),
            "timestep_txt": torch.tensor(0).unsqueeze(0).to(get_local_torch_device(),
                                        dtype=torch.bfloat16), # for ar model, we set txt timestep to 0
            "text_states":
                training_batch.prompt_embed,
            "text_states_2": None,
            "encoder_attention_mask": training_batch.prompt_mask,
            "timestep_r": None,
            "vision_states": training_batch.vision_states,
            "mask_type": self.task_type,
            "guidance": None,
            "extra_kwargs": extra_kwargs,
            "return_dict": False,

            # Teacher forcing
            "clean_x": clean_x_concat,
            "aug_timesteps": aug_timesteps,

            # PRoPE camera control
            "viewmats": training_batch.viewmats,
            "Ks": training_batch.Ks,

            # Discrete action conditioning (WorldPlay)
            "action": getattr(training_batch, 'action', None),
        }
        return training_batch

    def _transformer_forward_and_compute_loss(
            self, training_batch: TrainingBatch) -> TrainingBatch:
        input_kwargs = training_batch.input_kwargs

        with set_forward_context(
                current_timestep=training_batch.current_timestep,
                attn_metadata=training_batch.attn_metadata):
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                model_pred = self.transformer(**input_kwargs)[0]

            if self.training_args.precondition_outputs:
                assert training_batch.sigmas is not None
                model_pred = training_batch.noisy_model_input - model_pred * training_batch.sigmas
            assert training_batch.latents is not None
            assert training_batch.noise is not None
            target = training_batch.latents if self.training_args.precondition_outputs else training_batch.noise - training_batch.latents
            i2v_mask = training_batch.i2v_mask
            if training_batch.select_window_out_flag == 1 and self.causal:
                i2v_mask[:,:,:-4,...] = 0 # only compute the last chunk for outside window training 
            assert model_pred.shape == target.shape, f"model_pred.shape: {model_pred.shape}, target.shape: {target.shape}"

            diff = (model_pred.float() * i2v_mask - target.float() * i2v_mask) ** 2
            loss = diff.sum() / max(i2v_mask.sum(), 1) / self.training_args.gradient_accumulation_steps

            loss.backward()
            avg_loss = loss.detach().clone()

        dist.all_reduce(avg_loss, op=dist.ReduceOp.MAX)
        training_batch.total_loss += avg_loss.item()

        return training_batch

    def _clip_grad_norm(self, training_batch: TrainingBatch) -> TrainingBatch:
        max_grad_norm = self.training_args.max_grad_norm

        # TODO(will): perhaps move this into transformer api so that we can do
        # the following:
        # grad_norm = transformer.clip_grad_norm_(max_grad_norm)
        if max_grad_norm is not None:
            model_parts = [self.transformer]
            grad_norm = clip_grad_norm_while_handling_failing_dtensor_cases(
                [p for m in model_parts for p in m.parameters()],
                max_grad_norm,
                foreach=None,
            )
            assert grad_norm is not float('nan') or grad_norm is not float(
                'inf')
            grad_norm = grad_norm.item() if grad_norm is not None else 0.0
        else:
            grad_norm = 0.0
        training_batch.grad_norm = grad_norm
        return training_batch

    def train_one_step(self, training_batch: TrainingBatch) -> TrainingBatch:
        training_batch = self._prepare_training(training_batch)

        for _ in range(self.training_args.gradient_accumulation_steps):
            training_batch = self._get_next_batch(training_batch)

            training_batch = self._prepare_ar_dit_inputs(training_batch)

            # Periodic SP consistency check (before forward pass, after all data is ready)
            # self._sp_consistency_check(training_batch, training_batch.current_timestep)

            training_batch = self._build_input_kwargs(training_batch)

            training_batch = self._transformer_forward_and_compute_loss(
                training_batch)

        training_batch = self._clip_grad_norm(training_batch)
        grad_norm = torch.tensor(training_batch.grad_norm).to(get_local_torch_device())
        dist.all_reduce(grad_norm, op=dist.ReduceOp.MAX)
        isme = False
        grad_norm_me = torch.tensor(training_batch.grad_norm).to(get_local_torch_device()).item()
        if grad_norm_me >= grad_norm.item():
            isme = True
        training_batch.grad_norm = grad_norm.item()

        if (isme or self.global_rank == 0) and training_batch.grad_norm >= 10.0:
            print("Skipping optimizer step, rank:{}, grad_norm: {}, step: {}, video_path: {}".format(
                self.global_rank, training_batch.grad_norm, training_batch.current_timestep, training_batch.video_path))

        if training_batch.grad_norm < 10.0:  # Removed: (not self.action) - camera control
            self.optimizer.step()
            self.lr_scheduler.step()

        training_batch.total_loss = training_batch.total_loss
        training_batch.grad_norm = training_batch.grad_norm
        return training_batch

    def _resume_from_checkpoint(self) -> None:
        logger.info("Loading checkpoint from %s",
                    self.training_args.resume_from_checkpoint)
        resumed_step = load_checkpoint(
            self.transformer, self.global_rank,
            self.training_args.resume_from_checkpoint, self.optimizer,
            self.train_dataloader, self.lr_scheduler,
            self.noise_random_generator)
        if resumed_step > 0:
            self.init_steps = resumed_step
            logger.info("Successfully resumed from step %s", resumed_step)
        else:
            logger.warning("Failed to load checkpoint, starting from step 0")
            self.init_steps = 0

    def train(self) -> None:
        # torch._inductor.config.max_autotune = True
        assert self.seed is not None, "seed must be set"
        set_random_seed(self.seed + self.global_rank)
        logger.info('rank: %s: start training',
                    self.global_rank,
                    local_main_process_only=False)
        if not self.post_init_called:
            self.post_init()
        num_trainable_params = _get_trainable_params(self.transformer)
        logger.info("Starting training with %s B trainable parameters",
                    round(num_trainable_params / 1e9, 3))

        # Set random seeds for deterministic training
        self.noise_random_generator = torch.Generator(device="cpu").manual_seed(
            self.seed)
        self.noise_gen_cuda = torch.Generator(device="cuda").manual_seed(
            self.seed)
        self.validation_random_generator = torch.Generator(
            device="cpu").manual_seed(self.seed)
        logger.info("Initialized random seeds with seed: %s", self.seed)

        self.noise_scheduler = LocalFlowMatchScheduler(shift=1.0, reverse=True, solver="euler")

        if self.training_args.resume_from_checkpoint:
            self._resume_from_checkpoint()

        self.train_loader_iter = iter(self.train_dataloader)

        step_times: deque[float] = deque(maxlen=100)

        self._log_training_info()

        # Train!
        progress_bar = tqdm(
            range(0, self.training_args.max_train_steps),
            initial=self.init_steps,
            desc="Steps",
            # Only show the progress bar once on each machine.
            disable=self.local_rank > 0,
        )

        if self.training_args.log_validation and self.init_steps == 0:
            initial_validation_batch = TrainingBatch()
            initial_validation_batch.current_timestep = 0
            initial_validation_batch.current_vsa_sparsity = 0.0
            # self._run_validation(initial_validation_batch, 0) # uncomment this for initial validation
            self.transformer.train()

        for step in range(self.init_steps + 1,
                          self.training_args.max_train_steps + 1):

            self.train_dataset.update_max_frames(step)

            start_time = time.perf_counter()
            current_vsa_sparsity = 0.0

            training_batch = TrainingBatch()
            training_batch.current_timestep = step
            training_batch.current_vsa_sparsity = current_vsa_sparsity
            training_batch = self.train_one_step(training_batch)

            loss = training_batch.total_loss
            grad_norm = training_batch.grad_norm

            step_time = time.perf_counter() - start_time
            step_times.append(step_time)
            avg_step_time = sum(step_times) / len(step_times)

            progress_bar.set_postfix({
                "loss": f"{loss:.4f}",
                "step_time": f"{step_time:.2f}s",
                "grad_norm": grad_norm,
            })
            progress_bar.update(1)
            if self.global_rank == 0:
                wandb.log(
                    {
                        "train_loss": loss,
                        "learning_rate": self.lr_scheduler.get_last_lr()[0],
                        "step_time": step_time,
                        "avg_step_time": avg_step_time,
                        "grad_norm": grad_norm,
                        "vsa_sparsity": current_vsa_sparsity,
                    },
                    step=step,
                )

            if self.training_args.log_validation and step % self.training_args.validation_steps == 0:
                # self._run_validation(training_batch, step)
                self.transformer.train()

            if step % self.training_args.checkpointing_steps == 0:
                print(f"[Checkpoint] Saving checkpoint at step {step} START ...")
                save_checkpoint(self.transformer, self.global_rank,
                                self.training_args.output_dir, step,
                                self.optimizer, self.train_dataloader,
                                self.lr_scheduler, self.noise_random_generator)
                print(f"[Checkpoint] Saving checkpoint at step {step} DONE.")
                self.transformer.train()
                self.sp_group.barrier()

        wandb.finish()
        save_checkpoint(self.transformer, self.global_rank,
                        self.training_args.output_dir,
                        self.training_args.max_train_steps, self.optimizer,
                        self.train_dataloader, self.lr_scheduler,
                        self.noise_random_generator)

        if get_sp_group():
            cleanup_dist_env_and_memory()

    @torch.no_grad()
    def _run_validation(self, training_batch: TrainingBatch, step: int) -> None:
        logger.info("Running validation at step %s", step)
        self.transformer.eval()

        vae = self.get_module("vae")

        # All ranks in the same SP group must run validation on identical inputs.
        sp_group_idx = self.global_rank // self.sp_world_size
        num_sp_groups = self.world_size // self.sp_world_size
        val_round = step // max(1, self.training_args.validation_steps)
        n_samples = len(self.validation_samples)
        _action_idx = sp_group_idx % 4
        val_idx = int((val_round * max(1, n_samples // max(1, num_sp_groups // 4)) + sp_group_idx // 4) % n_samples)
        val_data = self.validation_samples[val_idx]

        # Move validation data to device
        latents = val_data["latent"].to(self.device, dtype=torch.bfloat16)

        # Conditionally load image conditioning based on task type
        if self.task_type == "i2v":
            image_cond = val_data["image_cond"].to(self.device, dtype=torch.bfloat16)
            vision_states = val_data["vision_states"].to(self.device, dtype=torch.bfloat16)
        else:
            # For t2v, use zero tensors
            image_cond = torch.zeros_like(val_data["image_cond"]).to(self.device, dtype=torch.bfloat16)
            vision_states = torch.zeros_like(val_data["vision_states"]).to(self.device, dtype=torch.bfloat16)

        prompt_embed = val_data["prompt_embeds"].to(self.device, dtype=torch.bfloat16)
        prompt_mask = val_data["prompt_mask"].to(self.device, dtype=torch.bfloat16)
        byt5_text_states = val_data["byt5_text_states"].to(self.device, dtype=torch.bfloat16)
        byt5_text_mask = val_data["byt5_text_mask"].to(self.device, dtype=torch.bfloat16)

        val_noise_generator = torch.Generator(device=self.device).manual_seed(
            self.seed + val_idx
        )
        val_cpu_generator = torch.Generator(device="cpu").manual_seed(
            self.seed + val_idx
        )
        x = torch.randn(
            latents.shape,
            generator=val_noise_generator,
            device=self.device,
            dtype=latents.dtype,
        )
        assert self.task_type == "i2v", "the t2v is supported without testing now"
        multitask_mask = self.get_task_mask(self.task_type, x.shape[2]).to(self.device)
        cond_latents = self._prepare_cond_latents(self.task_type, image_cond, x, multitask_mask)
        validation_clean_x = None
        validation_aug_timesteps = None
        if self.training_args.use_teacher_forcing:
            if self.training_args.noise_augmentation_max_timestep > 0:
                aug_u = compute_density_for_timestep_sampling(
                    weighting_scheme="uniform",
                    batch_size=latents.shape[0] * latents.shape[2],
                    generator=val_cpu_generator,
                )
                aug_indices = (
                    aug_u * self.training_args.noise_augmentation_max_timestep
                ).long()
                aug_indices = aug_indices.clamp(
                    0, self.noise_scheduler.config.num_train_timesteps - 1
                )
                validation_aug_timesteps = self.noise_scheduler.timesteps[
                    aug_indices
                ].to(device=self.device, dtype=torch.bfloat16)
                aug_sigmas = get_sigmas(
                    self.noise_scheduler,
                    latents.device,
                    validation_aug_timesteps,
                    n_dim=latents.ndim,
                    dtype=latents.dtype,
                )
                aug_sigmas = rearrange(
                    aug_sigmas,
                    "(B D) C T H W -> B C (D T) H W",
                    D=latents.shape[2],
                )
                validation_clean_latents = (
                    1.0 - aug_sigmas
                ) * latents + aug_sigmas * x
            else:
                validation_clean_latents = latents
                validation_aug_timesteps = torch.zeros(
                    latents.shape[0] * latents.shape[2],
                    device=self.device,
                    dtype=torch.bfloat16,
                )
            clean_cond_latents = self._prepare_cond_latents(
                self.task_type,
                image_cond,
                validation_clean_latents,
                multitask_mask,
            )
            validation_clean_x = torch.concat(
                [validation_clean_latents, clean_cond_latents], dim=1
            )

        validation_num_steps = 20
        if self.training_args.validation_sampling_steps:
            try:
                parsed_steps = [
                    int(step_str.strip())
                    for step_str in self.training_args.validation_sampling_steps.split(",")
                    if step_str.strip()
                ]
                parsed_steps = [step for step in parsed_steps if step > 0]
                if parsed_steps:
                    validation_num_steps = parsed_steps[0]
            except ValueError:
                logger.warning(
                    "Invalid validation_sampling_steps=%s, falling back to %d",
                    self.training_args.validation_sampling_steps,
                    validation_num_steps,
                )

        scheduler = LocalFlowMatchScheduler(shift=self.train_time_shift, reverse=True, solver="euler")
        scheduler.set_timesteps(validation_num_steps, device=self.device)

        # Build synthetic viewmats/Ks for validation: auto-select direction by val_idx
        from hyvideo.generate_custom_trajectory import generate_camera_trajectory_local
        import numpy as _np
        _B, _C, _T, _H, _W = latents.shape
        _step = 0.08
        _action_names = ["forward", "backward", "left", "right"]
        _action_map = [{"forward": _step}, {"forward": -_step}, {"right": -_step}, {"right": _step}]
        _action_name = _action_names[_action_idx]
        _move = _action_map[_action_idx]
        _poses_4x4 = generate_camera_trajectory_local([_move] * (_T - 1))  # c2w matrices
        # Invert c2w to w2c for PRoPE
        _viewmats = _np.zeros((_T, 4, 4), dtype=_np.float32)
        for _i, P in enumerate(_poses_4x4):
            R_c2w = P[:3, :3]
            t_c2w = P[:3, 3]
            _viewmats[_i, :3, :3] = R_c2w.T
            _viewmats[_i, :3, 3] = -R_c2w.T @ t_c2w
            _viewmats[_i, 3, 3] = 1.0
        _viewmats = torch.from_numpy(_viewmats).unsqueeze(0).to(self.device, dtype=torch.bfloat16)  # (1, T, 4, 4)
        # Default intrinsics (normalized): fx/(cx*2), fy/(cy*2) with raw fx=fy=969.697, cx=960, cy=540
        _fx_norm = 969.6969696969696 / (960.0 * 2)   # 0.50505...
        _fy_norm = 969.6969696969696 / (540.0 * 2)   # 0.89787...
        _K = _np.array([[_fx_norm, 0, 0.5], [0, _fy_norm, 0.5], [0, 0, 1]], dtype=_np.float32)
        _Ks = _np.tile(_K, (_T, 1, 1))
        _Ks = torch.from_numpy(_Ks).unsqueeze(0).to(self.device, dtype=torch.bfloat16)  # (1, T, 3, 3)

        # Discrete action for validation (if model has action_in)
        _action = None
        if hasattr(self.transformer, 'action_in'):
            _action = torch.from_numpy(
                discretize_poses_to_actions(_viewmats[0].cpu().float().numpy())
            ).unsqueeze(0).to(self.device, dtype=torch.bfloat16)  # (1, T)

        # assert validation_clean_x is None, "clean_x must not be passed during validation: teacher forcing is training-only"
        # assert validation_aug_timesteps is None, "aug_timesteps must not be passed during validation: teacher forcing is training-only"
        for i, t in enumerate(scheduler.timesteps):
            logger.info("Validation denoising step %d/%d", i + 1, validation_num_steps)
            timesteps_in = t.unsqueeze(0).expand(x.shape[0] * x.shape[2]).to(self.device, dtype=torch.bfloat16)
            extra_kwargs = {"byt5_text_states": byt5_text_states, "byt5_text_mask": byt5_text_mask}
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                pred = self.transformer(
                    hidden_states=torch.concat([x, cond_latents], dim=1),
                    timestep=timesteps_in,
                    timestep_txt=torch.tensor(0).unsqueeze(0).to(self.device, dtype=torch.bfloat16),
                    text_states=prompt_embed,
                    text_states_2=None,
                    encoder_attention_mask=prompt_mask,
                    timestep_r=None,
                    vision_states=vision_states,
                    mask_type=self.task_type,
                    guidance=None,
                    extra_kwargs=extra_kwargs,
                    return_dict=False,
                    clean_x=validation_clean_x,
                    aug_timesteps=validation_aug_timesteps,
                    viewmats=_viewmats,
                    Ks=_Ks,
                    action=_action,
                )[0]
            x = scheduler.step(pred, t, x).prev_sample

        # VAE decode may use SP collectives when sp_size > 1, so every rank in
        # the SP group must participate in decode. Only the group leader saves.
        vae = self.get_module("vae").to(self.device)
        scaling_factor = vae.config.scaling_factor
        shift_factor = getattr(vae.config, "shift_factor", None)
        if shift_factor:
            x_decoded = x / scaling_factor + shift_factor
        else:
            x_decoded = x / scaling_factor

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            video = vae.decode(x_decoded).sample  # [B, C, T, H, W]

        if self.rank_in_sp_group == 0:
            video = (video.float().clamp(-1, 1) + 1) / 2
            frames = rearrange(video[0], "c t h w -> t h w c")
            frames = (frames.cpu().numpy() * 255).astype(np.uint8)

            validation_dir = os.path.join(
                self.training_args.output_dir,
                "validation",
                f"step_{step:07d}",
            )
            os.makedirs(validation_dir, exist_ok=True)
            filename = os.path.join(
                validation_dir,
                f"sample_{val_idx:02d}_spgroup_{sp_group_idx:02d}_{_action_name}.mp4",
            )
            imageio.mimsave(filename, frames, fps=8)
            logger.info("Saved validation video to %s", filename)
            del frames

        vae.cpu()
        del vae, video, x_decoded
        torch.cuda.empty_cache()
        self.sp_group.barrier()

    def _log_training_info(self) -> None:
        total_batch_size = (self.world_size *
                            self.training_args.gradient_accumulation_steps /
                            self.training_args.sp_size *
                            self.training_args.train_sp_batch_size)
        logger.info("***** Running training *****")
        logger.info("  Num examples = %s", len(self.train_dataset))
        logger.info("  Dataloader size = %s", len(self.train_dataloader))
        logger.info("  Num Epochs = %s", self.num_train_epochs)
        logger.info("  Resume training from step %s",
                    self.init_steps)  # type: ignore
        logger.info("  Instantaneous batch size per device = %s",
                    self.training_args.train_batch_size)
        logger.info(
            "  Total train batch size (w. data & sequence parallel, accumulation) = %s",
            total_batch_size)
        logger.info("  Gradient Accumulation steps = %s",
                    self.training_args.gradient_accumulation_steps)
        logger.info("  Total optimization steps = %s",
                    self.training_args.max_train_steps)
        logger.info("  Total training parameters per FSDP shard = %s B",
                    round(_get_trainable_params(self.transformer) / 1e9, 3))
        # print dtype
        logger.info("  Master weight dtype: %s",
                    self.transformer.parameters().__next__().dtype)

        gpu_memory_usage = torch.cuda.memory_allocated() / 1024**2
        logger.info("GPU memory usage before train_one_step: %s MB",
                    gpu_memory_usage)
        logger.info("VSA validation sparsity: %s",
                    self.training_args.VSA_sparsity)

    def _prepare_validation_batch(self, sampling_param: SamplingParam,
                                  training_args: TrainingArgs,
                                  validation_batch: dict[str, Any],
                                  num_inference_steps: int) -> ForwardBatch:
        sampling_param.prompt = validation_batch['prompt']
        sampling_param.height = training_args.num_height
        sampling_param.width = training_args.num_width
        sampling_param.num_inference_steps = num_inference_steps
        sampling_param.data_type = "video"
        assert self.seed is not None
        sampling_param.seed = self.seed

        latents_size = [(sampling_param.num_frames - 1) // 4 + 1,
                        sampling_param.height // 8, sampling_param.width // 8]
        n_tokens = latents_size[0] * latents_size[1] * latents_size[2]
        temporal_compression_factor = training_args.pipeline_config.vae_config.arch_config.temporal_compression_ratio
        num_frames = (training_args.num_latent_t -
                      1) * temporal_compression_factor + 1
        sampling_param.num_frames = num_frames
        batch = ForwardBatch(
            **shallow_asdict(sampling_param),
            latents=None,
            generator=self.validation_random_generator,
            n_tokens=n_tokens,
            eta=0.0,
            VSA_sparsity=training_args.VSA_sparsity,
        )

        return batch

    @torch.no_grad()
    def _log_validation(self, transformer, training_args, global_step) -> None:
        """
        Generate a validation video and log it to wandb to check the quality during training.
        """
        raise NotImplementedError("Training pipelines must implement this method")


def main(args) -> None:
    logger.info("Starting training pipeline...")

    pipeline = ARCameraTrainingPipeline.from_pretrained(
        args.pretrained_model_name_or_path, args=args)
    pipeline.train()
    logger.info("Training pipeline done")

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.set_start_method('spawn', force=True)

    from trainer.trainer_args import TrainingArgs
    from trainer.utils import FlexibleArgumentParser
    parser = FlexibleArgumentParser()
    parser = TrainingArgs.add_cli_args(parser)
    parser = TrainerArgs.add_cli_args(parser)
    args = parser.parse_args()
    args.dit_cpu_offload = False
    main(args)
