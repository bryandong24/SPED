# SPDX-License-Identifier: Apache-2.0
"""
Consistency Distillation (CD) training pipeline for causal AR models.

Algorithm (online teacher, no pre-computed ODE trajectory):
    1. Sample timestep index n  → t, t_next from scheduler
    2. Add noise online: latent_t = scheduler.add_noise(clean, noise, t)
    3. Teacher CFG forward (frozen, no grad) + Euler step → latent_t_next
    4. Student forward at (latent_t, t)       → cm_pred_t   (pred_x0)
    5. EMA student forward at (latent_t_next, t_next) → cm_pred_t_next (pred_x0)
    6. loss = MSE(cm_pred_t, cm_pred_t_next)
"""
import dataclasses
import math
import os
import random
import time
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Iterator
from typing import Any

import imageio
import numpy as np
import torch
import torch.nn.functional as F
import torch.distributed as dist
from hyvideo.schedulers.scheduling_flow_match_discrete import FlowMatchDiscreteScheduler
from einops import rearrange
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm.auto import tqdm

from trainer.distributed.parallel_state import get_sp_parallel_rank, get_sp_world_size
import trainer.envs as envs
from trainer.dataset import build_hunyuan_w_mem_dataloader
from trainer.dataset.dataloader.schema import pyarrow_schema_t2v, pyarrow_schema_i2v
from trainer.distributed import (cleanup_dist_env_and_memory, get_local_torch_device,
                                  get_sp_group, get_world_group)
from trainer.trainer_args import TrainerArgs, TrainingArgs, WorkloadType
from trainer.forward_context import set_forward_context
from trainer.logger import init_logger
from trainer.pipelines import ComposedPipelineBase, ForwardBatch, TrainingBatch
from trainer.training.activation_checkpoint import apply_activation_checkpointing
from trainer.training.training_utils import (
    clip_grad_norm_while_handling_failing_dtensor_cases,
    get_scheduler, load_checkpoint, save_checkpoint,
)
from trainer.utils import set_random_seed
from trainer.training.muon import get_muon_optimizer
from trainer.training.ema import EMA

import wandb  # isort: skip

from algorithms.flow_matching import pred_x0_from_flow, add_flow_noise
from algorithms.consistency_distillation import teacher_cfg_euler_step, consistency_loss

logger = init_logger(__name__)


def _get_trainable_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def merge_tensor_by_mask(tensor_1, tensor_2, mask, dim):
    assert tensor_1.shape == tensor_2.shape
    masked_indices = torch.nonzero(mask).squeeze(1)
    tmp = tensor_1.clone()
    if dim == 0:
        tmp[masked_indices] = tensor_2[masked_indices]
    elif dim == 1:
        tmp[:, masked_indices] = tensor_2[:, masked_indices]
    elif dim == 2:
        tmp[:, :, masked_indices] = tensor_2[:, :, masked_indices]
    return tmp


class ConsistencyDistillationPipeline(ComposedPipelineBase, ABC):
    _required_config_modules = ["scheduler", "transformer", "vae"]
    train_dataloader: StatefulDataLoader
    train_loader_iter: Iterator[dict[str, Any]]
    current_epoch: int = 0

    def __init__(self, model_path, trainer_args, required_config_modules=None, loaded_modules=None):
        trainer_args.inference_mode = False
        self.lora_training = trainer_args.lora_training
        if self.lora_training and trainer_args.lora_rank is None:
            raise ValueError("lora rank must be set when using lora training")
        set_random_seed(trainer_args.seed)
        super().__init__(model_path, trainer_args, required_config_modules, loaded_modules)

    def create_pipeline_stages(self, trainer_args: TrainerArgs):
        raise RuntimeError("create_pipeline_stages should not be called for training pipeline")

    def set_schemas(self):
        if self.training_args.workload_type == WorkloadType.I2V:
            self.train_dataset_schema = pyarrow_schema_i2v
        else:
            self.train_dataset_schema = pyarrow_schema_t2v

    def initialize_training_pipeline(self, training_args: TrainingArgs):
        logger.info("Initializing CD training pipeline...")
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
        self._is_wan_model = type(self.transformer).__name__ == "CausalWanModel"
        self.seed = training_args.seed
        self.task_type = training_args.workload_type.value
        self.set_schemas()
        self.causal = training_args.causal
        self.training_args.use_teacher_forcing = self.causal
        self.cfg_scale = getattr(training_args, 'cfg_scale', 5.0)
        assert training_args.training_cfg_rate == 0.0, \
            "Distillation student must see real prompts; set --training_cfg_rate 0.0"

        assert self.seed is not None
        set_random_seed(self.seed)
        self.transformer.train()

        if training_args.enable_gradient_checkpointing_type is not None:
            self.transformer = apply_activation_checkpointing(
                self.transformer,
                checkpointing_type=training_args.enable_gradient_checkpointing_type)

        self.set_trainable()
        self.optimizer = get_muon_optimizer(
            model=self.transformer,
            lr=training_args.learning_rate,
            weight_decay=training_args.weight_decay,
            adamw_betas=(0.9, 0.999),
            adamw_eps=1e-8,
        )
        self.init_steps = 0
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

        self.train_dataset, self.train_dataloader = build_hunyuan_w_mem_dataloader(
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
            task_type=self.task_type,
        )

        self.num_update_steps_per_epoch = math.ceil(
            len(self.train_dataloader) /
            training_args.gradient_accumulation_steps * training_args.sp_size /
            training_args.train_sp_batch_size)
        self.num_train_epochs = math.ceil(
            training_args.max_train_steps / self.num_update_steps_per_epoch)
        self.current_epoch = 0

        # Load frozen teacher (same arch as student, separate weights)
        self._load_teacher()

        # EMA tracker for student → used as target network
        self.ema = EMA(self.transformer, decay=0.999, mode="local_shard")
        logger.info("Initialized EMA target network")

        # Load neg prompt embeddings for CFG — keep on GPU to avoid per-step transfer
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

        if self.global_rank == 0:
            project = training_args.tracker_project_name or "trainer"
            wandb.login(key=training_args.wandb_key)
            wandb.init(
                config=dataclasses.asdict(training_args),
                name=training_args.wandb_run_name,
                entity=training_args.wandb_entity,
                project=project,
            )

        self.get_module("vae").cpu()

        if training_args.log_validation:
            import json
            with open(training_args.json_path) as f:
                index = json.load(f)
            self.validation_samples = []
            for i in range(min(8, len(index))):
                data = torch.load(index[i]["latent_path"], map_location="cpu", weights_only=True)
                # TODO: 大谬。没有考虑 window_frame, 遇到 21 frames 这种会直接全 load, 导致不能被 chunsize 整除，评测有问题
                # 分析: 训练dataloader有 //4*4 对齐+window_frames裁剪(hunyuan_w_mem_dataset.py:177-181), 但此处缺失
                # 后果: 21帧→torch_causal chunk_num=21//4=5, 第21帧token成孤岛(只attend text, 无vision self-attn), 末尾生成崩坏
                # 修复: 加载后做 t_crop = min(window_frames, t // 4 * 4)
                self.validation_samples.append(data)
            logger.info("Loaded %d fixed validation samples", len(self.validation_samples))

    def _load_teacher(self):
        """Load frozen teacher — always same checkpoint as student."""
        import glob
        from trainer.models.loader.fsdp_load import maybe_load_fsdp_model
        from trainer.models.registry import ModelRegistry

        load_dir = self.trainer_args.load_from_dir
        ar_action_dir = self.training_args.ar_action_load_from_dir
        if not load_dir:
            raise ValueError("load_from_dir is not set; cannot load teacher")

        cls_name = self.trainer_args.cls_name
        model_cls, _ = ModelRegistry.resolve_model_cls(cls_name)
        effective_dir = os.path.dirname(ar_action_dir) if ar_action_dir else load_dir
        safetensors_list = glob.glob(os.path.join(str(effective_dir), "*.safetensors"))
        if not safetensors_list:
            raise ValueError(f"No safetensors files found in {effective_dir}")

        self.teacher = maybe_load_fsdp_model(
            load_from_dir=load_dir,
            ar_action_load_from_dir=ar_action_dir,
            cls_name=cls_name,
            model_cls=model_cls,
            init_params={},
            weight_dir_list=safetensors_list,
            device=self.device,
            hsdp_replicate_dim=self.trainer_args.hsdp_replicate_dim,
            hsdp_shard_dim=self.trainer_args.hsdp_shard_dim,
            cpu_offload=False,
            pin_cpu_memory=False,
            fsdp_inference=True,
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            output_dtype=None,
            training_mode=False,
        )
        logger.info("Teacher loaded from same checkpoint as student: %s", effective_dir)

        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad_(False)
        logger.info("Teacher frozen on GPU (bf16)")

    @abstractmethod
    def initialize_validation_pipeline(self, training_args: TrainingArgs):
        raise NotImplementedError

    # ──────────────────────────────────────────────
    #  Data loading
    # ──────────────────────────────────────────────

    def _prepare_training(self, training_batch: TrainingBatch) -> TrainingBatch:
        self.transformer.train()
        self.optimizer.zero_grad()
        training_batch.total_loss = 0.0
        return training_batch

    def _get_next_batch(self, training_batch: TrainingBatch) -> TrainingBatch:
        batch = next(self.train_loader_iter, None)
        if batch is None:
            self.current_epoch += 1
            logger.info("Starting epoch %s", self.current_epoch)
            self.train_loader_iter = iter(self.train_dataloader)
            batch = next(self.train_loader_iter)

        dev = get_local_torch_device()
        bf = torch.bfloat16

        if self.global_rank == 0 and training_batch.current_timestep == 1:
            logger.info("First batch shapes: " + ", ".join(
                f"{k}: {v.shape}" for k, v in batch.items() if hasattr(v, 'shape')))

        training_batch.latents = batch["latent"].to(dev, dtype=bf)
        training_batch.prompt_embed = batch["prompt_embed"].to(dev, dtype=bf)
        training_batch.video_path = batch.get('video_path', batch.get('path'))
        if isinstance(training_batch.video_path, list):
            training_batch.video_path = training_batch.video_path[0]
        training_batch.image_cond = batch.get('image_cond').to(dev, dtype=bf)
        training_batch.vision_states = batch.get('vision_states').to(dev, dtype=bf)
        training_batch.prompt_mask = batch.get('prompt_mask').to(dev, dtype=bf)
        training_batch.byt5_text_states = batch.get('byt5_text_states').to(dev, dtype=bf)
        training_batch.byt5_text_mask = batch.get('byt5_text_mask').to(dev, dtype=bf)
        swof = batch.get('select_window_out_flag', 0)
        training_batch.select_window_out_flag = swof[0] if isinstance(swof, (list, torch.Tensor)) else swof
        training_batch.i2v_mask = batch.get('i2v_mask').to(dev, dtype=bf)
        return training_batch

    # ──────────────────────────────────────────────
    #  Helpers
    # ──────────────────────────────────────────────

    def pred_x0(self, sample, model_pred, sigmas):
        return pred_x0_from_flow(noisy=sample, flow_pred=model_pred, sigma=sigmas, compute_dtype=sample.dtype)

    def _prepare_cond_latents(self, task_type, cond_latents, latents, multitask_mask):
        if cond_latents is not None and task_type == 'i2v':
            latents_concat = cond_latents.repeat(1, 1, latents.shape[2], 1, 1)
            latents_concat[:, :, 1:, :, :] = 0.0
        else:
            latents_concat = torch.zeros_like(latents)
        mask_zeros = torch.zeros(latents.shape[0], 1, latents.shape[2], latents.shape[3], latents.shape[4])
        mask_ones = torch.ones_like(mask_zeros)
        mask_concat = merge_tensor_by_mask(mask_zeros.cpu(), mask_ones.cpu(),
                                           mask=multitask_mask.cpu(), dim=2).to(latents.device)
        return torch.concat([latents_concat, mask_concat], dim=1)

    def get_task_mask(self, task_type, latent_target_length):
        mask = torch.zeros(latent_target_length)
        if task_type == "i2v":
            mask[0] = 1.0
        elif task_type != "t2v":
            raise ValueError(f"{task_type} is not supported!")
        return mask

    def _build_transformer_kwargs(self, noisy, timesteps, training_batch, clean_x=None, aug_timesteps=None):
        if self._is_wan_model:
            return self._build_wan_kwargs(noisy, timesteps, training_batch, clean_x, aug_timesteps)
        multitask_mask = self.get_task_mask(self.task_type, noisy.shape[2]).to(self.device)
        cond = self._prepare_cond_latents(self.task_type, training_batch.image_cond, noisy, multitask_mask)
        return {
            "hidden_states": torch.concat([noisy, cond], dim=1),
            "timestep": timesteps.to(self.device, dtype=torch.bfloat16),
            "timestep_txt": torch.tensor(0).unsqueeze(0).to(self.device, dtype=torch.bfloat16),
            "text_states": training_batch.prompt_embed,
            "text_states_2": None,
            "encoder_attention_mask": training_batch.prompt_mask,
            "timestep_r": None,
            "vision_states": training_batch.vision_states,
            "mask_type": self.task_type,
            "guidance": None,
            "extra_kwargs": {"byt5_text_states": training_batch.byt5_text_states,
                             "byt5_text_mask": training_batch.byt5_text_mask},
            "return_dict": False,
            "clean_x": clean_x,
            "aug_timesteps": aug_timesteps,
        }

    def _build_wan_kwargs(self, noisy, timesteps, training_batch, clean_x=None, aug_timesteps=None):
        """Build kwargs for CausalWanModel forward.

        noisy: [B, C, T, H, W]
        timesteps: [B*T] flat
        """
        B, C, T, H, W = noisy.shape
        # CausalWanModel expects x as [B, C, F, H, W] (iterable over batch)
        x = noisy  # already [B, C, T, H, W]
        # t: [B, F] per-frame timesteps
        t = timesteps.view(B, T)
        # context: list of [L, D] tensors
        context = list(training_batch.prompt_embed)
        # seq_len: max sequence length for positional encoding
        patch_size = self.transformer.patch_size if hasattr(self.transformer, 'patch_size') else (1, 2, 2)
        seq_len = T * (H // patch_size[1]) * (W // patch_size[2])
        kwargs = {
            "x": x,
            "t": t,
            "context": context,
            "seq_len": seq_len,
        }
        if clean_x is not None:
            kwargs["clean_x"] = clean_x
        if aug_timesteps is not None:
            kwargs["aug_t"] = aug_timesteps.view(B, T)
        return kwargs

    def _build_uncond_kwargs(self, noisy, timesteps, training_batch, clean_x=None, aug_timesteps=None):
        if self._is_wan_model:
            return self._build_wan_uncond_kwargs(noisy, timesteps, training_batch, clean_x, aug_timesteps)
        dev, bf = self.device, torch.bfloat16
        B = noisy.shape[0]
        if self.neg_prompt_pt is not None:
            neg_embed = self.neg_prompt_pt['negative_prompt_embeds'][0].to(dev, dtype=bf).unsqueeze(0).expand(B, -1, -1)
            neg_mask = self.neg_prompt_pt['negative_prompt_mask'][0].to(dev, dtype=bf).unsqueeze(0).expand(B, -1)
            neg_byt5 = self.neg_byt5_pt['byt5_text_states'][0].to(dev, dtype=bf).unsqueeze(0).expand(B, -1, -1)
            neg_byt5_mask = self.neg_byt5_pt['byt5_text_mask'][0].to(dev, dtype=bf).unsqueeze(0).expand(B, -1)
        else:
            neg_embed = torch.zeros_like(training_batch.prompt_embed)
            neg_mask = training_batch.prompt_mask
            neg_byt5 = torch.zeros_like(training_batch.byt5_text_states)
            neg_byt5_mask = training_batch.byt5_text_mask
        multitask_mask = self.get_task_mask(self.task_type, noisy.shape[2]).to(dev)
        cond = self._prepare_cond_latents(self.task_type, training_batch.image_cond, noisy, multitask_mask)
        return {
            "hidden_states": torch.concat([noisy, cond], dim=1),
            "timestep": timesteps.to(dev, dtype=bf),
            "timestep_txt": torch.tensor(0).unsqueeze(0).to(dev, dtype=bf),
            "text_states": neg_embed,
            "text_states_2": None,
            "encoder_attention_mask": neg_mask,
            "timestep_r": None,
            "vision_states": training_batch.vision_states,
            "mask_type": self.task_type,
            "guidance": None,
            "extra_kwargs": {"byt5_text_states": neg_byt5, "byt5_text_mask": neg_byt5_mask},
            "return_dict": False,
            "clean_x": clean_x,
            "aug_timesteps": aug_timesteps,
        }

    def _build_wan_uncond_kwargs(self, noisy, timesteps, training_batch, clean_x=None, aug_timesteps=None):
        """Build unconditional kwargs for CausalWanModel (CFG negative)."""
        dev, bf = self.device, torch.bfloat16
        B = noisy.shape[0]
        if self.neg_prompt_pt is not None:
            neg_embed = self.neg_prompt_pt['negative_prompt_embeds'][0].to(dev, dtype=bf).unsqueeze(0).expand(B, -1, -1)
        else:
            neg_embed = torch.zeros_like(training_batch.prompt_embed)
        # Create a shallow copy of training_batch with swapped prompt_embed
        orig_embed = training_batch.prompt_embed
        training_batch.prompt_embed = neg_embed
        kwargs = self._build_wan_kwargs(noisy, timesteps, training_batch, clean_x, aug_timesteps)
        training_batch.prompt_embed = orig_embed
        return kwargs

    # ──────────────────────────────────────────────
    #  Core CD step
    # ──────────────────────────────────────────────

    def _prepare_cd_inputs(self, training_batch: TrainingBatch) -> TrainingBatch:
        clean_latent = training_batch.latents  # [B, C, T, H, W]
        B, latent_t = clean_latent.shape[0], clean_latent.shape[2]
        dev, bf = self.device, torch.bfloat16

        # Sample random adjacent pair (i, i+1) from inference schedule
        N = len(self.noise_scheduler.timesteps)
        i = random.randrange(N)

        sigma_t = self.noise_scheduler.sigmas[i]
        sigma_t_next = self.noise_scheduler.sigmas[i + 1]
        t_val = self.noise_scheduler.timesteps[i]
        t_next_val = self.noise_scheduler.timesteps[i + 1] if i + 1 < N else torch.tensor(0.0, device=dev, dtype=bf)

        # Reshape sigmas for broadcasting: [1, 1, 1, 1, 1]
        sigma_shape = [1] * clean_latent.ndim
        sigmas = sigma_t.to(dev, dtype=bf).view(*sigma_shape).expand_as(clean_latent)
        sigmas_next = sigma_t_next.to(dev, dtype=bf).view(*sigma_shape).expand_as(clean_latent)

        timesteps_flat = t_val.to(dev, dtype=bf).expand(B * latent_t).contiguous()
        timesteps_next_flat = t_next_val.to(dev, dtype=bf).expand(B * latent_t).contiguous()

        if self.training_args.sp_size > 1:
            sp_group = get_sp_group()
            sp_group.broadcast(timesteps_flat, src=0)
            sp_group.broadcast(timesteps_next_flat, src=0)

        # Online noise: x_t = (1 - σ) * clean + σ * noise
        noise = torch.randn(clean_latent.shape, generator=self.noise_gen_cuda, device=dev, dtype=bf)
        latent_t_noisy = add_flow_noise(clean_latent, noise, sigmas)

        # Teacher-forcing clean context
        clean_x_concat, aug_ts = None, None
        if self.training_args.use_teacher_forcing:
            if self._is_wan_model:
                # CausalWanModel handles clean_x internally (no channel concat)
                clean_x_concat = clean_latent
            else:
                multitask_mask = self.get_task_mask(self.task_type, latent_t).to(dev)
                clean_cond = self._prepare_cond_latents(self.task_type, training_batch.image_cond,
                                                        clean_latent, multitask_mask)
                clean_x_concat = torch.concat([clean_latent, clean_cond], dim=1)
            aug_ts = torch.zeros(B * latent_t, device=dev, dtype=bf)

        # Teacher CFG forward + Euler step → latent_t_next
        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=bf):
                v_cond = self._call_model(self.teacher, **self._build_transformer_kwargs(
                    latent_t_noisy, timesteps_flat, training_batch, clean_x_concat, aug_ts))
                v_uncond = self._call_model(self.teacher, **self._build_uncond_kwargs(
                    latent_t_noisy, timesteps_flat, training_batch, clean_x_concat, aug_ts))

            latent_t_next = teacher_cfg_euler_step(
                v_cond=v_cond,
                v_uncond=v_uncond,
                latent_t=latent_t_noisy,
                t=sigmas,
                t_next=sigmas_next,
                guidance_scale=self.cfg_scale,
                timestep_scale=1.0,  # HY15 uses sigma directly, no /1000
            )

        logger.debug("CD step %d/%d: σ=%.4f→%.4f, t=%.1f→%.1f",
                     i, N, sigma_t.item(), sigma_t_next.item(),
                     t_val.item(), t_next_val.item())

        training_batch.noisy_model_input = latent_t_noisy
        training_batch.timesteps = timesteps_flat
        training_batch.sigmas = sigmas
        training_batch.noise = noise
        training_batch.raw_latent_shape = clean_latent.shape

        # Store CD-specific state for _build_input_kwargs / loss computation
        self._cd_target_noisy = latent_t_next
        self._cd_target_timesteps = timesteps_next_flat
        self._cd_target_sigmas = sigmas_next
        self._cd_clean_x_concat = clean_x_concat
        self._cd_aug_ts = aug_ts
        return training_batch

    def _build_input_kwargs(self, training_batch: TrainingBatch) -> TrainingBatch:
        training_batch.input_kwargs = self._build_transformer_kwargs(
            training_batch.noisy_model_input, training_batch.timesteps,
            training_batch, self._cd_clean_x_concat, self._cd_aug_ts)
        training_batch._target_input_kwargs = self._build_transformer_kwargs(
            self._cd_target_noisy, self._cd_target_timesteps,
            training_batch, self._cd_clean_x_concat, self._cd_aug_ts)
        return training_batch

    def _call_model(self, model, **kwargs):
        """Call a model (student or teacher) and return the flow/velocity prediction tensor."""
        out = model(**kwargs)
        if isinstance(out, (tuple, list)):
            return out[0]
        return out

    def _call_transformer(self, **kwargs):
        """Call the student transformer."""
        return self._call_model(self.transformer, **kwargs)

    def _transformer_forward_and_compute_loss(self, training_batch: TrainingBatch) -> TrainingBatch:
        with set_forward_context(current_timestep=training_batch.current_timestep,
                                 attn_metadata=training_batch.attn_metadata):
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                v_student = self._call_transformer(**training_batch.input_kwargs)
                cm_pred_t = self.pred_x0(training_batch.noisy_model_input, v_student,
                                         training_batch.sigmas)
                with torch.no_grad():
                    with self.ema.apply_to_model(self.transformer):
                        v_target = self._call_transformer(**training_batch._target_input_kwargs)
                    cm_pred_t_next = self.pred_x0(self._cd_target_noisy, v_target,
                                                  self._cd_target_sigmas)

            loss = consistency_loss(cm_pred_t, cm_pred_t_next, reduction="mean")
            loss = loss / self.training_args.gradient_accumulation_steps
            loss.backward()
            avg_loss = loss.detach().clone()

        dist.all_reduce(avg_loss, op=dist.ReduceOp.MAX)
        training_batch.total_loss += avg_loss.item()
        return training_batch

    def _clip_grad_norm(self, training_batch: TrainingBatch) -> TrainingBatch:
        max_grad_norm = self.training_args.max_grad_norm
        if max_grad_norm is not None:
            grad_norm = clip_grad_norm_while_handling_failing_dtensor_cases(
                list(self.transformer.parameters()), max_grad_norm, foreach=None)
            grad_norm = grad_norm.item() if grad_norm is not None else 0.0
        else:
            grad_norm = 0.0
        training_batch.grad_norm = grad_norm
        return training_batch

    def _get_ema_decay(self, step: int) -> float:
        return 0.999

    def train_one_step(self, training_batch: TrainingBatch) -> TrainingBatch:
        training_batch = self._prepare_training(training_batch)
        for _ in range(self.training_args.gradient_accumulation_steps):
            training_batch = self._get_next_batch(training_batch)
            training_batch = self._prepare_cd_inputs(training_batch)
            training_batch = self._build_input_kwargs(training_batch)
            training_batch = self._transformer_forward_and_compute_loss(training_batch)
        training_batch = self._clip_grad_norm(training_batch)
        grad_norm = torch.tensor(training_batch.grad_norm).to(get_local_torch_device())
        dist.all_reduce(grad_norm, op=dist.ReduceOp.MAX)
        training_batch.grad_norm = grad_norm.item()
        if training_batch.grad_norm < 10.0:
            self.optimizer.step()
            self.lr_scheduler.step()
        self.ema.decay = self._get_ema_decay(training_batch.current_timestep)
        self.ema.update(self.transformer)
        return training_batch

    # ──────────────────────────────────────────────
    #  Training loop
    # ──────────────────────────────────────────────

    def _resume_from_checkpoint(self):
        logger.info("Loading checkpoint from %s", self.training_args.resume_from_checkpoint)
        resumed_step = load_checkpoint(
            self.transformer, self.global_rank,
            self.training_args.resume_from_checkpoint, self.optimizer,
            self.train_dataloader, self.lr_scheduler, self.noise_random_generator)
        self.init_steps = resumed_step if resumed_step > 0 else 0
        logger.info("Resumed from step %s", self.init_steps)

    def train(self):
        assert self.seed is not None
        set_random_seed(self.seed + self.global_rank)
        if not self.post_init_called:
            self.post_init()
        self.noise_random_generator = torch.Generator(device="cpu").manual_seed(self.seed)
        self.noise_gen_cuda = torch.Generator(device="cuda").manual_seed(self.seed)
        self.validation_random_generator = torch.Generator(device="cpu").manual_seed(self.seed)
        # Build CD timestep schedule: same as inference with N steps + shift
        cd_num_steps = self.training_args.cd_num_steps
        cd_shift = self.training_args.ode_shift
        self.noise_scheduler = FlowMatchDiscreteScheduler(shift=cd_shift, reverse=True, solver="euler")
        self.noise_scheduler.set_timesteps(cd_num_steps, device=self.device)
        # scheduler.sigmas: (cd_num_steps+1,) from σ_0=1.0 down to σ_N=0.0 (after shift)
        # scheduler.timesteps: (cd_num_steps,) = sigmas[:-1] * 1000
        logger.info("CD schedule: %d steps, shift=%.1f, sigmas range [%.4f, %.4f]",
                     cd_num_steps, cd_shift,
                     self.noise_scheduler.sigmas[0].item(),
                     self.noise_scheduler.sigmas[-1].item())
        if self.training_args.resume_from_checkpoint:
            self._resume_from_checkpoint()
        self.train_loader_iter = iter(self.train_dataloader)
        step_times: deque[float] = deque(maxlen=100)
        self._log_training_info()
        progress_bar = tqdm(range(0, self.training_args.max_train_steps),
                            initial=self.init_steps, desc="Steps", disable=self.local_rank > 0)
        for step in range(self.init_steps + 1, self.training_args.max_train_steps + 1):
            self.train_dataset.update_max_frames(step)
            start_time = time.perf_counter()
            training_batch = TrainingBatch()
            training_batch.current_timestep = step
            training_batch = self.train_one_step(training_batch)
            step_time = time.perf_counter() - start_time
            step_times.append(step_time)
            progress_bar.set_postfix({
                "loss": f"{training_batch.total_loss:.4f}",
                "step_time": f"{step_time:.2f}s",
                "grad_norm": training_batch.grad_norm,
            })
            progress_bar.update(1)
            if self.global_rank == 0:
                wandb.log({
                    "train_loss": training_batch.total_loss,
                    "learning_rate": self.lr_scheduler.get_last_lr()[0],
                    "step_time": step_time,
                    "avg_step_time": sum(step_times) / len(step_times),
                    "grad_norm": training_batch.grad_norm,
                    "ema_decay": self.ema.decay,
                }, step=step)
            if self.training_args.log_validation and step % self.training_args.validation_steps == 0:
                self._run_validation(training_batch, step)
                self.transformer.train()
            if step % self.training_args.checkpointing_steps == 0:
                with self.ema.apply_to_model(self.transformer):
                    save_checkpoint(self.transformer, self.global_rank,
                                    self.training_args.output_dir, step,
                                    self.optimizer, self.train_dataloader,
                                    self.lr_scheduler, self.noise_random_generator)
                self.transformer.train()
                self.sp_group.barrier()
        wandb.finish()
        with self.ema.apply_to_model(self.transformer):
            save_checkpoint(self.transformer, self.global_rank,
                            self.training_args.output_dir, self.training_args.max_train_steps,
                            self.optimizer, self.train_dataloader,
                            self.lr_scheduler, self.noise_random_generator)
        if get_sp_group():
            cleanup_dist_env_and_memory()

    @torch.no_grad()
    def _run_validation(self, training_batch: TrainingBatch, step: int):
        logger.info("Running validation at step %s", step)
        self.transformer.eval()
        sp_group_idx = self.global_rank // self.sp_world_size
        val_idx = sp_group_idx % len(self.validation_samples)
        val_data = self.validation_samples[val_idx]
        dev, bf = self.device, torch.bfloat16
        latents = val_data["latent"].to(dev, dtype=bf)
        image_cond = val_data["image_cond"].to(dev, dtype=bf)
        vision_states = val_data["vision_states"].to(dev, dtype=bf)
        prompt_embed = val_data["prompt_embeds"].to(dev, dtype=bf)
        prompt_mask = val_data["prompt_mask"].to(dev, dtype=bf)
        byt5_text_states = val_data["byt5_text_states"].to(dev, dtype=bf)
        byt5_text_mask = val_data["byt5_text_mask"].to(dev, dtype=bf)
        val_gen = torch.Generator(device=dev).manual_seed(self.seed + val_idx)
        x = torch.randn(latents.shape, generator=val_gen, device=dev, dtype=bf)
        multitask_mask = self.get_task_mask(self.task_type, x.shape[2]).to(dev)
        cond_latents = self._prepare_cond_latents(self.task_type, image_cond, x, multitask_mask)
        clean_x_val = latents.clone()
        aug_ts_val = torch.zeros(latents.shape[0] * latents.shape[2], device=dev, dtype=bf)
        validation_num_steps = 20
        if self.training_args.validation_sampling_steps:
            try:
                parsed = [int(s) for s in self.training_args.validation_sampling_steps.split(",") if s.strip()]
                if parsed:
                    validation_num_steps = parsed[0]
            except ValueError:
                pass
        scheduler = FlowMatchDiscreteScheduler(
            shift=self.training_args.ode_shift, reverse=True, solver="euler")
        scheduler.set_timesteps(validation_num_steps, device=dev)
        with self.ema.apply_to_model(self.transformer):
            for t in scheduler.timesteps:
                timesteps_in = t.unsqueeze(0).expand(x.shape[0] * x.shape[2]).to(dev, dtype=bf)
                clean_cond = self._prepare_cond_latents(self.task_type, image_cond, clean_x_val, multitask_mask)
                clean_x_concat = torch.concat([clean_x_val, clean_cond], dim=1)
                with torch.autocast(device_type="cuda", dtype=bf):
                    if self._is_wan_model:
                        B, C, T, H, W = x.shape
                        patch_size = self.transformer.patch_size if hasattr(self.transformer, 'patch_size') else (1, 2, 2)
                        wan_kwargs = {
                            "x": x,
                            "t": timesteps_in.view(B, T),
                            "context": list(prompt_embed),
                            "seq_len": T * (H // patch_size[1]) * (W // patch_size[2]),
                            "clean_x": clean_x_val,
                            "aug_t": aug_ts_val.view(B, T),
                        }
                        pred = self.transformer(**wan_kwargs)
                    else:
                        out = self.transformer(
                            hidden_states=torch.concat([x, cond_latents], dim=1),
                            timestep=timesteps_in,
                            timestep_txt=torch.tensor(0).unsqueeze(0).to(dev, dtype=bf),
                            text_states=prompt_embed, text_states_2=None,
                            encoder_attention_mask=prompt_mask, timestep_r=None,
                            vision_states=vision_states, mask_type=self.task_type,
                            guidance=None,
                            extra_kwargs={"byt5_text_states": byt5_text_states, "byt5_text_mask": byt5_text_mask},
                            return_dict=False, clean_x=clean_x_concat, aug_timesteps=aug_ts_val,
                        )
                        pred = out[0] if isinstance(out, (tuple, list)) else out
                x = scheduler.step(pred, t, x).prev_sample
        vae = self.get_module("vae").to(dev)
        scaling_factor = vae.config.scaling_factor
        shift_factor = getattr(vae.config, "shift_factor", None)
        x_decoded = x / scaling_factor + shift_factor if shift_factor else x / scaling_factor
        with torch.autocast(device_type="cuda", dtype=bf):
            video = vae.decode(x_decoded).sample
        if self.rank_in_sp_group == 0:
            video = (video.float().clamp(-1, 1) + 1) / 2
            frames = rearrange(video[0], "c t h w -> t h w c")
            frames = (frames.cpu().numpy() * 255).astype(np.uint8)
            val_dir = os.path.join(self.training_args.output_dir, "validation", f"step_{step:07d}")
            os.makedirs(val_dir, exist_ok=True)
            fname = os.path.join(val_dir, f"sample_{val_idx:02d}_spgroup_{sp_group_idx:02d}.mp4")
            imageio.mimsave(fname, frames, fps=8)
            logger.info("Saved validation video to %s", fname)
            del frames
        vae.cpu()
        del vae, video, x_decoded
        torch.cuda.empty_cache()
        self.sp_group.barrier()

    def _log_training_info(self):
        total_batch_size = (self.world_size *
                            self.training_args.gradient_accumulation_steps /
                            self.training_args.sp_size *
                            self.training_args.train_sp_batch_size)
        logger.info("***** Running Consistency Distillation training *****")
        logger.info("  Num examples = %s", len(self.train_dataset))
        logger.info("  Dataloader size = %s", len(self.train_dataloader))
        logger.info("  Num Epochs = %s", self.num_train_epochs)
        logger.info("  Resume training from step %s", self.init_steps)
        logger.info("  Instantaneous batch size per device = %s", self.training_args.train_batch_size)
        logger.info("  Total train batch size = %s", total_batch_size)
        logger.info("  Gradient Accumulation steps = %s", self.training_args.gradient_accumulation_steps)
        logger.info("  Total optimization steps = %s", self.training_args.max_train_steps)
        logger.info("  Total training parameters per FSDP shard = %s B",
                    round(_get_trainable_params(self.transformer) / 1e9, 3))
        logger.info("  CFG scale = %s", self.cfg_scale)
        logger.info("  EMA decay = %s", self.ema.decay)
        gpu_memory_usage = torch.cuda.memory_allocated() / 1024**2
        logger.info("GPU memory usage before train_one_step: %s MB", gpu_memory_usage)
