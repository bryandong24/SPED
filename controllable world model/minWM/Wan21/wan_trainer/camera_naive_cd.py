"""Camera-controlled consistency distillation trainer (Stage 2b).

Standalone trainer that swaps:
1. Model: CameraNaiveConsistency (use_camera=True)
2. Dataset: CameraLatentLMDBDataset (provides viewmats/Ks)
3. fwdbwd_one_step: passes viewmats/Ks doubled (teacher forcing) to generator_loss

All other logic (save, train loop, FSDP, EMA) inherited from base Trainer.
"""

import gc
import logging
from wan_utils.dataset import cycle, CameraLatentLMDBDataset
from wan_utils.distributed import EMA_FSDP, fsdp_wrap, fsdp_state_dict, launch_distributed_job, get_fsdp_process_group, get_sp_data_sampler, get_sp_seed_offset
from wan_utils.misc import set_seed, merge_dict_list
import torch.distributed as dist
from omegaconf import OmegaConf
import torch
import wandb
import time
import os

from wan_trainer.naive_cd import Trainer as _Base


class Trainer(_Base):
    def __init__(self, config):
        self.config = config
        self.step = 0

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        launch_distributed_job(sp_size=getattr(config, "sp_size", 1))
        from model import CameraNaiveConsistency
        fsdp_pg = get_fsdp_process_group()
        global_rank = dist.get_rank()
        self.world_size = dist.get_world_size()

        self.dtype = torch.bfloat16 if config.mixed_precision else torch.float32
        self.device = torch.cuda.current_device()
        self.is_main_process = global_rank == 0
        self.causal = config.causal
        self.disable_wandb = config.disable_wandb

        if config.seed == 0:
            random_seed = torch.randint(0, 10000000, (1,), device=self.device)
            dist.broadcast(random_seed, src=0)
            config.seed = random_seed.item()

        set_seed(config.seed + get_sp_seed_offset())

        if self.is_main_process and not self.disable_wandb:
            wandb.login(host=config.wandb_host, key=config.wandb_key)
            wandb.init(
                config=OmegaConf.to_container(config, resolve=True),
                name=config.config_name,
                entity=config.wandb_entity,
                project=config.wandb_project,
                dir=config.wandb_save_dir
            )

        self.output_path = config.logdir

        self.model = CameraNaiveConsistency(config, device=self.device)

        self.model.generator = fsdp_wrap(
            self.model.generator,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.generator_fsdp_wrap_strategy,
            cpu_offload=True,
            process_group=fsdp_pg
        )
        self.model.generator_ema = fsdp_wrap(
            self.model.generator_ema,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.generator_fsdp_wrap_strategy,
            cpu_offload=True,
            process_group=fsdp_pg
        )
        self.model.teacher = fsdp_wrap(
            self.model.teacher,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.real_score_fsdp_wrap_strategy,
            cpu_offload=True,
            process_group=fsdp_pg
        )
        self.model.text_encoder = fsdp_wrap(
            self.model.text_encoder,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.text_encoder_fsdp_wrap_strategy,
            cpu_offload=True,
            process_group=fsdp_pg
        )

        self.generator_optimizer = torch.optim.AdamW(
            [param for param in self.model.generator.parameters() if param.requires_grad],
            lr=config.lr, betas=(config.beta1, config.beta2), weight_decay=config.weight_decay
        )

        self.generator_ema = EMA_FSDP(self.model.generator, decay=self.config.ema_weight)

        dataset = CameraLatentLMDBDataset(config.data_path, max_pair=int(1e8))
        sampler = get_sp_data_sampler(dataset, shuffle=True, drop_last=True)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=config.batch_size, sampler=sampler, num_workers=8)
        if dist.get_rank() == 0:
            print("DATASET SIZE %d" % len(dataset))
        self.dataloader = cycle(dataloader)

        rename_param = (
            lambda name: name.replace("_fsdp_wrapped_module.", "")
            .replace("_checkpoint_wrapped_module.", "")
            .replace("_orig_mod.", "")
        )
        self.name_to_trainable_params = {}
        for n, p in self.model.generator.named_parameters():
            if not p.requires_grad:
                continue
            self.name_to_trainable_params[rename_param(n)] = p
        ema_weight = config.ema_weight
        self.generator_ema = None
        if (ema_weight is not None) and (ema_weight > 0.0):
            self.generator_ema = EMA_FSDP(self.model.generator, decay=ema_weight)

        if getattr(config, "generator_ckpt", False):
            print(f"Loading pretrained generator from {config.generator_ckpt}")
            state_dict = torch.load(config.generator_ckpt, map_location="cpu")
            if "generator" in state_dict:
                state_dict = state_dict["generator"]
                fixed = {}
                for k, v in state_dict.items():
                    if k.startswith("model._fsdp_wrapped_module."):
                        k = k.replace("model._fsdp_wrapped_module.", "model.", 1)
                    fixed[k] = v
                state_dict = fixed
            elif "model" in state_dict:
                state_dict = state_dict["model"]
            elif "generator_ema" in state_dict:
                gen_sd = state_dict["generator_ema"]
                fixed = {}
                for k, v in gen_sd.items():
                    if k.startswith("model._fsdp_wrapped_module."):
                        k = k.replace("model._fsdp_wrapped_module.", "model.", 1)
                    fixed[k] = v
                state_dict = fixed

            self.model.generator.load_state_dict(state_dict, strict=True)
            self.model.teacher.load_state_dict(state_dict, strict=True)

        self.max_grad_norm_generator = getattr(config, "max_grad_norm_generator", 10.0)
        self.max_grad_norm_critic = getattr(config, "max_grad_norm_critic", 10.0)
        self.previous_time = None

    def fwdbwd_one_step(self, batch, clean_latent=None):
        self.model.eval()

        if self.step % 20 == 0:
            torch.cuda.empty_cache()

        text_prompts = batch["prompts"]
        batch_size = len(text_prompts)
        image_or_video_shape = list(self.config.image_or_video_shape)
        image_or_video_shape[0] = batch_size

        viewmats = batch["viewmats"].to(device=self.device, dtype=self.dtype)
        Ks = batch["Ks"].to(device=self.device, dtype=self.dtype)

        with torch.no_grad():
            conditional_dict = self.model.text_encoder(text_prompts=text_prompts)
            if not getattr(self, "unconditional_dict", None):
                unconditional_dict = self.model.text_encoder(
                    text_prompts=[self.config.negative_prompt] * batch_size)
                unconditional_dict = {k: v.detach() for k, v in unconditional_dict.items()}
                self.unconditional_dict = unconditional_dict
            else:
                unconditional_dict = self.unconditional_dict

        generator_loss, generator_log_dict = self.model.generator_loss(
            conditional_dict=conditional_dict,
            unconditional_dict=unconditional_dict,
            clean_latent=clean_latent,
            ema_model=self.generator_ema,
            viewmats=viewmats,
            Ks=Ks
        )
        generator_loss.backward()
        if self.is_main_process and not getattr(self, "_first_bp_logged", False):
            print("[Trainer Entry] Wan21/wan_trainer/camera_naive_cd.py :: Trainer.fwdbwd_one_step (first BP done)", flush=True)
            self._first_bp_logged = True
        generator_grad_norm = self.model.generator.clip_grad_norm_(self.max_grad_norm_generator)

        generator_log_dict.update({
            "generator_loss": generator_loss,
            "generator_grad_norm": generator_grad_norm
        })

        return generator_log_dict
