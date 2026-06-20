"""Camera-controlled ODE regression trainer (Stage 2a).

Standalone trainer that swaps:
1. Model: CameraODERegression (use_camera=True)
2. Dataset: CameraODERegressionLMDBDataset (provides viewmats/Ks)
3. train_one_step: passes viewmats/Ks to generator_loss

All other logic (save, train loop, FSDP) inherited from base Trainer.
"""

import gc
import logging
from wan_utils.dataset import CameraODERegressionLMDBDataset, cycle
from collections import defaultdict
from wan_utils.misc import set_seed
import torch.distributed as dist
from omegaconf import OmegaConf
import torch
import wandb
import time
import os

from wan_utils.distributed import barrier, fsdp_wrap, fsdp_state_dict, launch_distributed_job, get_fsdp_process_group, get_sp_data_sampler, get_sp_seed_offset

from wan_trainer.ode import Trainer as _Base


class Trainer(_Base):
    def __init__(self, config):
        self.config = config
        self.step = 0

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        launch_distributed_job(sp_size=getattr(config, "sp_size", 1))
        from model import CameraODERegression
        fsdp_pg = get_fsdp_process_group()
        global_rank = dist.get_rank()
        self.world_size = dist.get_world_size()

        self.dtype = torch.bfloat16 if config.mixed_precision else torch.float32
        self.device = torch.cuda.current_device()
        self.is_main_process = global_rank == 0
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

        assert config.trainer == "camera_ode", "Only camera_ode trainer is supported"
        self.model = CameraODERegression(config, device=self.device)

        self.model.generator = fsdp_wrap(
            self.model.generator,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.generator_fsdp_wrap_strategy,
            process_group=fsdp_pg
        )
        self.model.text_encoder = fsdp_wrap(
            self.model.text_encoder,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.text_encoder_fsdp_wrap_strategy,
            cpu_offload=getattr(config, "text_encoder_cpu_offload", False),
            process_group=fsdp_pg
        )

        if not config.no_visualize or config.load_raw_video:
            self.model.vae = self.model.vae.to(
                device=self.device, dtype=torch.bfloat16 if config.mixed_precision else torch.float32)

        self.generator_optimizer = torch.optim.AdamW(
            [param for param in self.model.generator.parameters() if param.requires_grad],
            lr=config.lr, betas=(config.beta1, config.beta2), weight_decay=config.weight_decay
        )

        dataset = CameraODERegressionLMDBDataset(
            config.data_path, max_pair=getattr(config, "max_pair", int(1e8)))
        sampler = get_sp_data_sampler(dataset, shuffle=True, drop_last=True)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=config.batch_size, sampler=sampler, num_workers=8)
        self.dataloader = cycle(dataloader)

        self.step = 0

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

        self.max_grad_norm = 10.0
        self.previous_time = None

    def train_one_step(self, loss_scale=1.0):
        self.model.eval()

        batch = next(self.dataloader)
        text_prompts = batch["prompts"]
        ode_latent = batch["ode_latent"].to(device=self.device, dtype=self.dtype)
        viewmats = batch["viewmats"].to(device=self.device, dtype=self.dtype)
        Ks = batch["Ks"].to(device=self.device, dtype=self.dtype)

        with torch.no_grad():
            conditional_dict = self.model.text_encoder(text_prompts=text_prompts)

        generator_loss, log_dict = self.model.generator_loss(
            ode_latent=ode_latent,
            conditional_dict=conditional_dict,
            viewmats=viewmats,
            Ks=Ks
        )

        unnormalized_loss = log_dict["unnormalized_loss"]
        timestep = log_dict["timestep"]

        if self.world_size > 1:
            gathered_unnormalized_loss = torch.zeros(
                [self.world_size, *unnormalized_loss.shape],
                dtype=unnormalized_loss.dtype, device=self.device)
            gathered_timestep = torch.zeros(
                [self.world_size, *timestep.shape],
                dtype=timestep.dtype, device=self.device)

            dist.all_gather_into_tensor(gathered_unnormalized_loss, unnormalized_loss)
            dist.all_gather_into_tensor(gathered_timestep, timestep)
        else:
            gathered_unnormalized_loss = unnormalized_loss
            gathered_timestep = timestep

        loss_breakdown = defaultdict(list)
        stats = {}

        for index, t in enumerate(timestep):
            loss_breakdown[str(int(t.item()) // 250 * 250)].append(
                unnormalized_loss[index].item())

        for key_t in loss_breakdown.keys():
            stats["loss_at_time_" + key_t] = sum(loss_breakdown[key_t]) / len(loss_breakdown[key_t])

        self.generator_optimizer.zero_grad()
        (generator_loss * loss_scale).backward()
        if self.is_main_process and not getattr(self, "_first_bp_logged", False):
            print("[Trainer Entry] Wan21/wan_trainer/camera_ode.py :: Trainer.train_one_step (first BP done)", flush=True)
            self._first_bp_logged = True
        generator_grad_norm = self.model.generator.clip_grad_norm_(self.max_grad_norm)
        self.generator_optimizer.step()

        if self.is_main_process and not self.disable_wandb:
            wandb_loss_dict = {
                "generator_loss": generator_loss.item(),
                "generator_grad_norm": generator_grad_norm.item(),
                **stats
            }
            wandb.log(wandb_loss_dict, step=self.step)

        if self.step % self.config.gc_interval == 0:
            if dist.get_rank() == 0:
                logging.info("DistGarbageCollector: Running GC.")
            gc.collect()
