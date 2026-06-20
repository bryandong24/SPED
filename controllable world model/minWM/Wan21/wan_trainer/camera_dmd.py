"""Camera-controlled DMD trainer (Stage 3).

Standalone trainer that swaps:
1. Model: CameraDMD (use_camera=True for generator/real_score/fake_score)
2. Dataset: CameraLatentLMDBDataset (provides viewmats/Ks)
3. fwdbwd_one_step: injects viewmats/Ks into conditional_dict before calling model

All other logic (save, train loop, FSDP, EMA) inherited from base Trainer.
"""

from wan_utils.dataset import cycle, CameraLatentLMDBDataset
from wan_utils.distributed import EMA_FSDP, fsdp_wrap, fsdp_state_dict, launch_distributed_job, get_fsdp_process_group, get_sp_data_sampler, get_sp_seed_offset
from wan_utils.misc import set_seed
import torch.distributed as dist
from omegaconf import OmegaConf
import torch
import wandb
import time
import os

from wan_trainer.distillation import Trainer as _Base


class Trainer(_Base):
    def __init__(self, config):
        self.config = config
        self.step = 0

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        launch_distributed_job(sp_size=getattr(config, "sp_size", 1))
        from model import CameraDMD
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

        self.model = CameraDMD(config, device=self.device)

        self.fake_score_state_dict_cpu = self.model.fake_score.state_dict()

        self.model.generator = fsdp_wrap(
            self.model.generator,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.generator_fsdp_wrap_strategy,
            cpu_offload=False,
            process_group=fsdp_pg
        )
        self.model.real_score = fsdp_wrap(
            self.model.real_score,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.real_score_fsdp_wrap_strategy,
            cpu_offload=False,
            process_group=fsdp_pg
        )
        self.model.fake_score = fsdp_wrap(
            self.model.fake_score,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.fake_score_fsdp_wrap_strategy,
            cpu_offload=False,
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
            [p for p in self.model.generator.parameters() if p.requires_grad],
            lr=config.lr, betas=(config.beta1, config.beta2),
            weight_decay=config.weight_decay
        )
        self.critic_optimizer = torch.optim.AdamW(
            [p for p in self.model.fake_score.parameters() if p.requires_grad],
            lr=config.lr_critic if hasattr(config, "lr_critic") else config.lr,
            betas=(config.beta1_critic, config.beta2_critic),
            weight_decay=config.weight_decay
        )

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

        def _load_ckpt(ckpt_path):
            sd = torch.load(ckpt_path, map_location="cpu")
            key = next((k for k in ("generator", "model", "generator_ema") if k in sd), None)
            if key:
                sd = sd[key]
            return {k.replace("model._fsdp_wrapped_module.", "model.", 1): v for k, v in sd.items()}

        if getattr(config, "generator_ckpt", False):
            print(f"Loading pretrained generator from {config.generator_ckpt}")
            self.model.generator.load_state_dict(_load_ckpt(config.generator_ckpt), strict=True)

        if getattr(config, "real_ckpt", False):
            print(f"Loading pretrained real_score from {config.real_ckpt}")
            self.model.real_score.load_state_dict(_load_ckpt(config.real_ckpt), strict=True)

        if getattr(config, "fake_ckpt", False):
            print(f"Loading pretrained fake_score from {config.fake_ckpt}")
            self.model.fake_score.load_state_dict(_load_ckpt(config.fake_ckpt), strict=True)

        if self.step < config.ema_start_step:
            self.generator_ema = None

        self.max_grad_norm_generator = getattr(config, "max_grad_norm_generator", 10.0)
        self.max_grad_norm_critic = getattr(config, "max_grad_norm_critic", 10.0)
        self.previous_time = None

    def fwdbwd_one_step(self, batch, train_generator, clean_latent=None):
        self.model.eval()

        if self.step % 20 == 0:
            torch.cuda.empty_cache()

        text_prompts = batch["prompts"]
        if self.config.i2v:
            image_latent = batch["ode_latent"][:, -1][:, 0:1, ].to(
                device=self.device, dtype=self.dtype)
        else:
            image_latent = None

        batch_size = len(text_prompts)
        image_or_video_shape = list(self.config.image_or_video_shape)
        image_or_video_shape[0] = batch_size

        with torch.no_grad():
            conditional_dict = self.model.text_encoder(text_prompts=text_prompts)

            if not getattr(self, "unconditional_dict", None):
                unconditional_dict = self.model.text_encoder(
                    text_prompts=[self.config.negative_prompt] * batch_size)
                unconditional_dict = {k: v.detach() for k, v in unconditional_dict.items()}
                self.unconditional_dict = unconditional_dict
            else:
                unconditional_dict = self.unconditional_dict

        conditional_dict["viewmats"] = batch["viewmats"].to(device=self.device, dtype=self.dtype)
        conditional_dict["Ks"] = batch["Ks"].to(device=self.device, dtype=self.dtype)

        if train_generator:
            generator_loss, generator_log_dict = self.model.generator_loss(
                image_or_video_shape=image_or_video_shape,
                conditional_dict=conditional_dict,
                unconditional_dict=unconditional_dict,
                clean_latent=clean_latent,
                initial_latent=image_latent if self.config.i2v else None
            )
            generator_loss.backward()
            if self.is_main_process and not getattr(self, "_first_bp_logged_generator", False):
                print("[Trainer Entry] Wan21/wan_trainer/camera_dmd.py :: Trainer.fwdbwd_one_step (first generator BP done)", flush=True)
                self._first_bp_logged_generator = True
            generator_grad_norm = self.model.generator.clip_grad_norm_(self.max_grad_norm_generator)
            generator_log_dict.update({"generator_loss": generator_loss,
                                       "generator_grad_norm": generator_grad_norm})
            return generator_log_dict

        critic_loss, critic_log_dict = self.model.critic_loss(
            image_or_video_shape=image_or_video_shape,
            conditional_dict=conditional_dict,
            unconditional_dict=unconditional_dict,
            clean_latent=clean_latent,
            initial_latent=image_latent if self.config.i2v else None
        )
        critic_loss.backward()
        if self.is_main_process and not getattr(self, "_first_bp_logged_critic", False):
            print("[Trainer Entry] Wan21/wan_trainer/camera_dmd.py :: Trainer.fwdbwd_one_step (first critic BP done)", flush=True)
            self._first_bp_logged_critic = True
        critic_grad_norm = self.model.fake_score.clip_grad_norm_(self.max_grad_norm_critic)
        critic_log_dict.update({"critic_loss": critic_loss,
                                "critic_grad_norm": critic_grad_norm})
        return critic_log_dict
