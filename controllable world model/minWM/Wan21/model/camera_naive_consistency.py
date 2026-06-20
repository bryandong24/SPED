"""Camera-controlled consistency distillation model (Stage 2b).

Inherits from NaiveConsistency and only overrides:
1. Model initialization (WanDiffusionWrapper with use_camera=True)
2. generator_loss to pass viewmats/Ks to the generator
"""

from typing import Optional, Tuple
import torch
import random

from model.naive_consistency import NaiveConsistency
from wan_utils.wan_wrapper import WanDiffusionWrapper


class CameraNaiveConsistency(NaiveConsistency):
    def _initialize_models(self, args, device):
        self.generator = WanDiffusionWrapper(
            **getattr(args, "model_kwargs", {}), is_causal=True
        )
        self.generator.model.requires_grad_(True)
        assert self.generator.use_camera is True
        self.generator_ema = WanDiffusionWrapper(
            **getattr(args, "model_kwargs", {}), is_causal=True
        )
        self.generator_ema.model.requires_grad_(False)
        assert self.generator_ema.use_camera is True
        self.teacher = WanDiffusionWrapper(
            **getattr(args, "model_kwargs", {}), is_causal=True
        )
        self.teacher.model.requires_grad_(False)
        assert self.teacher.use_camera is True
        from wan_utils.wan_wrapper import WanTextEncoder, WanVAEWrapper
        self.text_encoder = WanTextEncoder()
        self.text_encoder.requires_grad_(False)

        self.scheduler = self.generator.get_scheduler()
        self.scheduler.timesteps = self.scheduler.timesteps.to(device)

    def generator_loss(
        self,
        conditional_dict,
        unconditional_dict,
        clean_latent,
        ema_model,
        viewmats: Optional[torch.Tensor] = None,
        Ks: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
        from algorithms.consistency_distillation import teacher_cfg_euler_step, consistency_loss

        clean_latent = clean_latent.to(self.device).to(torch.bfloat16)
        B, num_frames = clean_latent.shape[:2]
        timestep_idx = random.randrange(self.discrete_cd_N - 1)

        t = self.scheduler.timesteps[timestep_idx]
        timestep = t * torch.ones([B, num_frames], device=self.device, dtype=torch.bfloat16)
        t_next = self.scheduler.timesteps[timestep_idx + 1]
        timestep_next = t_next * torch.ones([B, num_frames], device=self.device, dtype=torch.bfloat16)

        noise = torch.randn_like(clean_latent)
        latent_t = self.scheduler.add_noise(
            clean_latent, noise=noise,
            timestep=t * torch.ones([1], device=self.device)
        ).to(torch.bfloat16)

        with torch.no_grad():
            v_cond, _ = self.teacher(
                latent_t, conditional_dict, timestep, clean_x=clean_latent,
                viewmats=viewmats, Ks=Ks)
            v_uncond, _ = self.teacher(
                latent_t, unconditional_dict, timestep, clean_x=clean_latent,
                viewmats=viewmats, Ks=Ks)
            latent_t_next = teacher_cfg_euler_step(
                v_cond=v_cond,
                v_uncond=v_uncond,
                latent_t=latent_t,
                t=timestep,
                t_next=timestep_next,
                guidance_scale=self.guidance_scale,
                timestep_scale=1000.0,
            )

        if self.generator.model.block_mask is None and self.teacher.model.block_mask is not None:
            self.generator.model.block_mask = self.teacher.model.block_mask
            self.generator_ema.model.block_mask = self.teacher.model.block_mask

        _, cm_pred_t = self.generator(
            latent_t, conditional_dict, timestep, clean_x=clean_latent,
            viewmats=viewmats, Ks=Ks)

        with torch.no_grad():
            ema_model.copy_to(self.generator_ema)
            _, cm_pred_t_next = self.generator_ema(
                latent_t_next, conditional_dict, timestep_next, clean_x=clean_latent,
                viewmats=viewmats, Ks=Ks)

        with torch.enable_grad():
            loss = consistency_loss(cm_pred_t, cm_pred_t_next, reduction="mean")

        log_dict = {
            "unnormalized_loss": consistency_loss(cm_pred_t, cm_pred_t_next, reduction="none").mean(dim=[1, 2, 3, 4]).detach(),
        }

        return loss, log_dict
