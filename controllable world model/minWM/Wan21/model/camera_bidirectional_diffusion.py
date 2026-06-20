"""Camera-controlled bidirectional diffusion model.

Inherits from BidirectionalDiffusion and only overrides:
1. Model initialization (WanDiffusionWrapper with use_camera=True)
2. generator_loss to pass viewmats/Ks to the generator
"""

from typing import Optional, Tuple
import torch

from model.bidirectional_diffusion import BidirectionalDiffusion
from wan_utils.wan_wrapper import WanDiffusionWrapper


class CameraBidirectionalDiffusion(BidirectionalDiffusion):
    def _initialize_models(self, args, device):
        self.generator = WanDiffusionWrapper(
            **getattr(args, "model_kwargs", {}), is_causal=False
        )
        assert self.generator.use_camera is True
        self.generator.model.requires_grad_(True)

        from wan_utils.wan_wrapper import WanTextEncoder, WanVAEWrapper
        self.text_encoder = WanTextEncoder()
        self.text_encoder.requires_grad_(False)

        self.vae = WanVAEWrapper()
        self.vae.requires_grad_(False)

        self.scheduler = self.generator.get_scheduler()
        self.scheduler.timesteps = self.scheduler.timesteps.to(device)

    def generator_loss(
        self,
        image_or_video_shape,
        conditional_dict: dict,
        unconditional_dict: dict,
        clean_latent: torch.Tensor,
        initial_latent: torch.Tensor = None,
        viewmats: Optional[torch.Tensor] = None,
        Ks: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
        from algorithms.flow_matching import flow_matching_loss

        noise = torch.randn_like(clean_latent)
        batch_size, num_frame = image_or_video_shape[:2]

        index = self._get_timestep(
            0,
            self.scheduler.num_train_timesteps,
            image_or_video_shape[0],
            image_or_video_shape[1],
            self.num_frame_per_block,
            uniform_timestep=True,
        )
        timestep = self.scheduler.timesteps[index].to(
            dtype=self.dtype, device=self.device
        )
        noisy_latents = self.scheduler.add_noise(
            clean_latent.flatten(0, 1),
            noise.flatten(0, 1),
            timestep.flatten(0, 1),
        ).unflatten(0, (batch_size, num_frame))
        training_target = self.scheduler.training_target(
            clean_latent, noise, timestep
        )

        flow_pred, x0_pred = self.generator(
            noisy_image_or_video=noisy_latents,
            conditional_dict=conditional_dict,
            timestep=timestep,
            viewmats=viewmats,
            Ks=Ks,
        )

        weight = self.scheduler.training_weight(timestep).unflatten(
            0, (batch_size, num_frame)
        )
        weight = weight.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        loss = flow_matching_loss(flow_pred, training_target, weight=weight)

        return loss, {"x0": clean_latent.detach(), "x0_pred": x0_pred.detach()}
