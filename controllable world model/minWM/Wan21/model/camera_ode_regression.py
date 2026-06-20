"""Camera-controlled ODE regression model (Stage 2a).

Inherits from ODERegression and only overrides:
1. Model initialization (WanDiffusionWrapper with use_camera=True)
2. generator_loss to pass viewmats/Ks to the generator
"""

from typing import Optional, Tuple
import torch
import torch.nn.functional as F

from model.ode_regression import ODERegression
from wan_utils.wan_wrapper import WanDiffusionWrapper


class CameraODERegression(ODERegression):
    def _initialize_models(self, args, device):
        self.generator = WanDiffusionWrapper(
            **getattr(args, "model_kwargs", {}), is_causal=True
        )
        self.generator.model.requires_grad_(True)
        assert self.generator.use_camera is True
        from wan_utils.wan_wrapper import WanTextEncoder, WanVAEWrapper
        self.text_encoder = WanTextEncoder()
        self.text_encoder.requires_grad_(False)

        self.vae = WanVAEWrapper()
        self.vae.requires_grad_(False)

        self.scheduler = self.generator.get_scheduler()
        self.scheduler.timesteps = self.scheduler.timesteps.to(device)

    def generator_loss(
        self,
        ode_latent: torch.Tensor,
        conditional_dict: dict,
        viewmats: Optional[torch.Tensor] = None,
        Ks: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
        from algorithms.ode_regression import ode_regression_loss

        clean_latent = ode_latent[:, -1]
        target_latent = ode_latent[:, -2]
        ode_latent_valid = ode_latent[:, :-1]

        noisy_input, timestep = self._prepare_generator_input(
            ode_latent=ode_latent_valid, tf=True, causal=True)

        _, pred_image_or_video = self.generator(
            noisy_image_or_video=noisy_input,
            conditional_dict=conditional_dict,
            timestep=timestep,
            clean_x=clean_latent,
            viewmats=viewmats,
            Ks=Ks,
        )

        mask = timestep != 0
        loss = ode_regression_loss(pred_image_or_video, target_latent, mask=mask)

        log_dict = {
            "unnormalized_loss": F.mse_loss(pred_image_or_video, target_latent, reduction='none').mean(dim=[1, 2, 3, 4]).detach(),
            "timestep": timestep.float().mean(dim=1).detach(),
            "input": noisy_input.detach(),
            "output": pred_image_or_video.detach(),
        }

        return loss, log_dict
