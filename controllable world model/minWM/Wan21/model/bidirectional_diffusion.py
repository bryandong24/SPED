from typing import Tuple
import torch

from algorithms.flow_matching import flow_matching_loss
from model.base import BaseModel
from wan_utils.wan_wrapper import WanDiffusionWrapper, WanTextEncoder, WanVAEWrapper


class BidirectionalDiffusion(BaseModel):
    """
    Bidirectional diffusion training model.
    Same as CausalDiffusion but with:
      - is_causal=False  (WanModel with flash attention, no causal mask)
      - uniform_timestep=True  (all frames share the same timestep)
      - No teacher forcing  (no clean_x / aug_t passed to generator)
    """
    def __init__(self, args, device):
        super().__init__(args, device)
        self.num_frame_per_block = getattr(args, "num_frame_per_block", 1)
        self.independent_first_frame = getattr(args, "independent_first_frame", False)

        if args.gradient_checkpointing:
            self.generator.enable_gradient_checkpointing()

        self.num_train_timestep = args.num_train_timestep
        self.min_step = int(0.02 * self.num_train_timestep)
        self.max_step = int(0.98 * self.num_train_timestep)
        self.guidance_scale = args.guidance_scale
        self.timestep_shift = getattr(args, "timestep_shift", 1.0)

    def _initialize_models(self, args, device):
        self.generator = WanDiffusionWrapper(
            **getattr(args, "model_kwargs", {}), is_causal=False
        )
        self.generator.model.requires_grad_(True)

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
        initial_latent: torch.Tensor = None
    ) -> Tuple[torch.Tensor, dict]:
        noise = torch.randn_like(clean_latent)
        batch_size, num_frame = image_or_video_shape[:2]

        # uniform_timestep=True: all frames share the same t
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

        # Bidirectional: no clean_x, no aug_t
        flow_pred, x0_pred = self.generator(
            noisy_image_or_video=noisy_latents,
            conditional_dict=conditional_dict,
            timestep=timestep,
        )

        weight = self.scheduler.training_weight(timestep).unflatten(
            0, (batch_size, num_frame)
        )
        # weight: [B, F] -> [B, F, 1, 1, 1] for per-frame weighting
        weight = weight.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        loss = flow_matching_loss(flow_pred, training_target, weight=weight)

        log_dict = {
            "x0": clean_latent.detach(),
            "x0_pred": x0_pred.detach(),
        }
        return loss, log_dict
