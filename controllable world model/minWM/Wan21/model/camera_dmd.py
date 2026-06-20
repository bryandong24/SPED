"""Camera-controlled DMD model (Stage 3).

Inherits from DMD and only overrides _initialize_models to use use_camera=True
for generator, real_score, and fake_score.

viewmats/Ks are injected into conditional_dict by the trainer before calling
generator_loss/critic_loss, and propagate automatically through:
- _run_generator → _consistency_backward_simulation(**conditional_dict)
- compute_distribution_matching_loss → conditional_dict.get("viewmats")
- critic_loss → conditional_dict.get("viewmats")
"""

from model.dmd import DMD
from wan_utils.wan_wrapper import WanDiffusionWrapper, WanTextEncoder, WanVAEWrapper


class CameraDMD(DMD):
    def _initialize_models(self, args, device):
        real_model_name = getattr(args, "real_name", "Wan2.1-T2V-1.3B")
        fake_model_name = getattr(args, "fake_name", "Wan2.1-T2V-1.3B")
        model_kwargs = getattr(args, "model_kwargs", {})

        self.generator = WanDiffusionWrapper(**model_kwargs, is_causal=True)
        self.generator.model.requires_grad_(True)
        assert self.generator.use_camera is True
        self.real_score = WanDiffusionWrapper(model_name=real_model_name, **model_kwargs, is_causal=False)
        self.real_score.model.requires_grad_(False)
        assert self.real_score.use_camera is True
        self.fake_score = WanDiffusionWrapper(model_name=fake_model_name, **model_kwargs, is_causal=False)
        self.fake_score.model.requires_grad_(True)
        assert self.fake_score.use_camera is True
        self.text_encoder = WanTextEncoder()
        self.text_encoder.requires_grad_(False)

        self.vae = WanVAEWrapper()
        self.vae.requires_grad_(False)

        self.scheduler = self.generator.get_scheduler()
        self.scheduler.timesteps = self.scheduler.timesteps.to(device)
