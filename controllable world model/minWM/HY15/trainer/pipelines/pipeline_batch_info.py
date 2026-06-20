# SPDX-License-Identifier: Apache-2.0
"""
Data structures for functional pipeline processing.

This module defines the dataclasses used to pass state between pipeline components
in a functional manner, reducing the need for explicit parameter passing.
"""

import pprint
from dataclasses import asdict, dataclass, field
from typing import Any

import torch


@dataclass
class ForwardBatch:
    """
    State passed through the pipeline execution.
    Used in validation during training.
    """
    data_type: str

    generator: torch.Generator | list[torch.Generator] | None = None

    # Image inputs
    image_embeds: list[torch.Tensor] = field(default_factory=list)

    # Text inputs
    prompt: str | list[str] | None = None
    output_path: str = "outputs/"
    prompt_embeds: list[torch.Tensor] = field(default_factory=list)
    prompt_attention_mask: list[torch.Tensor] | None = None

    # Batch info
    batch_size: int | None = None
    num_videos_per_prompt: int = 1
    seed: int | None = None
    seeds: list[int] | None = None

    # Latent tensors
    latents: torch.Tensor | None = None
    raw_latent_shape: torch.Tensor | None = None

    # Latent dimensions
    height_latents: list[int] | int | None = None
    width_latents: list[int] | int | None = None
    num_frames: list[int] | int = 1

    # Original dimensions (before VAE scaling)
    height: list[int] | int | None = None
    width: list[int] | int | None = None
    fps: list[int] | int | None = None

    # Timesteps
    timesteps: torch.Tensor | None = None

    # Scheduler parameters
    num_inference_steps: int = 50
    guidance_scale: float = 1.0

    n_tokens: int | None = None

    # Final output
    output: Any = None

    # VSA parameters
    VSA_sparsity: float = 0.0

    def __str__(self):
        return pprint.pformat(asdict(self), indent=2, width=120)


@dataclass
class TrainingBatch:
    current_timestep: int = 0
    current_vsa_sparsity: float = 0.0

    # Dataloader batch outputs
    latents: torch.Tensor | None = None
    prompt_embed: torch.Tensor | None = None
    neg_prompt_embed: torch.Tensor | None = None
    i2v_mask: torch.Tensor | None = None
    window_frames: int = 0
    per_seq_length: int = 0
    current_start: int = 0
    current_end: int = 0
    video_path: str = None
    stage_one: bool = False
    gt_latent: torch.Tensor | None = None
    select_window_out_flag: int = 0

    raw_latent_shape: torch.Tensor | None = None
    noise_latents: torch.Tensor | None = None

    image_cond: torch.Tensor | None = None
    vision_states: torch.Tensor | None = None
    viewmats: torch.Tensor | None = None
    Ks: torch.Tensor | None = None
    prompt_mask: torch.Tensor | None = None
    byt5_text_states: torch.Tensor | None = None
    byt5_text_mask: torch.Tensor | None = None
    preprocessed_image: torch.Tensor | None = None

    # Transformer inputs
    noisy_model_input: torch.Tensor | None = None
    timesteps: torch.Tensor | None = None
    sigmas: torch.Tensor | None = None
    noise: torch.Tensor | None = None

    # Teacher forcing inputs
    clean_x: torch.Tensor | None = None        # clean latents (optionally augmented with small noise)
    aug_timesteps: torch.Tensor | None = None   # timestep for clean tokens
    ode_regress_target: torch.Tensor | None = None # target for ODE regression (e.g., clean latents)

    attn_metadata_vsa: Any | None = None
    attn_metadata: Any | None = None

    # input kwargs
    input_kwargs: dict[str, Any] | None = None

    # Training outputs
    total_loss: float | None = None
    grad_norm: float | None = None

    # Distillation losses
    generator_loss: float = 0.0
    fake_score_loss: float = 0.0
