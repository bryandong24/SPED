# SPDX-License-Identifier: Apache-2.0
"""
DMD utility functions adapted for HY-WorldPlay.
"""
import copy
from typing import Any

import torch

from trainer.logger import init_logger
from trainer.pipelines.pipeline_batch_info import TrainingBatch

logger = init_logger(__name__)


def clone_batch_value(value: Any) -> Any:
    """Clone values in a TrainingBatch without tensor deepcopy."""
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, dict):
        return {k: clone_batch_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clone_batch_value(v) for v in value]
    return copy.deepcopy(value)


def clone_training_batch(batch: TrainingBatch) -> TrainingBatch:
    """Clone a TrainingBatch."""
    cloned_batch = TrainingBatch()
    for key, value in batch.__dict__.items():
        setattr(cloned_batch, key, clone_batch_value(value))
    return cloned_batch


def parse_denoising_steps(steps_str: str) -> list[int]:
    """Parse denoising steps from comma-separated string."""
    if not steps_str:
        return [1000, 750, 500, 250]
    return [int(x.strip()) for x in steps_str.split(",")]


def create_gradient_mask(
    num_frames: int,
    num_frames_to_keep: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Create a gradient mask for dynamic frame generation.

    Last num_frames_to_keep frames have mask=True,
    First num_frames - num_frames_to_keep frames have mask=False.

    Args:
        num_frames: Total number of frames
        num_frames_to_keep: Number of frames from the end to keep gradients
        device: Device to create the mask on

    Returns:
        Gradient mask with shape [num_frames]
    """
    mask = torch.zeros(num_frames, device=device, dtype=torch.float)
    if num_frames > num_frames_to_keep:
        mask[-num_frames_to_keep:] = 1.0
    else:
        mask[:] = 1.0
    return mask


def select_memory_frames(
    viewmats: torch.Tensor,
    current_frame_idx: int,
    memory_frames: int = 20,
    temporal_context_size: int = 12,
    device: torch.device = None,
) -> list[int]:
    """
    Select memory frames for AR rollout based on camera poses.

    This is a simplified version of select_aligned_memory_frames from HY-WorldPlay.
    For DMD training, we can use a simpler strategy.

    Args:
        viewmats: Camera viewmats [T, 3, 4]
        current_frame_idx: Current frame index to generate
        memory_frames: Number of memory frames to select
        temporal_context_size: Temporal context size for selection
        device: Device to use

    Returns:
        List of selected frame indices
    """
    num_total_frames = viewmats.shape[0]

    # Simple strategy: select recent frames before current_frame_idx
    start_idx = max(0, current_frame_idx - memory_frames)
    selected_indices = list(range(start_idx, current_frame_idx))

    # Ensure we don't exceed memory_frames
    if len(selected_indices) > memory_frames:
        # Sample uniformly if we have too many candidates
        step = len(selected_indices) // memory_frames
        selected_indices = selected_indices[::step][:memory_frames]

    return selected_indices


def get_sp_frame_indices(
    total_frames: int,
    sp_world_size: int,
    rank_in_sp_group: int,
) -> tuple[int, int]:
    """
    Get frame indices for current SP rank.

    Args:
        total_frames: Total number of frames
        sp_world_size: Sequence parallel world size
        rank_in_sp_group: Current rank in SP group

    Returns:
        (start_idx, end_idx) for this rank
    """
    frames_per_rank = total_frames // sp_world_size
    start_idx = rank_in_sp_group * frames_per_rank
    end_idx = start_idx + frames_per_rank
    return start_idx, end_idx
