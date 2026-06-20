"""Self-Forcing sub-functions.

Exit step sampling, gradient mask creation/application.
Shared by HY15 and Wan21.
"""

import torch
from torch import Tensor
from typing import List, Optional


def sample_exit_step(
    num_denoising_steps: int,
    num_blocks: int,
    same_across_blocks: bool = True,
    device: Optional[torch.device] = None,
) -> List[int]:
    """Sample exit step indices for self-forcing truncated denoising.

    Each block gets an exit step index in [0, num_denoising_steps).
    At the exit step, the generator runs with grad; other steps are no-grad.

    Args:
        num_denoising_steps: total number of denoising steps
        num_blocks: number of temporal blocks
        same_across_blocks: if True, all blocks share the same exit step
        device: device for random generation
    """
    if same_across_blocks:
        idx = torch.randint(0, num_denoising_steps, (1,), device=device).item()
        return [idx] * num_blocks
    else:
        indices = torch.randint(
            0, num_denoising_steps, (num_blocks,), device=device
        )
        return indices.tolist()


def create_gradient_mask(
    total_frames: int,
    last_n_frames: int,
    shape: List[int],
    frame_dim: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Create a gradient mask that is 1 for the last N frames, 0 elsewhere.

    Used to restrict DMD loss to only the last N generated frames,
    avoiding gradient through early (context) frames.

    Args:
        total_frames: total number of frames in the video
        last_n_frames: number of trailing frames to include in gradient
        shape: full tensor shape (e.g. [B, F, C, H, W])
        frame_dim: which dimension is the frame dimension
        device: target device
        dtype: mask dtype (float for multiplication, bool for indexing)
    """
    mask = torch.zeros(shape, device=device, dtype=dtype)
    start = total_frames - last_n_frames
    if start < 0:
        start = 0
    # Build slice for the frame dimension
    slices = [slice(None)] * len(shape)
    slices[frame_dim] = slice(start, total_frames)
    mask[tuple(slices)] = 1.0 if dtype != torch.bool else True
    return mask


def apply_gradient_mask(video: Tensor, mask: Tensor) -> Tensor:
    """Apply gradient mask: pass gradient only where mask is nonzero.

    For masked regions, detach the tensor (stop gradient).
    For unmasked regions, keep the gradient.

    result = video * mask + video.detach() * (1 - mask)

    Args:
        video: tensor with gradient [B, F, C, H, W]
        mask: float mask, 1 = keep grad, 0 = detach
    """
    return video * mask + video.detach() * (1.0 - mask)
