"""ODE regression algorithm.

Extracts noisy input and target from pre-computed ODE trajectories,
then computes MSE regression loss.
"""

import torch
from torch import Tensor
from typing import List, Optional, Tuple


def prepare_ode_input(
    ode_trajectory: Tensor,
    denoising_step_list: List[int],
    batch_size: int,
    device: torch.device,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Sample a random timestep from ODE trajectory and extract input/target.

    The trajectory is ordered from most noisy (index 0) to near-clean (index -1).
    The last entry (index -1) is the clean latent; the second-to-last (index -2)
    is the regression target.

    Args:
        ode_trajectory: [B, N_steps, F, C, H, W] pre-computed ODE solutions.
            Ordered noisy → clean. Last entry = clean, second-to-last = target.
        denoising_step_list: list of timestep values corresponding to each step.
        batch_size: batch size B.
        device: target device.

    Returns:
        noisy_input: [B, F, C, H, W] sampled noisy latent.
        clean_latent: [B, F, C, H, W] the clean end of the trajectory.
        target_latent: [B, F, C, H, W] the regression target (second-to-last).
        timestep: [B, F] the timestep for each sample/frame.
    """
    # ode_trajectory: [B, N_steps, F, C, H, W]
    clean_latent = ode_trajectory[:, -1]    # [B, F, C, H, W]
    target_latent = ode_trajectory[:, -2]   # [B, F, C, H, W]
    valid_trajectory = ode_trajectory[:, :-1]  # exclude clean

    num_steps = valid_trajectory.shape[1]
    num_frames = valid_trajectory.shape[2]
    num_channels = valid_trajectory.shape[3]
    height = valid_trajectory.shape[4]
    width = valid_trajectory.shape[5]

    # Uniform random step index per sample (same across frames for TF)
    step_idx = torch.randint(0, num_steps, (batch_size, 1), device=device)
    # step_idx: [B, 1] -> gather index: [B, 1, F, C, H, W]
    gather_idx = step_idx.view(batch_size, 1, 1, 1, 1, 1).expand(
        batch_size, 1, num_frames, num_channels, height, width
    )

    noisy_input = torch.gather(
        valid_trajectory.to(device), dim=1, index=gather_idx
    ).squeeze(1)  # [B, F, C, H, W]

    # Build timestep tensor [B, F]
    denoising_steps = torch.tensor(denoising_step_list, device=device)
    timestep = denoising_steps[step_idx.expand(-1, num_frames)]  # [B, F]

    return noisy_input, clean_latent, target_latent, timestep


def ode_regression_loss(
    x0_pred: Tensor,
    target: Tensor,
    weight: Optional[Tensor] = None,
    mask: Optional[Tensor] = None,
) -> Tensor:
    """MSE regression loss between predicted x0 and ODE target.

    Args:
        x0_pred: model's x0 prediction [B, F, C, H, W]
        target: ODE regression target, same shape
        weight: optional per-element weight, broadcastable
        mask: boolean mask, True = include in loss
    """
    if mask is not None:
        x0_pred = x0_pred[mask]
        target = target[mask]
    loss = (x0_pred.float() - target.float()).pow(2)
    if weight is not None:
        if mask is not None:
            weight = weight[mask]
        loss = loss * weight
    return loss.mean()
