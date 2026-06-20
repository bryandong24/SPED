"""Consistency Distillation algorithm.

Teacher CFG -> Euler step -> student/EMA consistency loss.
Shared by HY15 and Wan21.
"""

import torch
from torch import Tensor
from typing import Tuple


def sample_cd_timestep_pair(
    num_steps: int,
    sigmas: Tensor,
    timesteps: Tensor,
    device: torch.device,
    include_terminal: bool = False,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Sample adjacent (t, t_next) pair for consistency distillation.

    Args:
        num_steps: total number of discrete steps
        sigmas: [N+1] sigma schedule from scheduler
        timesteps: [N] timestep schedule from scheduler
        device: target device
        include_terminal: if True, allow sampling the terminal step

    Returns:
        sigma_t: scalar sigma at step t
        sigma_t_next: scalar sigma at step t+1
        t: scalar timestep at step t
        t_next: scalar timestep at step t+1
    """
    max_idx = num_steps - 1 if not include_terminal else num_steps
    idx = torch.randint(0, max_idx, (1,), device=device).item()
    t = timesteps[idx]
    t_next = timesteps[idx + 1] if idx + 1 < len(timesteps) else torch.tensor(0.0, device=device)
    sigma_t = sigmas[idx]
    sigma_t_next = sigmas[idx + 1]
    return sigma_t, sigma_t_next, t, t_next


def teacher_cfg_euler_step(
    v_cond: Tensor,
    v_uncond: Tensor,
    latent_t: Tensor,
    t: Tensor,
    t_next: Tensor,
    guidance_scale: float,
    timestep_scale: float = 1000.0,
) -> Tensor:
    """Teacher CFG + single Euler step to produce the target latent.

    v_cfg = v_uncond + guidance_scale * (v_cond - v_uncond)
    dt = (t - t_next) / timestep_scale
    latent_t_next = latent_t - dt * v_cfg

    Args:
        v_cond: teacher conditional velocity prediction
        v_uncond: teacher unconditional velocity prediction
        latent_t: noisy latent at timestep t
        t: current timestep [B, F] or scalar
        t_next: next timestep [B, F] or scalar
        guidance_scale: CFG scale
        timestep_scale: divisor for dt (1000 for Wan21, 1 for HY15)
    """
    v_cfg = v_uncond + guidance_scale * (v_cond - v_uncond)
    dt = (t - t_next) / timestep_scale
    # Reshape dt for broadcasting: [B, F] -> [B, F, 1, 1, 1]
    while dt.dim() < v_cfg.dim():
        dt = dt.unsqueeze(-1)
    return latent_t - dt * v_cfg


def consistency_loss(
    cm_pred_t: Tensor,
    cm_pred_t_next: Tensor,
    reduction: str = "mean",
) -> Tensor:
    """Consistency distillation loss: MSE between student(t) and EMA(t_next).

    Args:
        cm_pred_t: student model's consistency prediction at t
        cm_pred_t_next: EMA model's consistency prediction at t_next (detached)
        reduction: "mean" or "none"
    """
    loss = (cm_pred_t.float() - cm_pred_t_next.float()).pow(2)
    if reduction == "mean":
        return loss.mean()
    return loss
