"""
src/schedule.py

Noise schedules for DDPM. A schedule defines beta_t (the variance of noise
added at each timestep) and derives the quantities needed for both the
forward process (adding noise) and reverse process (denoising).
"""

import torch
import math


def linear_beta_schedule(timesteps: int, beta_start: float = 1e-4, beta_end: float = 0.02) -> torch.Tensor:
    """
    Original DDPM schedule (Ho et al., 2020). Beta increases linearly
    from beta_start to beta_end across timesteps.
    """
    return torch.linspace(beta_start, beta_end, timesteps)


def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    """
    Cosine schedule (Nichol & Dhariwal, 2021 - 'Improved DDPM').
    Defines alpha_bar_t directly as a cosine function of t, then derives
    beta_t from it. Produces a gentler noise ramp than linear, which
    empirically improves sample quality on smaller datasets.

    s is a small offset preventing beta_t from being too small near t=0.
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps) / timesteps
    alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]  # normalize so alpha_bar_0 = 1
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, min=1e-4, max=0.999)  # clip for numerical stability


class NoiseSchedule:
    """
    Wraps a beta schedule and precomputes every derived quantity needed
    for the forward process (q_sample) and reverse process (sampling).

    Naming follows the DDPM paper (Ho et al., 2020) so it's easy to
    cross-reference the equations while reading this code:
        beta_t          : variance of noise added at step t
        alpha_t         = 1 - beta_t
        alpha_bar_t     = product of alpha_1 ... alpha_t (cumulative product)
    """

    def __init__(self, timesteps: int, schedule_type: str = "linear"):
        self.timesteps = timesteps
        self.schedule_type = schedule_type

        if schedule_type == "linear":
            betas = linear_beta_schedule(timesteps)
        elif schedule_type == "cosine":
            betas = cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f"Unknown schedule_type: {schedule_type}")

        self.betas = betas                                          # (T,)
        self.alphas = 1.0 - betas                                    # (T,)
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)      # alpha_bar_t, (T,)

        # Shifted cumulative product, used in the posterior variance formula
        # (needed later for sampling, precomputing now so it's all in one place)
        self.alphas_cumprod_prev = torch.cat(
            [torch.tensor([1.0]), self.alphas_cumprod[:-1]]
        )

        # Precompute square roots used repeatedly in q_sample / sampling
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

        # Posterior variance: beta_tilde_t from the DDPM paper (eq. 7)
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )

    def to(self, device):
        """Move all precomputed tensors to a device (cpu/cuda)."""
        for attr in [
            "betas", "alphas", "alphas_cumprod", "alphas_cumprod_prev",
            "sqrt_alphas_cumprod", "sqrt_one_minus_alphas_cumprod",
            "posterior_variance",
        ]:
            setattr(self, attr, getattr(self, attr).to(device))
        return self