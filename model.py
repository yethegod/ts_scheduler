"""Time-series DDPM built on diffusers' 1D UNet.

This module implements the core DDPM math (noise schedule, training loss,
sampling loop) around a UNet1DModel from huggingface-diffusers.
"""

from __future__ import annotations

import math
import inspect
from typing import Iterable, Literal, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    # Recent diffusers
    from diffusers import UNet1DModel
except ImportError:  # pragma: no cover - older diffusers naming
    from diffusers import Unet1dModel as UNet1DModel



BetaSchedule = Literal["cosine", "sigmoid"]


def cosine_beta_schedule(num_steps: int, s: float = 0.008) -> torch.Tensor:
    """Cosine schedule from Nichol & Dhariwal."""
    steps = num_steps
    t = torch.linspace(0, steps, steps + 1, dtype=torch.float64)
    alphas_cumprod = torch.cos(((t / steps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return betas.clamp(min=1e-5, max=0.999).to(torch.float32)


def sigmoid_beta_schedule(
    num_steps: int, beta_start: float = 1e-4, beta_end: float = 0.02
) -> torch.Tensor:
    """Smooth sigmoid schedule covering beta_start..beta_end."""
    steps = torch.linspace(-6, 6, num_steps, dtype=torch.float32)
    betas = torch.sigmoid(steps) * (beta_end - beta_start) + beta_start
    return betas.clamp(min=1e-5, max=0.999)


def make_beta_schedule(num_steps: int, schedule: BetaSchedule = "cosine") -> torch.Tensor:
    if schedule == "cosine":
        return cosine_beta_schedule(num_steps)
    if schedule == "sigmoid":
        return sigmoid_beta_schedule(num_steps)
    raise ValueError(f"Unknown beta schedule: {schedule}")


def _default_block_types(n_blocks: int) -> Tuple[str, ...]:
    # DownBlock1D / UpBlock1D mirror pairs for a vanilla UNet1DModel.
    return tuple(["DownBlock1D"] * n_blocks), tuple(["UpBlock1D"] * n_blocks)


class TimeSeriesDDPM(nn.Module):
    """DDPM for 1D time-series built around diffusers' UNet1DModel."""

    def __init__(
        self,
        sample_size: int,
        in_channels: int = 1,
        base_channels: int = 64,
        channel_mults: Sequence[int] | None = (1, 2, 4, 8),
        layers_per_block: int = 2,
        beta_schedule: BetaSchedule = "cosine",
        num_train_timesteps: int = 1000,
        dropout: float = 0.1,
        act_fn: str = "silu",
        norm_num_groups: int = 32,
        down_block_types: Iterable[str] | None = None,
        up_block_types: Iterable[str] | None = None,
    ) -> None:
        super().__init__()

        block_out_channels = tuple(base_channels * m for m in channel_mults or (1, 2, 4, 8))
        n_blocks = len(block_out_channels)

        if down_block_types is None or up_block_types is None:
            down_block_types, up_block_types = _default_block_types(n_blocks)

        # diffusers UNet1DModel signature varies across versions; only pass supported args.
        unet_sig = inspect.signature(UNet1DModel.__init__).parameters
        kwargs = dict(
            sample_size=sample_size,
            in_channels=in_channels,
            out_channels=in_channels,
            layers_per_block=layers_per_block,
            block_out_channels=block_out_channels,
            down_block_types=tuple(down_block_types),
            up_block_types=tuple(up_block_types),
            norm_num_groups=norm_num_groups,
            act_fn=act_fn,
        )
        if "dropout" in unet_sig:
            kwargs["dropout"] = dropout

        self.unet = UNet1DModel(**kwargs)

        betas = make_beta_schedule(num_train_timesteps, beta_schedule)
        alphas = 1.0 - betas
        alpha_cumprod = torch.cumprod(alphas, dim=0)
        alpha_cumprod_prev = torch.cat(
            [torch.ones(1, dtype=torch.float32), alpha_cumprod[:-1]], dim=0
        )

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_cumprod", alpha_cumprod)
        self.register_buffer("alpha_cumprod_prev", alpha_cumprod_prev)

        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alpha_cumprod))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alpha_cumprod)
        )
        self.register_buffer("sqrt_recip_alphas", torch.sqrt(1.0 / alphas))
        self.register_buffer(
            "posterior_variance",
            betas * (1.0 - alpha_cumprod_prev) / (1.0 - alpha_cumprod),
        )

        self.num_train_timesteps = num_train_timesteps
        self.sample_size = sample_size
        self.in_channels = in_channels

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """Predict noise epsilon at given timesteps."""
        return self.unet(x, timesteps).sample

    def add_noise(
        self, x0: torch.Tensor, timesteps: torch.Tensor, noise: torch.Tensor | None = None
    ) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_alpha_cumprod_t = self._extract(self.sqrt_alphas_cumprod, timesteps, x0.shape)
        sqrt_one_minus_alpha_cumprod_t = self._extract(
            self.sqrt_one_minus_alphas_cumprod, timesteps, x0.shape
        )
        return sqrt_alpha_cumprod_t * x0 + sqrt_one_minus_alpha_cumprod_t * noise

    def training_loss(
        self, x0: torch.Tensor, noise: torch.Tensor | None = None
    ) -> torch.Tensor:
        """MSE between predicted and true noise, with random timestep sampling."""
        b = x0.shape[0]
        device = x0.device
        timesteps = torch.randint(0, self.num_train_timesteps, (b,), device=device, dtype=torch.long)
        if noise is None:
            noise = torch.randn_like(x0)
        x_noisy = self.add_noise(x0, timesteps, noise)
        noise_pred = self.forward(x_noisy, timesteps)
        return F.mse_loss(noise_pred, noise)

    @torch.no_grad()
    def p_sample(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """One reverse-diffusion step."""
        betas_t = self._extract(self.betas, t, x.shape)
        sqrt_recip_alphas_t = self._extract(self.sqrt_recip_alphas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(
            self.sqrt_one_minus_alphas_cumprod, t, x.shape
        )

        # Predict noise and compute model mean.
        eps_theta = self.forward(x, t)
        model_mean = sqrt_recip_alphas_t * (x - betas_t * eps_theta / sqrt_one_minus_alphas_cumprod_t)

        # Add stochasticity except for the final step.
        posterior_variance_t = self._extract(self.posterior_variance, t, x.shape)
        noise = torch.randn_like(x)
        nonzero_mask = (t != 0).float().view(x.shape[0], *((1,) * (x.ndim - 1)))
        return model_mean + nonzero_mask * torch.sqrt(posterior_variance_t) * noise

    @torch.no_grad()
    def sample(
        self,
        num_samples: int,
        device: torch.device | None = None,
        initial_noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Generate samples by iterating p_sample from pure noise."""
        if device is None:
            device = next(self.parameters()).device
        x = (
            initial_noise
            if initial_noise is not None
            else torch.randn(
                num_samples, self.in_channels, self.sample_size, device=device
            )
        )
        for i in reversed(range(self.num_train_timesteps)):
            t = torch.full((num_samples,), i, device=device, dtype=torch.long)
            x = self.p_sample(x, t)
        return x

    @staticmethod
    def _extract(a: torch.Tensor, t: torch.Tensor, x_shape: torch.Size) -> torch.Tensor:
        """Gather 1D tensor `a` at positions t and reshape to broadcast over `x_shape`."""
        out = a.gather(-1, t)
        return out.reshape(t.shape[0], *((1,) * (len(x_shape) - 1)))


__all__ = [
    "TimeSeriesDDPM",
    "make_beta_schedule",
    "cosine_beta_schedule",
    "sigmoid_beta_schedule",
]
