
"""Bilevel noise-schedule DDPM for 1D time series.

This module mirrors the API of your `model.py` (TimeSeriesDDPM) but replaces the
fixed beta schedule with a *learnable* (bilevel-friendly) schedule parameterization.

Design goals:
- Schedule is low-dimensional (K knots) so SPSA/ZO outer loop is cheap.
- Schedule is guaranteed monotone in logSNR and has fixed endpoints (stable).
- Inner loop is standard DDPM noise-prediction training.
- Sampling uses the learned schedule (same math as DDPM).

Usage sketch (outer loop SPSA):
    model = TimeSeriesDDPMBilevel(..., schedule_num_knots=16).to(device)

    q = model.get_schedule_vector()              # vector for SPSA
    model.set_schedule_vector(q_plus);  ...      # train / evaluate
    model.set_schedule_vector(q_minus); ...      # train / evaluate

Notes:
- You can warm-start the schedule from cosine/sigmoid.
- If you change schedule parameters, call `model.refresh_schedule()` before sampling/training.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Literal, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from mlp import MLP1DModel

BetaSchedule = Literal["cosine", "sigmoid"]


# ----------------------------
# Fixed (baseline) schedules
# ----------------------------
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


# ----------------------------
# Learnable schedule (logSNR)
# ----------------------------
@dataclass(frozen=True)
class ScheduleBuffers:
    betas: torch.Tensor
    alphas: torch.Tensor
    alpha_cumprod: torch.Tensor
    alpha_cumprod_prev: torch.Tensor
    sqrt_alphas_cumprod: torch.Tensor
    sqrt_one_minus_alphas_cumprod: torch.Tensor
    sqrt_recip_alphas: torch.Tensor
    posterior_variance: torch.Tensor


class PiecewiseLinearLogSNRSchedule(nn.Module):
    """Monotone piecewise-linear logSNR schedule with fixed endpoints.

    Parameterization:
        logsnr_0 = logsnr_max (fixed)
        logsnr_K = logsnr_min (fixed)
        deltas_k = softplus(raw_deltas_k) > 0  for k=1..K
        logsnr_k = logsnr_max - scale * cumsum(deltas)_k
        where scale makes logsnr_K hit logsnr_min exactly.

    Then interpolate logSNR across diffusion steps t=0..T-1.
    """

    def __init__(
        self,
        num_train_timesteps: int,
        num_knots: int = 16,
        logsnr_max: float = 20.0,
        logsnr_min: float = -20.0,
        beta_min: float = 1e-5,
        beta_max: float = 0.999,
        init_from: BetaSchedule | None = "cosine",
    ) -> None:
        super().__init__()
        if num_knots < 2:
            raise ValueError("num_knots must be >= 2")
        if not (logsnr_max > logsnr_min):
            raise ValueError("Require logsnr_max > logsnr_min")

        self.num_train_timesteps = int(num_train_timesteps)
        self.num_knots = int(num_knots)
        self.logsnr_max = float(logsnr_max)
        self.logsnr_min = float(logsnr_min)
        self.beta_min = float(beta_min)
        self.beta_max = float(beta_max)

        # We keep K-1 positive increments between knots (excluding the first fixed knot).
        # raw_deltas shape: (num_knots-1,)
        self.raw_deltas = nn.Parameter(torch.zeros(self.num_knots - 1, dtype=torch.float32))

        # Optional warm start from a known beta schedule.
        if init_from is not None:
            with torch.no_grad():
                betas = make_beta_schedule(self.num_train_timesteps, init_from)
                self.init_from_betas_(betas)

    @staticmethod
    def _softplus_inv(y: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        # y = softplus(x) = log(1 + exp(x))
        # x = log(exp(y) - 1)
        y = torch.clamp(y, min=eps)
        return torch.log(torch.expm1(y))

    def knot_values(self) -> torch.Tensor:
        """Return logSNR values at knots: shape (num_knots,)."""
        deltas = F.softplus(self.raw_deltas)  # positive
        cum = torch.cumsum(deltas, dim=0)     # (K-1,)
        total = cum[-1].clamp_min(1e-8)
        scale = (self.logsnr_max - self.logsnr_min) / total
        # knots[0] = logsnr_max; knots[k] = logsnr_max - scale*cum[k-1]
        knots_tail = self.logsnr_max - scale * cum
        knots = torch.cat([torch.tensor([self.logsnr_max], device=knots_tail.device, dtype=knots_tail.dtype),
                           knots_tail], dim=0)
        return knots

    def logsnr_per_timestep(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Interpolate logSNR at every timestep t=0..T-1. Shape (T,)."""
        T = self.num_train_timesteps
        K = self.num_knots

        knots = self.knot_values().to(device=device, dtype=dtype)  # (K,)
        if T == 1:
            return knots[:1]

        # Map each timestep t to a position u in [0, K-1]
        t = torch.arange(T, device=device, dtype=dtype)
        u = t * (K - 1) / (T - 1)  # [0, K-1]
        k0 = torch.floor(u).to(torch.long)          # [0..K-1]
        k1 = torch.clamp(k0 + 1, max=K - 1)
        w = (u - k0.to(dtype)).clamp(0.0, 1.0)

        s0 = knots.gather(0, k0)
        s1 = knots.gather(0, k1)
        return (1.0 - w) * s0 + w * s1

    def alpha_cumprod(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """alpha_bar(t) in DDPM, computed as sigmoid(logSNR(t)). Shape (T,)."""
        logsnr = self.logsnr_per_timestep(device=device, dtype=dtype)
        # alpha^2 = sigmoid(logsnr), sigma^2 = sigmoid(-logsnr)
        alpha_bar = torch.sigmoid(logsnr)
        # avoid exact 0/1 for numerical stability
        return alpha_bar.clamp(min=1e-8, max=1.0 - 1e-8)

    def betas(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Convert alpha_bar to DDPM betas. Shape (T,)."""
        alpha_bar = self.alpha_cumprod(device=device, dtype=dtype)  # (T,)
        prev = torch.cat([torch.ones(1, device=device, dtype=dtype), alpha_bar[:-1]], dim=0)
        alphas = (alpha_bar / prev).clamp(min=1e-8, max=1.0)        # per-step alpha
        betas = (1.0 - alphas).clamp(min=self.beta_min, max=self.beta_max)
        return betas.to(dtype=dtype)

    @torch.no_grad()
    def init_from_betas_(self, betas: torch.Tensor) -> None:
        """Warm-start raw_deltas to roughly match a provided beta schedule (T,)."""
        betas = betas.detach().to(dtype=torch.float32).clamp(self.beta_min, self.beta_max)
        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0).clamp(min=1e-8, max=1.0 - 1e-8)
        logsnr = torch.log(alpha_bar) - torch.log1p(-alpha_bar)  # log(alpha/(1-alpha))

        # sample logsnr at knot locations
        T = betas.numel()
        K = self.num_knots
        if T == 1:
            target_knots = torch.tensor([self.logsnr_max], dtype=torch.float32, device=logsnr.device)
        else:
            t_knots = torch.linspace(0, T - 1, K, device=logsnr.device)
            idx0 = torch.floor(t_knots).to(torch.long)
            idx1 = torch.clamp(idx0 + 1, max=T - 1)
            w = (t_knots - idx0.float()).clamp(0.0, 1.0)
            target_knots = (1.0 - w) * logsnr[idx0] + w * logsnr[idx1]

        # enforce endpoints to our fixed max/min
        target_knots[0] = float(self.logsnr_max)
        target_knots[-1] = float(self.logsnr_min)

        # compute positive deltas between knots (monotone decreasing)
        diffs = (target_knots[:-1] - target_knots[1:]).clamp_min(1e-4)  # (K-1,) positive
        # We want deltas that after scaling reproduce diffs. In our parameterization,
        # diffs are proportional to softplus(raw_deltas). Start with equal scale.
        deltas = diffs / diffs.mean().clamp_min(1e-8)
        self.raw_deltas.copy_(self._softplus_inv(deltas))

    # ---- helpers for SPSA outer loop ----
    def get_vector(self) -> torch.Tensor:
        """Return the unconstrained schedule vector (raw_deltas)."""
        return self.raw_deltas.detach().clone()

    @torch.no_grad()
    def set_vector_(self, v: torch.Tensor) -> None:
        """Set the unconstrained schedule vector (raw_deltas) in-place."""
        if v.shape != self.raw_deltas.shape:
            raise ValueError(f"Expected vector of shape {tuple(self.raw_deltas.shape)}, got {tuple(v.shape)}")
        self.raw_deltas.copy_(v.to(device=self.raw_deltas.device, dtype=self.raw_deltas.dtype))

    @torch.no_grad()
    def project_(self) -> None:
        """Optional projection for numerical stability (keeps raw_deltas bounded)."""
        # This doesn't change the monotonic constraint (softplus), only avoids extreme scales.
        self.raw_deltas.clamp_(min=-20.0, max=20.0)

    def make_buffers(self, device: torch.device, dtype: torch.dtype) -> ScheduleBuffers:
        """Precompute all DDPM buffers for current schedule (on given device/dtype)."""
        betas = self.betas(device=device, dtype=dtype)  # (T,)
        alphas = (1.0 - betas).clamp(min=1e-8, max=1.0)
        alpha_cumprod = torch.cumprod(alphas, dim=0)
        alpha_cumprod_prev = torch.cat([torch.ones(1, device=device, dtype=dtype), alpha_cumprod[:-1]], dim=0)

        sqrt_alphas_cumprod = torch.sqrt(alpha_cumprod)
        sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alpha_cumprod)
        sqrt_recip_alphas = torch.sqrt(1.0 / alphas)

        posterior_variance = betas * (1.0 - alpha_cumprod_prev) / (1.0 - alpha_cumprod)
        posterior_variance = posterior_variance.clamp(min=1e-20)

        return ScheduleBuffers(
            betas=betas,
            alphas=alphas,
            alpha_cumprod=alpha_cumprod,
            alpha_cumprod_prev=alpha_cumprod_prev,
            sqrt_alphas_cumprod=sqrt_alphas_cumprod,
            sqrt_one_minus_alphas_cumprod=sqrt_one_minus_alphas_cumprod,
            sqrt_recip_alphas=sqrt_recip_alphas,
            posterior_variance=posterior_variance,
        )


# ----------------------------
# DDPM wrapper using schedule
# ----------------------------
class TimeSeriesDDPMBilevel(nn.Module):
    """DDPM for 1D time-series with a learnable (bilevel) noise schedule."""

    def __init__(
        self,
        sample_size: int,
        in_channels: int = 1,
        num_train_timesteps: int = 1000,
        # backbone
        d_model: int = 256,
        d_mlp: int | None = None,
        num_layers: int = 6,
        dropout: float = 0.1,
        time_embed_dim: int | None = None,
        # schedule
        schedule_num_knots: int = 16,
        schedule_init_from: BetaSchedule | None = "cosine",
        logsnr_max: float = 20.0,
        logsnr_min: float = -20.0,
    ) -> None:
        super().__init__()
        d_mlp = d_mlp if d_mlp is not None else 2 * d_model

        self.unet = MLP1DModel(
            sample_size=sample_size,
            in_channels=in_channels,
            out_channels=in_channels,
            d_model=d_model,
            d_mlp=d_mlp,
            num_layers=num_layers,
            dropout=dropout,
            time_embed_dim=time_embed_dim,
        )

        self.schedule = PiecewiseLinearLogSNRSchedule(
            num_train_timesteps=num_train_timesteps,
            num_knots=schedule_num_knots,
            logsnr_max=logsnr_max,
            logsnr_min=logsnr_min,
            init_from=schedule_init_from,
        )

        self.num_train_timesteps = int(num_train_timesteps)
        self.sample_size = int(sample_size)
        self.in_channels = int(in_channels)

        # Small cache: recompute when schedule is changed via refresh_schedule()
        self._cached_buffers: ScheduleBuffers | None = None
        self._cached_device: torch.device | None = None
        self._cached_dtype: torch.dtype | None = None

    # ---- schedule helpers ----
    def refresh_schedule(self) -> None:
        """Invalidate cached buffers (call after set_schedule_vector_ / schedule updates)."""
        self._cached_buffers = None
        self._cached_device = None
        self._cached_dtype = None

    def _get_schedule_buffers(self, device: torch.device, dtype: torch.dtype) -> ScheduleBuffers:
        if (
            self._cached_buffers is None
            or self._cached_device != device
            or self._cached_dtype != dtype
        ):
            # Buffers are cached without grad tracking; schedule is updated via SPSA, not backprop.
            with torch.no_grad():
                self._cached_buffers = self.schedule.make_buffers(device=device, dtype=dtype)
            self._cached_device = device
            self._cached_dtype = dtype
        return self._cached_buffers

    def get_schedule_vector(self) -> torch.Tensor:
        return self.schedule.get_vector()

    @torch.no_grad()
    def set_schedule_vector_(self, v: torch.Tensor, project: bool = True) -> None:
        self.schedule.set_vector_(v)
        if project:
            self.schedule.project_()
        self.refresh_schedule()

    # ---- DDPM core ----
    def forward(self, x: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """Predict noise epsilon at given timesteps."""
        return self.unet(x, timesteps).sample

    def add_noise(
        self, x0: torch.Tensor, timesteps: torch.Tensor, noise: torch.Tensor | None = None
    ) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(x0)
        buf = self._get_schedule_buffers(device=x0.device, dtype=x0.dtype)
        sqrt_alpha_cumprod_t = self._extract(buf.sqrt_alphas_cumprod, timesteps, x0.shape)
        sqrt_one_minus_alpha_cumprod_t = self._extract(buf.sqrt_one_minus_alphas_cumprod, timesteps, x0.shape)
        return sqrt_alpha_cumprod_t * x0 + sqrt_one_minus_alpha_cumprod_t * noise

    def training_loss(self, x0: torch.Tensor, noise: torch.Tensor | None = None) -> torch.Tensor:
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
        buf = self._get_schedule_buffers(device=x.device, dtype=x.dtype)
        betas_t = self._extract(buf.betas, t, x.shape)
        sqrt_recip_alphas_t = self._extract(buf.sqrt_recip_alphas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(buf.sqrt_one_minus_alphas_cumprod, t, x.shape)

        eps_theta = self.forward(x, t)
        model_mean = sqrt_recip_alphas_t * (x - betas_t * eps_theta / sqrt_one_minus_alphas_cumprod_t)

        posterior_variance_t = self._extract(buf.posterior_variance, t, x.shape)
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
            else torch.randn(num_samples, self.in_channels, self.sample_size, device=device)
        )

        # Ensure buffers are computed once on correct device/dtype.
        _ = self._get_schedule_buffers(device=x.device, dtype=x.dtype)

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
    "TimeSeriesDDPMBilevel",
    "PiecewiseLinearLogSNRSchedule",
    "make_beta_schedule",
    "cosine_beta_schedule",
    "sigmoid_beta_schedule",
]
