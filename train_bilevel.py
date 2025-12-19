#!/usr/bin/env python3
"""
train_bilevel.py

Bilevel (SPSA) learning of a diffusion noise schedule on synthetic RBF-kernel GP time series,
with an optional cosine baseline for comparison.

This script is intentionally simple and matches the methodology described in the report:
- Data: fixed-length sequences sampled from a zero-mean GP with RBF kernel, controlled by length_scale.
- Model: DDPM with a FiLM-conditioned 1D MLP denoiser.
- Baseline: fixed cosine beta schedule (optional).
- Bilevel: outer-loop SPSA on a learnable schedule (piecewise-linear monotone logSNR),
          using truncated inner-loop fine-tuning from a shared warm-start checkpoint.

Outputs:
- metrics JSON (per length_scale)
- checkpoints for cosine and bilevel models
- learned schedule parameters

Example:
  python train_bilevel.py --length-scales 8 2 --mode both --device cuda

Notes:
- Keep data on CPU; move batches to GPU inside the training loop.
- For fast sanity runs, reduce --outer-iters and --inner-epochs and --eval-samples.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from model import TimeSeriesDDPM              # cosine baseline (fixed schedule)
from model_bilevel import TimeSeriesDDPMBilevel  # learnable schedule + SPSA outer-loop


# -----------------------------
# Synthetic RBF-GP dataset
# -----------------------------

def rbf_covariance_matrix(L: int, length_scale: float, signal_variance: float = 1.0, noise_variance: float = 1e-6) -> np.ndarray:
    """RBF kernel covariance on an evenly spaced grid 0..L-1."""
    t = np.arange(L, dtype=np.float64)
    d2 = (t[:, None] - t[None, :]) ** 2
    K = signal_variance * np.exp(-0.5 * d2 / (length_scale ** 2))
    K = K + noise_variance * np.eye(L, dtype=np.float64)
    return K

def sample_gp_sequences(
    n: int,
    L: int,
    length_scale: float,
    signal_variance: float = 1.0,
    noise_variance: float = 1e-6,
    rng: np.random.Generator | None = None,
) -> torch.Tensor:
    """Return tensor of shape [n, 1, L] sampled from GP(0, K)."""
    rng = rng or np.random.default_rng(0)
    K = rbf_covariance_matrix(L, length_scale, signal_variance, noise_variance)
    # Cholesky once, then sample: x = z @ chol^T  (z~N(0,I))
    chol = np.linalg.cholesky(K)  # [L, L]
    z = rng.standard_normal(size=(n, L)).astype(np.float64)
    x = z @ chol.T
    x = x.astype(np.float32)
    return torch.from_numpy(x).unsqueeze(1)  # [n, 1, L]

def make_loaders(
    X_train: torch.Tensor,
    X_val: torch.Tensor,
    batch_size: int,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> Tuple[DataLoader, DataLoader]:
    train_loader = DataLoader(
        TensorDataset(X_train),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
    val_loader = DataLoader(
        TensorDataset(X_val),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    return train_loader, val_loader


# -----------------------------
# Metrics: Corr + Rel Frobenius, and Time-domain SWD
# -----------------------------

@torch.no_grad()
def correlation_matrix(X: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    X: [N, 1, L] or [N, L]
    Returns: [L, L] correlation across time indices, treating each time index as a random variable across samples.
    """
    if X.ndim == 3:
        X = X.squeeze(1)
    X = X.float()
    X = X - X.mean(dim=0, keepdim=True)
    std = X.std(dim=0, keepdim=True).clamp_min(eps)
    Xn = X / std
    # Corr = (Xn^T Xn) / (N-1)
    N = Xn.shape[0]
    return (Xn.T @ Xn) / max(N - 1, 1)

@torch.no_grad()
def rel_frob_corr(real: torch.Tensor, gen: torch.Tensor, eps: float = 1e-8) -> float:
    Rr = correlation_matrix(real, eps=eps)
    Rg = correlation_matrix(gen, eps=eps)
    num = torch.linalg.norm(Rg - Rr, ord="fro")
    den = torch.linalg.norm(Rr, ord="fro").clamp_min(eps)
    return float((num / den).item())

@torch.no_grad()
def sliced_wasserstein_time(
    real: torch.Tensor,
    gen: torch.Tensor,
    num_projections: int = 128,
    seed: int = 0,
) -> float:
    """
    Time-domain SWD: treat each length-L sequence as a point in R^L.
    Uses random Gaussian directions + sorting for 1D W1.
    Returns average W1 over projections.
    """
    if real.ndim == 3:
        real = real.squeeze(1)
    if gen.ndim == 3:
        gen = gen.squeeze(1)
    real = real.float().cpu()
    gen = gen.float().cpu()
    N, L = real.shape
    M, L2 = gen.shape
    assert L == L2

    g = torch.Generator(device="cpu")
    g.manual_seed(seed)

    # random directions
    dirs = torch.randn((num_projections, L), generator=g)
    dirs = dirs / (dirs.norm(dim=1, keepdim=True) + 1e-12)

    # projections: [K, N] and [K, M]
    pr = dirs @ real.T
    pg = dirs @ gen.T
    pr, _ = pr.sort(dim=1)
    pg, _ = pg.sort(dim=1)

    # match quantiles (assume N==M for simplest; otherwise interpolate)
    if N != M:
        # interpolate to common grid of size min(N,M)
        K = min(N, M)
        pr = pr[:, torch.linspace(0, N - 1, K).round().long()]
        pg = pg[:, torch.linspace(0, M - 1, K).round().long()]

    w1 = (pr - pg).abs().mean(dim=1)  # mean over samples for each projection
    return float(w1.mean().item())


# -----------------------------
# Training utilities
# -----------------------------

def set_seeds(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def train_epochs_ddpm(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    lr: float,
    epochs: int,
    grad_clip: float | None = 1.0,
    params: Iterable[torch.nn.Parameter] | None = None,
) -> float:
    """Train for `epochs` and return last epoch avg loss."""
    if params is None:
        params = model.parameters()
    params = list(params)
    if not params:
        raise ValueError("No parameters provided for optimization.")
    opt = torch.optim.Adam(params, lr=lr)
    model.train()
    last = 0.0
    for _ in range(epochs):
        total, n = 0.0, 0
        for (batch,) in loader:
            x0 = batch.to(device, non_blocking=True)
            loss = model.training_loss(x0)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            total += float(loss.item()) * x0.size(0)
            n += x0.size(0)
        last = total / max(n, 1)
    return last

@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    real_eval: torch.Tensor,
    device: torch.device,
    num_gen: int,
    swd_projections: int,
    seed: int,
) -> Dict[str, float]:
    model.eval()
    gen = model.sample(num_samples=num_gen, device=device).cpu()

    metrics = {
        "rel_frob_corr": rel_frob_corr(real_eval.cpu(), gen),
        "swd_time": sliced_wasserstein_time(real_eval.cpu(), gen, num_projections=swd_projections, seed=seed),
    }
    return metrics


# -----------------------------
# Bilevel SPSA outer loop
# -----------------------------

@dataclass
class SPSAConfig:
    outer_iters: int = 30
    delta: float = 0.05         # perturbation scale
    lr_outer: float = 0.2       # schedule update step size
    inner_epochs: int = 5       # truncated inner fine-tuning epochs (per perturbation)
    num_gen: int = 512
    swd_projections: int = 128
    outer_objective: str = "rel_frob"  # "rel_frob" | "swd" | "combo"
    combo_lambda: float = 1.0

def objective_from_metrics(m: Dict[str, float], cfg: SPSAConfig) -> float:
    if cfg.outer_objective == "rel_frob":
        return float(m["rel_frob_corr"])
    if cfg.outer_objective == "swd":
        return float(m["swd_time"])
    if cfg.outer_objective == "combo":
        return float(m["rel_frob_corr"] + cfg.combo_lambda * m["swd_time"])
    raise ValueError(f"Unknown outer_objective: {cfg.outer_objective}")

def spsa_bilevel_learn_schedule(
    length_scale: float,
    train_loader: DataLoader,
    real_eval: torch.Tensor,
    device: torch.device,
    args: argparse.Namespace,
    out_dir: Path,
) -> Dict[str, object]:
    """
    Outer loop:
      maintain schedule q (inside model_bilevel)
      for n in 1..N_outer:
        sample u in {+1,-1}^d
        evaluate f(q+delta u), f(q-delta u) using truncated fine-tuning from shared theta_anchor
        ghat = (f+ - f-) / (2 delta) * u
        q <- q - lr_outer * ghat
        (optional) update theta_anchor by training under new q for anchor_epochs (we use inner_epochs)
    """
    cfg = SPSAConfig(
        outer_iters=args.outer_iters,
        delta=args.spsa_delta,
        lr_outer=args.lr_outer,
        inner_epochs=args.inner_epochs,
        num_gen=args.eval_samples,
        swd_projections=args.swd_projections,
        outer_objective=args.outer_objective,
        combo_lambda=args.combo_lambda,
    )

    # Build bilevel model
    model = TimeSeriesDDPMBilevel(
        sample_size=args.sample_size,
        in_channels=1,
        num_train_timesteps=args.diffusion_steps,
        d_model=args.d_model,
        d_mlp=args.d_mlp,
        num_layers=args.num_layers,
        dropout=args.dropout,
        schedule_num_knots=args.schedule_knots,
        schedule_init_from=args.schedule_init_from,
    ).to(device)

    # Warm-start checkpoint (shared theta_anchor)
    if args.warmstart_ckpt and Path(args.warmstart_ckpt).exists():
        ckpt = torch.load(args.warmstart_ckpt, map_location="cpu")
        model.load_state_dict(ckpt, strict=True)
        print(f"[ls={length_scale}] Loaded warm-start checkpoint: {args.warmstart_ckpt}")
    else:
        # Optional quick pretrain under initial schedule
        pre = args.pretrain_epochs
        if pre > 0:
            print(f"[ls={length_scale}] Pretraining anchor for {pre} epochs (initial schedule={args.schedule_init_from})...")
            train_epochs_ddpm(
                model,
                train_loader,
                device,
                lr=args.lr_inner,
                epochs=pre,
                grad_clip=args.grad_clip,
                params=model.unet.parameters(),
            )

    theta_anchor = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    # Outer loop log
    history: List[Dict[str, float]] = []
    d = model.get_schedule_vector().numel()
    rng = np.random.default_rng(args.seed + int(1000 * length_scale))

    # Evaluate initial
    init_metrics = evaluate_model(model, real_eval, device, num_gen=cfg.num_gen, swd_projections=cfg.swd_projections, seed=args.seed)
    init_obj = objective_from_metrics(init_metrics, cfg)
    print(f"[ls={length_scale}] init metrics: {init_metrics} | obj={init_obj:.6f}")

    for it in range(1, cfg.outer_iters + 1):
        u = torch.from_numpy(rng.choice([-1.0, 1.0], size=(d,)).astype(np.float32))
        q = model.get_schedule_vector().detach().cpu()

        q_plus = q + cfg.delta * u
        q_minus = q - cfg.delta * u

        # f(q_plus)
        model.load_state_dict(theta_anchor, strict=True)
        model.set_schedule_vector_(q_plus, project=True)
        train_epochs_ddpm(
            model,
            train_loader,
            device,
            lr=args.lr_inner,
            epochs=cfg.inner_epochs,
            grad_clip=args.grad_clip,
            params=model.unet.parameters(),
        )
        m_plus = evaluate_model(model, real_eval, device, num_gen=cfg.num_gen, swd_projections=cfg.swd_projections, seed=args.seed + it)
        f_plus = objective_from_metrics(m_plus, cfg)

        # f(q_minus)
        model.load_state_dict(theta_anchor, strict=True)
        model.set_schedule_vector_(q_minus, project=True)
        train_epochs_ddpm(
            model,
            train_loader,
            device,
            lr=args.lr_inner,
            epochs=cfg.inner_epochs,
            grad_clip=args.grad_clip,
            params=model.unet.parameters(),
        )
        m_minus = evaluate_model(model, real_eval, device, num_gen=cfg.num_gen, swd_projections=cfg.swd_projections, seed=args.seed + 1234 + it)
        f_minus = objective_from_metrics(m_minus, cfg)

        # SPSA gradient estimate
        ghat = (f_plus - f_minus) / (2.0 * cfg.delta) * u  # shape [d]
        q_new = q - cfg.lr_outer * ghat

        # Apply update and refresh schedule
        model.load_state_dict(theta_anchor, strict=True)
        model.set_schedule_vector_(q_new, project=True)

        # Optionally update anchor (keeps theta_anchor in sync with schedule)
        if args.anchor_update:
            train_epochs_ddpm(
                model,
                train_loader,
                device,
                lr=args.lr_inner,
                epochs=cfg.inner_epochs,
                grad_clip=args.grad_clip,
                params=model.unet.parameters(),
            )
            theta_anchor = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        # Evaluate current schedule quickly (on current model; if anchor_update=False, this is just theta_anchor)
        cur_metrics = evaluate_model(model, real_eval, device, num_gen=cfg.num_gen, swd_projections=cfg.swd_projections, seed=args.seed + 9999 + it)
        cur_obj = objective_from_metrics(cur_metrics, cfg)

        row = {
            "iter": float(it),
            "f_plus": float(f_plus),
            "f_minus": float(f_minus),
            "obj": float(cur_obj),
            "rel_frob_corr": float(cur_metrics["rel_frob_corr"]),
            "swd_time": float(cur_metrics["swd_time"]),
        }
        history.append(row)
        print(f"[ls={length_scale}] it {it:03d}/{cfg.outer_iters} | f+= {f_plus:.5f} f-= {f_minus:.5f} | obj={cur_obj:.5f} | "
              f"RelFrob={cur_metrics['rel_frob_corr']:.5f} SWD={cur_metrics['swd_time']:.5f}")

    # Save
    out_dir.mkdir(parents=True, exist_ok=True)
    schedule_vec = model.get_schedule_vector().detach().cpu().numpy().tolist()
    ckpt_path = out_dir / f"bilevel_ls{length_scale:g}.pt"
    torch.save(model.state_dict(), ckpt_path)

    with (out_dir / f"bilevel_history_ls{length_scale:g}.json").open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    with (out_dir / f"bilevel_schedule_vec_ls{length_scale:g}.json").open("w", encoding="utf-8") as f:
        json.dump({"schedule_vec": schedule_vec}, f, indent=2)

    final_metrics = evaluate_model(model, real_eval, device, num_gen=cfg.num_gen, swd_projections=cfg.swd_projections, seed=args.seed + 424242)
    return {
        "final_metrics": final_metrics,
        "checkpoint": str(ckpt_path),
        "history_path": str(out_dir / f"bilevel_history_ls{length_scale:g}.json"),
        "schedule_path": str(out_dir / f"bilevel_schedule_vec_ls{length_scale:g}.json"),
    }


# -----------------------------
# Cosine baseline training
# -----------------------------

def train_cosine_baseline(
    length_scale: float,
    train_loader: DataLoader,
    real_eval: torch.Tensor,
    device: torch.device,
    args: argparse.Namespace,
    out_dir: Path,
) -> Dict[str, object]:
    model = TimeSeriesDDPM(
        sample_size=args.sample_size,
        in_channels=1,
        beta_schedule="cosine",
        num_train_timesteps=args.diffusion_steps,
        dropout=args.dropout,
        base_channels=args.d_model,  # maps to d_model in your model.py wrapper
        channel_mults=(1, 2, 4, 8),  # ignored in MLP-backed version; kept for compatibility
        layers_per_block=2,          # ignored
    ).to(device)

    # If your model.py is the MLP-backed version, you should pass d_model/d_mlp/num_layers explicitly.
    # If your model.py is still the diffusers UNet version, the above is fine.
    # If you want to force the MLP backbone baseline, replace with your MLP-based TimeSeriesDDPM (fixed cosine schedule).

    print(f"[ls={length_scale}] Training cosine baseline for {args.baseline_epochs} epochs...")
    train_epochs_ddpm(model, train_loader, device, lr=args.lr_inner, epochs=args.baseline_epochs, grad_clip=args.grad_clip)

    metrics = evaluate_model(model, real_eval, device, num_gen=args.eval_samples, swd_projections=args.swd_projections, seed=args.seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / f"cosine_ls{length_scale:g}.pt"
    torch.save(model.state_dict(), ckpt_path)
    with (out_dir / f"cosine_metrics_ls{length_scale:g}.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    return {"metrics": metrics, "checkpoint": str(ckpt_path)}


# -----------------------------
# CLI
# -----------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", type=str, default="both", choices=["bilevel", "cosine", "both"])
    p.add_argument("--length-scales", type=float, nargs="+", default=[8.0, 2.0])

    # Data
    p.add_argument("--num-train", type=int, default=2048)
    p.add_argument("--num-val", type=int, default=512)
    p.add_argument("--sample-size", type=int, default=256)
    p.add_argument("--signal-variance", type=float, default=1.0)
    p.add_argument("--noise-variance", type=float, default=1e-6)

    # Model / diffusion
    p.add_argument("--diffusion-steps", type=int, default=400)
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--d-mlp", type=int, default=512)
    p.add_argument("--num-layers", type=int, default=6)
    p.add_argument("--dropout", type=float, default=0.05)

    # Training
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--lr-inner", type=float, default=2e-4)
    p.add_argument("--grad-clip", type=float, default=1.0)

    # Baseline
    p.add_argument("--baseline-epochs", type=int, default=200)

    # Bilevel / SPSA
    p.add_argument("--outer-iters", type=int, default=30)
    p.add_argument("--inner-epochs", type=int, default=5)
    p.add_argument("--spsa-delta", type=float, default=0.05)
    p.add_argument("--lr-outer", type=float, default=0.2)
    p.add_argument("--schedule-knots", type=int, default=16)
    p.add_argument("--schedule-init-from", type=str, default="cosine", choices=["cosine", "sigmoid", "linear"])
    p.add_argument("--warmstart-ckpt", type=str, default=None)
    p.add_argument("--pretrain-epochs", type=int, default=0, help="If no warmstart_ckpt, optionally pretrain anchor for a few epochs.")
    p.add_argument("--anchor-update", action="store_true", help="Update theta_anchor each outer iter by training under updated schedule.")

    # Outer objective
    p.add_argument("--outer-objective", type=str, default="rel_frob", choices=["rel_frob", "swd", "combo"])
    p.add_argument("--combo-lambda", type=float, default=1.0)

    # Eval
    p.add_argument("--eval-samples", type=int, default=512)
    p.add_argument("--swd-projections", type=int, default=128)

    # System
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda", "mps"])
    p.add_argument("--out-dir", type=str, default="runs_bilevel")
    return p.parse_args()

def main() -> None:
    args = parse_args()
    set_seeds(args.seed)

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available; falling back to CPU.")
        device = torch.device("cpu")
    elif args.device == "mps" and not torch.backends.mps.is_available():
        print("MPS not available; falling back to CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    pin_memory = (device.type == "cuda")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results: Dict[str, Dict[str, object]] = {}

    for ls in args.length_scales:
        rng = np.random.default_rng(args.seed + int(1000 * ls))
        X_train = sample_gp_sequences(
            n=args.num_train,
            L=args.sample_size,
            length_scale=ls,
            signal_variance=args.signal_variance,
            noise_variance=args.noise_variance,
            rng=rng,
        )
        X_val = sample_gp_sequences(
            n=args.num_val,
            L=args.sample_size,
            length_scale=ls,
            signal_variance=args.signal_variance,
            noise_variance=args.noise_variance,
            rng=rng,
        )
        train_loader, _ = make_loaders(X_train, X_val, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=pin_memory)
        real_eval = X_val[: args.eval_samples].clone()

        print(f"\n===== length_scale={ls:g} | train={len(X_train)} val={len(X_val)} L={args.sample_size} =====")

        ls_dir = out_dir / f"ls{ls:g}"
        ls_dir.mkdir(parents=True, exist_ok=True)
        results[str(ls)] = {}

        if args.mode in ("cosine", "both"):
            cosine_res = train_cosine_baseline(ls, train_loader, real_eval, device, args, ls_dir)
            results[str(ls)]["cosine"] = cosine_res
# NOTE: If your `model.py` is already the MLP-backed version, you may want to implement
# a fixed-cosine schedule baseline using the same MLP denoiser for a fair comparison.
# The current `train_cosine_baseline` uses `TimeSeriesDDPM` from model.py.

        if args.mode in ("bilevel", "both"):
            bilevel_res = spsa_bilevel_learn_schedule(ls, train_loader, real_eval, device, args, ls_dir)
            results[str(ls)]["bilevel"] = bilevel_res

        # Write per-length-scale summary
        with (ls_dir / "summary.json").open("w", encoding="utf-8") as f:
            json.dump(results[str(ls)], f, indent=2)

    # Global summary
    with (out_dir / "summary_all.json").open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\nDone. Wrote results to:", out_dir.resolve())

if __name__ == "__main__":
    main()
