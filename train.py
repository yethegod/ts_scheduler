"""Train a simple TimeSeriesDDPM on synthetic sine waves and report sliced Wasserstein distance."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from evaluation import WassersteinDistances
from model import TimeSeriesDDPM


def make_sine_dataset(
    num_samples: int,
    sample_size: int,
    amplitude_range: Tuple[float, float] = (1.0, 1.0),
    freq_range: Tuple[float, float] = (1.0, 1.0),
    phase_range: Tuple[float, float] = (0.0, 2 * math.pi),
    noise_std: float = 0.0,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Create a batch of sine waves of shape (num_samples, 1, sample_size)."""
    if device is None:
        device = torch.device("cpu")
    t = torch.linspace(0, 2 * math.pi, sample_size, device=device, dtype=torch.float32)
    t = t.unsqueeze(0).repeat(num_samples, 1)

    amp = torch.empty(num_samples, 1, device=device).uniform_(*amplitude_range)
    freq = torch.empty(num_samples, 1, device=device).uniform_(*freq_range)
    phase = torch.empty(num_samples, 1, device=device).uniform_(*phase_range)

    waves = amp * torch.sin(freq * t + phase)
    if noise_std > 0:
        waves = waves + torch.randn_like(waves) * noise_std
    return waves.unsqueeze(1)


def get_dataloader(
    data: torch.Tensor, batch_size: int, num_workers: int = 0
) -> DataLoader:
    dataset = TensorDataset(data)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DDPM on sine curves.")
    parser.add_argument("--num-samples", type=int, default=2048, help="Number of training samples.")
    parser.add_argument("--eval-samples", type=int, default=512, help="Number of samples to draw for evaluation.")
    parser.add_argument("--sample-size", type=int, default=128, help="Number of points per sine wave.")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size.")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs.")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate.")
    parser.add_argument("--d-model", type=int, default=256, help="Hidden size of the MLP backbone.")
    parser.add_argument(
        "--d-mlp",
        type=int,
        default=None,
        help="Hidden size of the MLP feedforward; default is 2 * d_model.",
    )
    parser.add_argument("--num-layers", type=int, default=6, help="Number of FiLM MLP blocks.")
    parser.add_argument("--dropout", type=float, default=0.05, help="Dropout rate in MLP blocks.")
    parser.add_argument("--time-embed-dim", type=int, default=None, help="Timestep embedding dimension.")
    parser.add_argument(
        "--diffusion-steps", type=int, default=400, help="Number of diffusion timesteps for training."
    )
    parser.add_argument("--noise-std", type=float, default=0.0, help="Gaussian noise added to sine data.")
    parser.add_argument(
        "--amplitude-range",
        type=float,
        nargs=2,
        default=[1.0, 1.0],
        help="Uniform range for sine amplitudes.",
    )
    parser.add_argument(
        "--freq-range",
        type=float,
        nargs=2,
        default=[1.0, 1.0],
        help="Uniform range for sine frequencies.",
    )
    parser.add_argument(
        "--phase-range",
        type=float,
        nargs=2,
        default=[0.0, 2 * math.pi],
        help="Uniform range for sine phases.",
    )
    parser.add_argument("--num-workers", type=int, default=0, help="Dataloader workers.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA is available.")
    parser.add_argument(
        "--swd-projections",
        type=int,
        default=128,
        help="Number of projections for sliced Wasserstein distance.",
    )
    parser.add_argument(
        "--normalisation",
        type=str,
        default="none",
        choices=["none", "standardise"],
        help="Normalisation mode for Wasserstein computation.",
    )
    parser.add_argument(
        "--metric-path",
        type=str,
        default="swd_metrics.json",
        help="Where to write the sliced Wasserstein metrics.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Optional path to save the trained model state_dict.",
    )
    return parser.parse_args()


def train_epoch(
    model: TimeSeriesDDPM, dataloader: DataLoader, optim: torch.optim.Optimizer, device: torch.device
) -> float:
    model.train()
    total_loss = 0.0
    total_samples = 0
    for (batch,) in dataloader:
        batch = batch.to(device)
        loss = model.training_loss(batch)
        optim.zero_grad()
        loss.backward()
        optim.step()
        total_loss += loss.item() * batch.size(0)
        total_samples += batch.size(0)
    return total_loss / max(1, total_samples)


def evaluate_swd(
    real: torch.Tensor,
    generated: torch.Tensor,
    num_projections: int,
    normalisation: str,
    seed: int,
    metric_path: str | None,
) -> dict[str, float]:
    real_np = real.squeeze(1).cpu().numpy()
    gen_np = generated.squeeze(1).cpu().numpy()
    distances = WassersteinDistances(
        original_data=real_np, other_data=gen_np, normalisation=normalisation, seed=seed
    ).sliced_distances(num_projections)

    metrics = {
        "swd_mean": float(np.mean(distances)),
        "swd_std": float(np.std(distances)),
        "swd_median": float(np.median(distances)),
        "num_projections": int(num_projections),
    }
    print(
        f"Sliced Wasserstein: mean={metrics['swd_mean']:.4f}, "
        f"std={metrics['swd_std']:.4f}, median={metrics['swd_median']:.4f}"
    )

    if metric_path:
        path = Path(metric_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
    return metrics


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Using device: {device}")

    data_device = torch.device("cpu")  # keep dataset on CPU; move per-batch during training
    data = make_sine_dataset(
        num_samples=args.num_samples,
        sample_size=args.sample_size,
        amplitude_range=tuple(args.amplitude_range),
        freq_range=tuple(args.freq_range),
        phase_range=tuple(args.phase_range),
        noise_std=args.noise_std,
        device=data_device,
    )
    dataloader = get_dataloader(data, batch_size=args.batch_size, num_workers=args.num_workers)

    model = TimeSeriesDDPM(
        sample_size=args.sample_size,
        in_channels=1,
        num_train_timesteps=args.diffusion_steps,
        d_model=args.d_model,
        d_mlp=args.d_mlp,
        num_layers=args.num_layers,
        dropout=args.dropout,
        time_embed_dim=args.time_embed_dim,
    ).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        avg_loss = train_epoch(model, dataloader, optim, device)
        print(f"Epoch {epoch:03d}/{args.epochs} - loss: {avg_loss:.4f}")

    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Saved checkpoint to {checkpoint_path}")

    model.eval()
    with torch.no_grad():
        generated = model.sample(num_samples=args.eval_samples, device=device)
    _ = evaluate_swd(
        real=data[: args.eval_samples],
        generated=generated,
        num_projections=args.swd_projections,
        normalisation=args.normalisation,
        seed=args.seed,
        metric_path=args.metric_path,
    )


if __name__ == "__main__":
    main()
