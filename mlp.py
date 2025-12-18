import math
import torch
import torch.nn as nn

class _ModelOutput:
    def __init__(self, sample: torch.Tensor):
        self.sample = sample

def sinusoidal_timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    device = t.device
    half = dim // 2
    if half == 0:
        return torch.zeros((t.shape[0], dim), device=device, dtype=torch.float32)

    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(half, device=device, dtype=torch.float32)
        / max(half-1, 1)
    )
    args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
    if dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros((t.shape[0], 1), device=device)], dim=1)
    return emb

class FiLMResMLPBlock(nn.Module):
    def __init__(self, d_model: int, d_mlp: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_mlp),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_mlp, d_model),
        )

    def forward(self, h: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        x = self.norm(h)
        x = x * (1.0 + gamma) + beta
        return h + self.mlp(x)

class MLP1DModel(nn.Module):
    def __init__(
        self,
        sample_size: int,
        in_channels: int,
        out_channels: int,
        d_model: int = 256,
        d_mlp: int = 512,
        num_layers: int = 4,
        dropout: float = 0.0,
        time_embed_dim: int | None = None,
    ):
        super().__init__()
        self.sample_size = sample_size
        self.in_channels = in_channels
        self.out_channels = out_channels

        flat_in = sample_size * in_channels
        flat_out = sample_size * out_channels

        self.embed = nn.Linear(flat_in, d_model)
        self.unembed = nn.Linear(d_model, flat_out)

        te_dim = time_embed_dim or d_model
        self.te_dim = te_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(te_dim, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

        # shared FiLM generator (simple and strong enough)
        self.to_film = nn.ModuleList([nn.Linear(d_model, 2 * d_model) for _ in range(num_layers)])

        self.blocks = nn.ModuleList([
            FiLMResMLPBlock(d_model, d_mlp, dropout) for _ in range(num_layers)
        ])

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor):
        b, c, l = x.shape
        assert c == self.in_channels and l == self.sample_size

        h = x.reshape(b, c * l)
        h = self.embed(h)  # [B, d_model]

        te = sinusoidal_timestep_embedding(timesteps, self.te_dim)  # [B, te_dim]
        te = self.time_mlp(te)  # [B, d_model]

        # for blk in self.blocks:
        #   film = self.to_film(te)               # [B, 2*d_model]
        #    gamma, beta = film.chunk(2, dim=1)    # [B, d_model] each
        #    h = blk(h, gamma, beta)
        
        for i, blk in enumerate(self.blocks):
            gamma, beta = self.to_film[i](te).chunk(2, dim=1)
            h = blk(h, gamma, beta)

        out = self.unembed(h).reshape(b, self.out_channels, l)
        return _ModelOutput(sample=out)