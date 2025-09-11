# mlp_model.py
from typing import Optional
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------
# Sinusoidal timestep embedding (unchanged)
# ---------------------------------------------------------------
def sinusoidal_timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Standard sinusoidal embedding. Accepts scalar or [B] time.
    Returns [B, dim] when given [B]; otherwise [1, dim].
    """
    if t.dim() == 0:
        t = t[None]
    dtype, device = t.dtype, t.device
    half = dim // 2
    freq = torch.exp(-math.log(10000) * torch.arange(half, device=device, dtype=dtype) / max(1, half))
    args = t.to(dtype=dtype)[:, None] * freq[None, :]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


# ---------------------------------------------------------------
# Simple residual block with timestep add-conditioning
# ---------------------------------------------------------------
class ResidualBlock(nn.Module):
    """
    LayerNorm -> (h + W_t(t_emb)) -> Linear -> GELU -> Linear -> residual.
    Time is injected additively (broadcasted across nodes) for stability & simplicity.
    """
    def __init__(self, hidden: int, time_dim: int, dropout: float = 0.0, expansion: int = 4):
        super().__init__()
        hmid = hidden * expansion
        self.norm = nn.LayerNorm(hidden)
        self.t_proj = nn.Linear(time_dim, hidden)
        self.fc1 = nn.Linear(hidden, hmid)
        self.fc2 = nn.Linear(hmid, hidden)
        self.drop = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        # t_emb: [B, time_dim] or [1, time_dim]; broadcast over nodes
        B = h.shape[0] if h.dim() == 3 else 1
        if t_emb.dim() == 1:
            t_emb = t_emb[None, :]  # [1, T]
        if B > 1 and t_emb.size(0) == 1:
            t_emb = t_emb.expand(B, -1)  # [B, T]

        h_in = h
        h = self.norm(h)
        t_add = self.t_proj(t_emb)  # [B, H]
        if h.dim() == 3:
            t_add = t_add.unsqueeze(1)  # [B, 1, H]
        else:
            t_add = t_add.unsqueeze(0)  # [1, 1, H] to broadcast
        h = h + t_add
        h = self.fc1(h)
        h = F.gelu(h)
        h = self.drop(h)
        h = self.fc2(h)
        return h_in + h


# ---------------------------------------------------------------
# GraphX0MLP — simple, x0-prediction MLP with time add-conditioning
# ---------------------------------------------------------------
class GraphX0MLP(nn.Module):
    """
    Predicts x0 directly from noisy x_k and timestep t.

    Interface:
      forward(X, t) -> x0_hat
        - X: [N, d_in] or [B, N, d_in]
        - t: scalar int/float or [B] Long/Float timesteps
    """
    def __init__(
        self,
        in_dim: int,
        hidden: int = 128,
        depth: int = 6,
        time_dim: int = 128,
        dropout: float = 0.0,
        expansion: int = 4,
    ):
        super().__init__()
        self.time_dim = int(time_dim)
        self.in_proj = nn.Linear(in_dim, hidden)
        self.blocks = nn.ModuleList([
            ResidualBlock(hidden=hidden, time_dim=self.time_dim, dropout=dropout, expansion=expansion)
            for _ in range(depth)
        ])
        self.out_norm = nn.LayerNorm(hidden)
        self.out_proj = nn.Linear(hidden, in_dim)

        # small time MLP to let the model learn a nicer conditioning space
        self.t_mlp = nn.Sequential(
            nn.Linear(self.time_dim, self.time_dim),
            nn.SiLU(),
            nn.Linear(self.time_dim, self.time_dim),
        )

    def forward(self, X: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Returns x0_hat with same leading dims as X.
        """
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype

        # Ensure shapes: [B,N,D]
        single = False
        if X.dim() == 2:  # [N, D]
            X = X.unsqueeze(0)
            single = True

        X = X.to(device=device, dtype=dtype)

        # Build time embedding: scalar or [B]
        if not torch.is_tensor(t):
            t = torch.tensor(t, device=device)
        else:
            t = t.to(device=device)
        t = t.to(dtype=dtype)
        if t.dim() == 0:
            t = t[None]  # [1]
        elif t.dim() > 1:
            t = t.view(-1)  # flatten to [B]

        t_emb = sinusoidal_timestep_embedding(t, self.time_dim)  # [B, T] or [1, T]
        t_emb = self.t_mlp(t_emb)

        # Node-wise MLP with time add-conditioning
        h = self.in_proj(X)
        for blk in self.blocks:
            h = blk(h, t_emb)
        h = self.out_norm(h)
        x0_hat = self.out_proj(h)

        if single:
            x0_hat = x0_hat.squeeze(0)  # back to [N, D]
        return x0_hat
