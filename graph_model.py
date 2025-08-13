from typing import Optional
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------
# Utilities
# -----------------------

def sinusoidal_timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Standard transformer/DDPM sinusoidal embedding.
    t: shape [] or [B] or [N] (will be flattened and broadcast as needed)
    returns: [t.shape[0], dim]
    """
    if t.dim() == 0:
        t = t[None]
    dtype = t.dtype
    device = t.device

    half = dim // 2
    freq = torch.exp(-math.log(10000) * torch.arange(half, device=device, dtype=dtype) / max(1, half))
    args = t.to(dtype=dtype)[:, None] * freq[None, :]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0,1))
    return emb


def normalize_adj(A: torch.Tensor, add_self_loops: bool = True, eps: float = 1e-9) -> torch.Tensor:
    """
    Symmetric normalization  A_hat = D^{-1/2} (A + I) D^{-1/2}.
    Works with dense A. If your graphs are large/sparse, port this to sparse ops.
    """
    if add_self_loops:
        A = A + torch.eye(A.size(-1), device=A.device, dtype=A.dtype)
    deg = A.sum(-1)  # [N]
    inv_sqrt_deg = (deg + eps).pow(-0.5)
    D_inv_sqrt = torch.diag(inv_sqrt_deg)
    return D_inv_sqrt @ A @ D_inv_sqrt


# -----------------------
# Graph layers & blocks
# -----------------------

class GCNConv(nn.Module):
    """
    Simple GCN layer: H' = A_hat H W + b
    """
    def __init__(self, in_dim: int, out_dim: int, bias: bool = True):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim, bias=bias)

    def forward(self, X: torch.Tensor, A_hat: torch.Tensor) -> torch.Tensor:
        # Supports X as [N,H] or [B,N,H]; (N,N) @ (B,N,H) broadcasts on leading dim
        return A_hat @ self.lin(X)


class FiLM(nn.Module):
    """
    Feature-wise Linear Modulation from a conditioning vector (here: time embedding).
    y = gamma * x + beta
    """
    def __init__(self, cond_dim: int, hidden_dim: int):
        super().__init__()
        self.gamma = nn.Linear(cond_dim, hidden_dim)
        self.beta  = nn.Linear(cond_dim, hidden_dim)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        x:    [N, H] or [B, N, H]
        cond: [1, cond_dim], [B, cond_dim], or [N, cond_dim] (broadcast sensibly)
        """
        if cond.dim() == 1:
            cond = cond.unsqueeze(0)  # [1, cond_dim]

        if x.dim() == 3:
            B = x.size(0)
        else:
            B = 1

        if cond.size(0) == 1 and B > 1:
            cond = cond.expand(B, -1)  # [B, cond_dim]

        g = self.gamma(cond)  # [B, H] or [1, H]
        b = self.beta(cond)   # [B, H] or [1, H]

        if x.dim() == 3:
            g = g.unsqueeze(1)  # [B,1,H] -> broadcast over N
            b = b.unsqueeze(1)
        else:
            g = g.unsqueeze(0)  # [1,H] -> broadcast over N
            b = b.unsqueeze(0)

        return g * x + b


class ResidualGCNBlock(nn.Module):
    """
    Pre-norm residual graph block with GCNConv + FiLM(time) + GELU.
    """
    def __init__(self, hidden_dim: int, time_dim: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.conv1 = GCNConv(hidden_dim, hidden_dim)
        self.film1 = FiLM(time_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.film2 = FiLM(time_dim, hidden_dim)

    def forward(self, X: torch.Tensor, A_hat: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        # Block 1
        h = self.norm1(X)
        h = self.conv1(h, A_hat)
        h = self.film1(h, t_emb)
        h = F.gelu(h)
        h = self.dropout(h)
        X = X + h  # residual

        # Block 2
        h = self.norm2(X)
        h = self.conv2(h, A_hat)
        h = self.film2(h, t_emb)
        h = F.gelu(h)
        h = self.dropout(h)
        X = X + h  # residual
        return X


# -----------------------
# Denoiser model
# -----------------------

class GraphEpsDenoiser(nn.Module):
    """
    Predicts ε_theta(x_t, t, A) from node features and graph structure.

    Args:
      in_dim:    input feature dimension (d_in)
      hidden:    width of hidden node embeddings
      depth:     number of residual graph blocks
      time_dim:  dimension of timestep embedding
      dropout:   dropout rate inside blocks
      add_self_loops: whether to add self-loops before normalization

    Forward:
      X: [N, d_in] or [B, N, d_in]
      A: [N, N]
      t: scalar int or tensor-like (or [B])
      returns ε_hat: [N, d_in] or [B, N, d_in] matching X
    """
    def __init__(
        self,
        in_dim: int,
        hidden: int = 128,
        depth: int = 6,
        time_dim: int = 128,
        dropout: float = 0.0,
        add_self_loops: bool = True
    ):
        super().__init__()
        self.add_self_loops = add_self_loops

        self.in_proj = nn.Linear(in_dim, hidden)
        self.time_dim = time_dim

        self.blocks = nn.ModuleList([
            ResidualGCNBlock(hidden_dim=hidden, time_dim=time_dim, dropout=dropout)
            for _ in range(depth)
        ])
        self.out_norm = nn.LayerNorm(hidden)
        self.out_proj = nn.Linear(hidden, in_dim)

        # Post-process timestep embedding (better than raw sinusoid)
        self.t_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

    def forward(self, X: torch.Tensor, A: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        model_dtype = next(self.parameters()).dtype
        device = next(self.parameters()).device

        # Cast inputs to the model dtype/device
        X = X.to(dtype=model_dtype, device=device)
        A = A.to(dtype=model_dtype, device=device)

        # 1) Normalize adjacency once per forward
        A_hat = normalize_adj(A, add_self_loops=self.add_self_loops, eps=1e-9)

        # 2) Timestep embedding to [1, time_dim] (FiLM will broadcast over batch/nodes)
        if not torch.is_tensor(t):
            t = torch.tensor(t, device=device)
        else:
            t = t.to(device=device)
        t = t.to(dtype=model_dtype, device=device)

        t_emb = sinusoidal_timestep_embedding(t.flatten(), self.time_dim)  # [B, time_dim] or [1, time_dim]
        t_emb = t_emb.mean(dim=0, keepdim=True)                            # [1, time_dim]
        t_emb = self.t_mlp(t_emb)                                          # [1, time_dim]

        # 3) Node encoder
        h = self.in_proj(X)

        # 4) Residual graph blocks with FiLM(t)
        for blk in self.blocks:
            h = blk(h, A_hat, t_emb)

        # 5) Project back to ε
        h = self.out_norm(h)
        eps_hat = self.out_proj(h)
        return eps_hat
