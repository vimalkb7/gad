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
    Â = D^{-1/2} (A + I) D^{-1/2} (if add_self_loops True).
    """
    if add_self_loops:
        A = A + torch.eye(A.size(-1), device=A.device, dtype=A.dtype)
    deg = A.sum(-1)
    inv_sqrt_deg = (deg + eps).pow(-0.5)
    Dm12 = torch.diag(inv_sqrt_deg)
    return Dm12 @ A @ Dm12


# ---------- New: Laplacians & operator builder ----------

def laplacian_unnormalized(A: torch.Tensor) -> torch.Tensor:
    D = torch.diag(A.sum(-1))
    return D - A

def laplacian_symmetric(A: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    """
    L_sym = I - D^{-1/2} A D^{-1/2}
    """
    deg = A.sum(-1)
    inv_sqrt_deg = (deg + eps).pow(-0.5)
    Dm12 = torch.diag(inv_sqrt_deg)
    I = torch.eye(A.size(-1), dtype=A.dtype, device=A.device)
    return I - Dm12 @ A @ Dm12

def _normalize_operator_name(kind: str) -> str:
    k = kind.lower()
    aliases = {
        "adj": "adjacency",
        "norm_adj": "normalized_adjacency",
        "laplacian_sym": "laplacian_sym",
        "lgamma": "laplacian_gamma",
        "lgamma_sym": "laplacian_gamma_sym",
    }
    return aliases.get(k, k)

def build_graph_operator(
    A: torch.Tensor,
    operator_kind: str,
    *,
    gamma: float = 0.0,
    add_self_loops: bool = True,
    eps: float = 1e-9,
) -> torch.Tensor:
    """
    Returns an [N,N] operator among:
      'adjacency', 'normalized_adjacency',
      'laplacian', 'laplacian_sym',
      'laplacian_gamma', 'laplacian_gamma_sym'
    """
    kind = _normalize_operator_name(operator_kind)

    if kind == "adjacency":
        return A

    if kind == "normalized_adjacency":
        return normalize_adj(A, add_self_loops=add_self_loops, eps=eps)

    if kind == "laplacian":
        return laplacian_unnormalized(A)

    if kind == "laplacian_sym":
        return laplacian_symmetric(A, eps=eps)

    if kind == "laplacian_gamma":
        L = laplacian_unnormalized(A)
        N = A.size(-1)
        return L + float(gamma) * torch.eye(N, dtype=A.dtype, device=A.device)

    if kind == "laplacian_gamma_sym":
        Ls = laplacian_symmetric(A, eps=eps)
        N = A.size(-1)
        return Ls + float(gamma) * torch.eye(N, dtype=A.dtype, device=A.device)

    raise ValueError(
        f"Unknown operator_kind '{operator_kind}'. "
        "Use one of: 'adjacency','normalized_adjacency','laplacian',"
        "'laplacian_sym','laplacian_gamma','laplacian_gamma_sym' "
        "(aliases: adj, norm_adj, Lgamma, Lgamma_sym)."
    )


# -----------------------
# Graph layers & blocks
# -----------------------

class GCNConv(nn.Module):
    """
    Simple GCN layer: H' = M H W  where M is the chosen operator.
    """
    def __init__(self, in_dim: int, out_dim: int, bias: bool = True):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim, bias=bias)

    def forward(self, X: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
        return M @ self.lin(X)


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
            cond = cond.unsqueeze(0)

        B = x.size(0) if x.dim() == 3 else 1
        if cond.size(0) == 1 and B > 1:
            cond = cond.expand(B, -1)   # [B, cond_dim]

        g = self.gamma(cond)    # [B, H] or [1, H]
        b = self.beta(cond)     # [B, H] or [1, H]

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

    def forward(self, X: torch.Tensor, M: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.norm1(X)
        h = self.conv1(h, M)
        h = self.film1(h, t_emb)
        h = F.gelu(h)
        h = self.dropout(h)
        X = X + h

        h = self.norm2(X)
        h = self.conv2(h, M)
        h = self.film2(h, t_emb)
        h = F.gelu(h)
        h = self.dropout(h)
        X = X + h
        return X


# -----------------------
# Denoiser model
# -----------------------

class GraphEpsDenoiser(nn.Module):
    """
    ε_θ(x_t, t; A, choice) with user-selectable graph operator.

    operator_kind ∈ {
      'adjacency','normalized_adjacency',
      'laplacian','laplacian_sym',
      'laplacian_gamma','laplacian_gamma_sym'
      (aliases: 'adj','norm_adj','Lgamma','Lgamma_sym')
    }
    """
    def __init__(
        self,
        in_dim: int,
        hidden: int = 128,
        depth: int = 6,
        time_dim: int = 128,
        dropout: float = 0.0,
        add_self_loops: bool = True,        # only impacts normalized_adjacency
        *,
        operator_kind: str = "adjacency",
        gamma: float = 0.0,
        eps: float = 1e-9,
    ):
        super().__init__()
        self.add_self_loops = add_self_loops
        self.operator_kind = operator_kind
        self.gamma = float(gamma)
        self.eps = float(eps)

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

        X = X.to(dtype=model_dtype, device=device)
        A = A.to(dtype=model_dtype, device=device)

        # Build the selected operator
        M = build_graph_operator(
            A,
            operator_kind=self.operator_kind,
            gamma=self.gamma,
            add_self_loops=self.add_self_loops,
            eps=self.eps,
        )

        # Time embedding
        if not torch.is_tensor(t):
            t = torch.tensor(t, device=device)
        else:
            t = t.to(device=device)
        t = t.to(dtype=model_dtype, device=device)

        t_emb = sinusoidal_timestep_embedding(t.flatten(), self.time_dim)
        t_emb = self.t_mlp(t_emb)

        # Encode -> residual graph blocks -> project to ε
        h = self.in_proj(X)
        for blk in self.blocks:
            h = blk(h, M, t_emb)
        h = self.out_norm(h)
        return self.out_proj(h)
