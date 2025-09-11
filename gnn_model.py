# gnn_model.py
from typing import Optional
import math, torch
import torch.nn as nn
import torch.nn.functional as F

def sinusoidal_timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    if t.dim() == 0: 
        t = t[None]
    if not t.is_floating_point():
        t = t.to(dtype=torch.get_default_dtype())
    half = dim // 2
    device, dtype = t.device, t.dtype
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=device, dtype=dtype) / max(1, half))
    args = t[:, None] * freqs[None, :]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb

def laplacian_sym(A: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    d = A.sum(-1)
    inv_sqrt = (d + eps).pow(-0.5)
    Dm12 = torch.diag(inv_sqrt)
    I = torch.eye(A.size(-1), dtype=A.dtype, device=A.device)
    return I - Dm12 @ A @ Dm12

@torch.no_grad()
def spectral_radius(M: torch.Tensor, iters: int = 8, eps: float = 1e-6) -> float:
    v = torch.randn(M.size(0), device=M.device, dtype=M.dtype)
    for _ in range(iters):
        v = M @ v; v = v / (v.norm() + eps)
    return float(torch.dot(v, M @ v).clamp_min(eps))

class ChebFilter(nn.Module):
    """Single Chebyshev filter: sum_{k=0..K} T_k(M~) X W_k."""
    def __init__(self, in_dim: int, out_dim: int, K: int = 2):
        super().__init__()
        self.K = int(K)
        self.W = nn.ParameterList([nn.Parameter(torch.empty(in_dim, out_dim)) for _ in range(self.K+1)])
        for w in self.W: nn.init.kaiming_uniform_(w, a=math.sqrt(5))
        self.b = nn.Parameter(torch.zeros(out_dim))

    def forward(self, X: torch.Tensor, M_tilde: torch.Tensor) -> torch.Tensor:
        # X: [B,N,C], M_tilde: [N,N]
        T0 = X
        outs = [T0 @ self.W[0]]
        if self.K >= 1:
            T1 = torch.einsum('ij,bjd->bid', M_tilde, X)
            outs.append(T1 @ self.W[1])
            Tk_1, Tk = T0, T1
            for k in range(2, self.K+1):
                Tkp1 = 2.0 * torch.einsum('ij,bjd->bid', M_tilde, Tk) - Tk_1
                outs.append(Tkp1 @ self.W[k])
                Tk_1, Tk = Tk, Tkp1
        H = sum(outs) + self.b.view(1,1,-1)
        return H

class ResChebBlock(nn.Module):
    def __init__(self, dim: int, time_dim: int, K: int = 2, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.cheb1 = ChebFilter(dim, dim, K)
        self.fc_t1 = nn.Linear(time_dim, dim)   # FiLM-lite (add only)
        self.drop = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(dim)
        self.cheb2 = ChebFilter(dim, dim, K)
        self.fc_t2 = nn.Linear(time_dim, dim)

    def forward(self, X: torch.Tensor, M_tilde: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        B = X.size(0)
        h = self.norm1(X)
        h = self.cheb1(h, M_tilde) + self.fc_t1(t_emb).view(B,1,-1)
        h = F.gelu(h); h = self.drop(h)
        X = X + h

        h = self.norm2(X)
        h = self.cheb2(h, M_tilde) + self.fc_t2(t_emb).view(B,1,-1)
        h = F.gelu(h); h = self.drop(h)
        return X + h

class GraphX0Simple(nn.Module):
    """
    Predicts x0 (not eps). Fixed graph operator is built once from adjacency.
    forward(X_k, k) -> x0_hat  with shapes [B,N,D], [B] or scalar.
    """
    def __init__(
        self,
        A: torch.Tensor,          # [N,N] adjacency (symmetric)
        in_dim: int,
        hidden: int = 64,
        depth: int = 2,
        time_dim: int = 64,
        K: int = 2,
        dropout: float = 0.0,
        operator: str = "laplacian_sym",
        gamma: float = 1.0,       # used only if operator is L + gamma I or L_sym + gamma I
    ):
        super().__init__()
        A = A.to(memory_format=torch.contiguous_format)
        if operator == "laplacian_sym":
            M = laplacian_sym(A)
        elif operator in ("laplacian_gamma_sym", "Lgamma_sym"):
            M = laplacian_sym(A) + float(gamma) * torch.eye(A.size(0), dtype=A.dtype, device=A.device)
        else:
            raise ValueError("Use 'laplacian_sym' or 'laplacian_gamma_sym' for this simple model.")
        # rescale to [-1,1]
        rho = spectral_radius(M)
        self.register_buffer("M_tilde", (2.0 * M / rho) - torch.eye(M.size(0), dtype=M.dtype, device=M.device))

        self.time_dim = time_dim
        self.t_mlp = nn.Sequential(nn.Linear(time_dim, time_dim), nn.SiLU(), nn.Linear(time_dim, time_dim))

        self.in_proj = nn.Linear(in_dim, hidden)
        self.blocks = nn.ModuleList([ResChebBlock(hidden, time_dim, K=K, dropout=dropout) for _ in range(depth)])
        self.out_ln = nn.LayerNorm(hidden)
        self.out_proj = nn.Linear(hidden, in_dim)   # predicts x0 directly

    def forward(self, X: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if X.dim() == 2: X = X.unsqueeze(0)        # [N,D] -> [1,N,D]
        B = X.size(0)

        model_dtype = next(self.parameters()).dtype
        model_device = next(self.parameters()).device
        
        X = X.to(dtype=model_dtype, device=model_device)

        # time embedding
        if not torch.is_tensor(t): 
            t = torch.tensor(t, device=model_device, dtype=model_dtype)
        else:
            t = t.to(dtype=model_dtype, device=model_device)

        t = t.to(device=X.device, dtype=X.dtype).flatten()
        t_emb = self.t_mlp(sinusoidal_timestep_embedding(t, self.time_dim))  # [B,time_dim]

        h = self.in_proj(X)
        for blk in self.blocks:
            h = blk(h, self.M_tilde, t_emb)
        h = self.out_ln(h)
        x0_hat = self.out_proj(h)
        return x0_hat.squeeze(0)
