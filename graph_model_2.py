from typing import Optional, Tuple, List
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ================================================================
# Utilities: embeddings, operators, spectra, positional encodings
# ================================================================

def sinusoidal_timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Standard sinusoidal embedding. Accepts scalar, [B], or [N] time.
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


def laplacian_unnormalized(A: torch.Tensor) -> torch.Tensor:
    """L = D - A."""
    D = torch.diag(A.sum(-1))
    return D - A


def laplacian_symmetric(A: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    """L_sym = I - D^{-1/2} A D^{-1/2}."""
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


@torch.no_grad()
def spectral_radius_power_iter(M: torch.Tensor, iters: int = 8, eps: float = 1e-6) -> float:
    """
    Quick spectral radius / norm estimate (power iteration). Works for symmetric M.
    """
    v = torch.randn(M.size(0), device=M.device, dtype=M.dtype)
    for _ in range(iters):
        v = M @ v
        v = v / (v.norm() + eps)
    rho = float((v @ (M @ v)).item())
    return max(rho, eps)


def normalize_by_spectral_radius(M: torch.Tensor, min_div: float = 1.0) -> torch.Tensor:
    """
    Scale M by its spectral radius (≥ min_div) to keep multiplications stable.
    """
    rho = spectral_radius_power_iter(M)
    rho = max(rho, min_div)
    return M / rho


def spectral_posenc(A: torch.Tensor, k: int = 4) -> torch.Tensor:
    """
    Small-graph positional encodings: take the first k non-trivial eigenvectors of L_sym.
    """
    Ls = laplacian_symmetric(A)
    evals, evecs = torch.linalg.eigh(Ls)  # ascending
    # skip the constant eigenvector (index 0)
    if evecs.size(1) <= 1:
        return torch.zeros(A.size(0), 0, device=A.device, dtype=A.dtype)
    take = min(k, evecs.size(1) - 1)
    return evecs[:, 1:1 + take]  # [N, take]


# ===================================
# Layers: Gated FiLM, Cheb filter, MHSA
# ===================================

class GatedFiLM(nn.Module):
    """
    Gated FiLM: y = σ(g(cond)) * x + b(cond).
    """
    def __init__(self, cond_dim: int, hidden_dim: int):
        super().__init__()
        self.to_gb = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
        )
        self.sig = nn.Sigmoid()

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        if cond.dim() == 1:
            cond = cond.unsqueeze(0)
        if x.dim() == 3 and cond.size(0) == 1:
            cond = cond.expand(x.size(0), -1)
        gb = self.to_gb(cond)
        g, b = gb.chunk(2, dim=-1)
        g = self.sig(g)
        if x.dim() == 3:
            g = g.unsqueeze(1)
            b = b.unsqueeze(1)
        else:
            g = g.unsqueeze(0)
            b = b.unsqueeze(0)
        return g * x + b


class ChebTimeFilterConv(nn.Module):
    """
    Time-conditioned Chebyshev polynomial graph filter.

    y = sum_{k=0..K} α_k(t) [ T_k(M̃) X W_k ],
    where M̃ is M rescaled to [-1,1] via spectral radius estimate.

    Shapes:
      X: [N,Cin] or [B,N,Cin]
      M: [N,N]   (symmetric)
      t_emb: [B,Td] or [1,Td]
      returns: [N,Cout] or [B,N,Cout]
    """
    def __init__(self, in_dim: int, out_dim: int, time_dim: int, K: int = 3, bias: bool = True):
        super().__init__()
        self.K = int(K)
        self.W = nn.ParameterList([nn.Parameter(torch.empty(in_dim, out_dim)) for _ in range(self.K + 1)])
        for w in self.W:
            nn.init.kaiming_uniform_(w, a=math.sqrt(5))
        self.bias = nn.Parameter(torch.zeros(out_dim)) if bias else None
        self.coeff_mlp = nn.Sequential(
            nn.Linear(time_dim, (self.K + 1) * 2),
            nn.SiLU(),
            nn.Linear((self.K + 1) * 2, (self.K + 1)),
        )

    def _rescale(self, M: torch.Tensor) -> torch.Tensor:
        return (2.0 * M / spectral_radius_power_iter(M)) - torch.eye(M.size(0), device=M.device, dtype=M.dtype)

    def _cheb_basis_apply(self, X: torch.Tensor, M_tilde: torch.Tensor) -> List[torch.Tensor]:
        # Returns list [T0X, T1X, ..., TKX] matching X batch shape
        if X.dim() == 2:
            T0 = X
            out = [T0]
            if self.K >= 1:
                T1 = M_tilde @ X
                out.append(T1)
                Tkm1, Tk = T0, T1
                for _ in range(2, self.K + 1):
                    Tkp1 = 2.0 * (M_tilde @ Tk) - Tkm1
                    out.append(Tkp1)
                    Tkm1, Tk = Tk, Tkp1
            return out
        else:
            # [B,N,C]
            B = X.size(0)
            # broadcasting: (N,N) @ (B,N,C) via einsum
            T0 = X
            out = [T0]
            if self.K >= 1:
                T1 = torch.einsum("ij,bjd->bid", M_tilde, X)
                out.append(T1)
                Tkm1, Tk = T0, T1
                for _ in range(2, self.K + 1):
                    Tkp1 = 2.0 * torch.einsum("ij,bjd->bid", M_tilde, Tk) - Tkm1
                    out.append(Tkp1)
                    Tkm1, Tk = Tk, Tkp1
            return out

    def forward(self, X: torch.Tensor, M: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        M_tilde = self._rescale(M)
        basis = self._cheb_basis_apply(X, M_tilde)  # list length K+1

        # Project each basis term with its own W_k, then combine via α_k(t)
        if X.dim() == 2:
            Hk = [Bk @ Wk for Bk, Wk in zip(basis, self.W)]  # [N,Cout] per k
            alpha = self.coeff_mlp(t_emb)  # [1,K+1] or [B,K+1] but here B=1
            alpha = alpha.squeeze(0)  # [K+1]
            H = sum(a * h for a, h in zip(alpha, Hk))
        else:
            B = X.size(0)
            Hk = [Bk @ Wk for Bk, Wk in zip(basis, self.W)]  # each [B,N,Cout]
            alpha = self.coeff_mlp(t_emb)  # [B,K+1]
            # stack along "k" then weight
            Hstack = torch.stack(Hk, dim=-1)  # [B,N,Cout,K+1]
            alpha = alpha.view(B, 1, 1, self.K + 1)  # broadcast
            H = (Hstack * alpha).sum(dim=-1)  # [B,N,Cout]

        if self.bias is not None:
            H = H + (self.bias if X.dim() == 2 else self.bias.view(1, 1, -1))
        return H


class EdgeMaskedMHSA(nn.Module):
    """
    Lightweight adjacency-masked multi-head self-attention.
    """
    def __init__(self, dim: int, heads: int = 4, attn_drop: float = 0.0, proj_drop: float = 0.0):
        super().__init__()
        self.heads = heads
        self.scale = (dim // heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim, bias=True)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, X: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        squeeze = False
        if X.dim() == 2:
            X = X.unsqueeze(0)
            squeeze = True
        B, N, D = X.shape
        H = self.heads
        d = D // H

        qkv = self.qkv(X).reshape(B, N, 3, H, d).permute(2, 0, 3, 1, 4)
        Q, K, V = qkv[0], qkv[1], qkv[2]  # [B,H,N,d]

        attn_logits = (Q @ K.transpose(-2, -1)) * self.scale  # [B,H,N,N]

        # edge mask (+ self)
        mask = (A > 0).to(dtype=X.dtype, device=X.device)
        mask = mask + torch.eye(N, device=X.device, dtype=X.dtype)
        attn_logits = attn_logits + (mask.unsqueeze(0).unsqueeze(0) - 1.0) * 1e9

        attn = attn_logits.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ V).transpose(2, 3).reshape(B, N, D)
        out = self.proj_drop(self.proj(out))
        return out.squeeze(0) if squeeze else out


# ============================
# Residual Graph Block (Cheb+FiLM+Attn hook)
# ============================

class ResidualGCNBlock(nn.Module):
    def __init__(self, hidden_dim: int, time_dim: int, dropout: float = 0.0, K: int = 3):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.conv1 = ChebTimeFilterConv(hidden_dim, hidden_dim, time_dim, K=K)
        self.film1 = GatedFiLM(time_dim, hidden_dim)
        self.drop = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(hidden_dim)
        self.conv2 = ChebTimeFilterConv(hidden_dim, hidden_dim, time_dim, K=K)
        self.film2 = GatedFiLM(time_dim, hidden_dim)

    def forward(self, X: torch.Tensor, M: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.norm1(X)
        h = self.conv1(h, M, t_emb)
        h = self.film1(h, t_emb)
        h = F.gelu(h)
        h = self.drop(h)
        X = X + h

        h = self.norm2(X)
        h = self.conv2(h, M, t_emb)
        h = self.film2(h, t_emb)
        h = F.gelu(h)
        h = self.drop(h)
        X = X + h
        return X


# ============================
# Model: GraphEpsDenoiser (V2)
# ============================

class GraphEpsDenoiser(nn.Module):
    """
    Powerful ε-predictor with user-selectable operator and time-conditioned Chebyshev filtering.

    Key options:
      - operator_kind: 'adjacency' | 'normalized_adjacency' | 'laplacian' | 'laplacian_sym'
                       | 'laplacian_gamma' | 'laplacian_gamma_sym'  (aliases: adj, norm_adj, Lgamma, Lgamma_sym)
      - gamma:         used for L_gamma*
      - cheb_K:        Chebyshev order (default 3)
      - att_every:     insert edge-masked MHSA every k blocks (0 disables attention)
      - att_heads:     attention heads
      - use_spectral_pe: append top-k(L_sym) eigenvectors to node features
      - k_pe:          number of positional eigenvectors
      - spectral_norm_operator: normalize M by spectral radius each forward
      - static_graph:  cache M and PE for speed when graph stays fixed
      - mix_two_operators: if True, blends operator_kind with operator_kind2 via α(t)
    """
    def __init__(
        self,
        in_dim: int,
        hidden: int = 128,
        depth: int = 6,
        time_dim: int = 128,
        dropout: float = 0.0,
        add_self_loops: bool = True,           # impacts normalized_adjacency
        *,
        operator_kind: str = "adjacency",
        gamma: float = 0.0,
        eps: float = 1e-9,
        cheb_K: int = 3,
        att_every: int = 2,
        att_heads: int = 4,
        use_spectral_pe: bool = True,
        k_pe: int = 4,
        spectral_norm_operator: bool = True,
        static_graph: bool = False,
        mix_two_operators: bool = False,
        operator_kind2: str = "laplacian_gamma_sym",
    ):
        super().__init__()
        self.add_self_loops = add_self_loops
        self.operator_kind = operator_kind
        self.gamma = float(gamma)
        self.eps = float(eps)
        self.cheb_K = int(cheb_K)
        self.att_every = int(att_every)
        self.att_heads = int(att_heads)
        self.use_spectral_pe = bool(use_spectral_pe)
        self.k_pe = int(k_pe)
        self.spectral_norm_operator = bool(spectral_norm_operator)
        self.static_graph = bool(static_graph)
        self.mix_two_operators = bool(mix_two_operators)
        self.operator_kind2 = operator_kind2

        # t-embedding head
        self.time_dim = int(time_dim)
        self.t_mlp = nn.Sequential(
            nn.Linear(self.time_dim, self.time_dim),
            nn.SiLU(),
            nn.Linear(self.time_dim, self.time_dim),
        )

        # Optional mixer between two operators (produces α∈[0,1])
        if self.mix_two_operators:
            self.mix_mlp = nn.Sequential(
                nn.Linear(self.time_dim, self.time_dim),
                nn.SiLU(),
                nn.Linear(self.time_dim, 1),
                nn.Sigmoid(),
            )

        # Input projection (account for positional encodings)
        in_dim_eff = in_dim + (self.k_pe if self.use_spectral_pe and self.k_pe > 0 else 0)
        self.in_proj = nn.Linear(in_dim_eff, hidden)

        # Residual blocks
        self.blocks = nn.ModuleList([
            ResidualGCNBlock(hidden_dim=hidden, time_dim=self.time_dim, dropout=dropout, K=self.cheb_K)
            for _ in range(depth)
        ])

        # Attention blocks every att_every
        if self.att_every and self.att_every > 0:
            self.attn_blocks = nn.ModuleList([
                EdgeMaskedMHSA(hidden, heads=self.att_heads) for _ in range(depth // self.att_every)
            ])
        else:
            self.attn_blocks = nn.ModuleList([])

        self.out_norm = nn.LayerNorm(hidden)
        self.out_proj = nn.Linear(hidden, in_dim)

        # Lazy caches for static graphs
        self._cache = dict(M=None, PE=None, A_sig=None)

    # --------------- helpers ---------------

    def _build_operator(self, A: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        """
        Build the operator M depending on configuration (and optionally mix two).
        """
        if self.static_graph and self._cache["M"] is not None and self._cache["A_sig"] is A.data_ptr():
            M = self._cache["M"]
        else:
            M1 = build_graph_operator(
                A, operator_kind=self.operator_kind, gamma=self.gamma,
                add_self_loops=self.add_self_loops, eps=self.eps
            )
            if self.mix_two_operators:
                M2 = build_graph_operator(
                    A, operator_kind=self.operator_kind2, gamma=self.gamma,
                    add_self_loops=self.add_self_loops, eps=self.eps
                )
                alpha = self.mix_mlp(t_emb)  # [B,1] or [1,1]
                if t_emb.size(0) == 1:
                    alpha = alpha.item()
                    M = alpha * M1 + (1.0 - alpha) * M2
                else:
                    # If batch > 1, we take the mean α to keep a single M per forward
                    alpha = alpha.mean().item()
                    M = alpha * M1 + (1.0 - alpha) * M2
            else:
                M = M1

            if self.spectral_norm_operator:
                M = normalize_by_spectral_radius(M)

            if self.static_graph:
                self._cache.update(M=M, A_sig=A.data_ptr())

        return M

    def _build_posenc(self, A: torch.Tensor) -> torch.Tensor:
        if not (self.use_spectral_pe and self.k_pe > 0):
            return None
        if self.static_graph and self._cache["PE"] is not None and self._cache["A_sig"] is A.data_ptr():
            return self._cache["PE"]
        PE = spectral_posenc(A, k=self.k_pe)
        if self.static_graph:
            self._cache["PE"] = PE
        return PE

    # --------------- forward ---------------

    def forward(self, X: torch.Tensor, A: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Inputs:
          X: [N, d_in] or [B, N, d_in]
          A: [N, N] adjacency (symmetric, dense)
          t: scalar int/float or [B]
        Returns:
          ε̂: [N, d_in] or [B, N, d_in]
        """
        model_dtype = next(self.parameters()).dtype
        device = next(self.parameters()).device

        X = X.to(dtype=model_dtype, device=device)
        A = A.to(dtype=model_dtype, device=device)

        # Time embedding
        if not torch.is_tensor(t):
            t = torch.tensor(t, device=device)
        else:
            t = t.to(device=device)
        t = t.to(dtype=model_dtype, device=device)
        t_emb = sinusoidal_timestep_embedding(t.flatten(), self.time_dim)
        t_emb = self.t_mlp(t_emb)  # [B,Td] or [1,Td]

        # Positional encodings
        PE = self._build_posenc(A)
        if PE is not None:
            if X.dim() == 3:
                X = torch.cat([X, PE.unsqueeze(0).expand(X.size(0), -1, -1)], dim=-1)
            else:
                X = torch.cat([X, PE], dim=-1)

        # Operator (single M per forward)
        M = self._build_operator(A, t_emb)

        # Encode
        h = self.in_proj(X)

        # Residual blocks + optional attention
        att_idx = 0
        for i, blk in enumerate(self.blocks):
            h = blk(h, M, t_emb)
            if self.att_every and (i + 1) % self.att_every == 0 and att_idx < len(self.attn_blocks):
                # Build a boolean adjacency for masking in attention
                A_mask = (A > 0).to(h.dtype)
                h = h + self.attn_blocks[att_idx](h, A_mask)
                att_idx += 1

        # Project to ε
        h = self.out_norm(h)
        eps_hat = self.out_proj(h)
        return eps_hat

    # --------------- utils ---------------

    def reset_static_cache(self):
        self._cache = dict(M=None, PE=None, A_sig=None)
