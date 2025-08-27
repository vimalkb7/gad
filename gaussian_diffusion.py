from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Literal

import math
import torch
from torch import nn
import torch.nn.functional as F


# ===============================================================
# Graph-Aware Gaussian DDPM (discrete, graph filter form)
# - Supports time-warp schedules (uniform / cosine / poly / loglinear)
# - Exact forward 0->k marginals and per-step (k-1)->k transitions
# - Reverse sampling:
#     (A) Score-based (Anderson/Song)   : μθ = A_k^{-1}(x_k - Q_k sθ)
#     (B) DDPM posterior w/ x0-estimate : μθ = B_post_k x0_hat + C_post_k x_k,
#         with exact posterior covariance ˜Σ_post_k
# - ε-pred training (default), with consistent score sθ = B_k^{-T} ε̂
# ===============================================================


# ------------------------- Schedules -------------------------
def _build_warp(T: int,
                warp: str,
                tau_T: float,
                poly_p: float,
                log_eps: float,
                dt_uniform: float,
                device: torch.device,
                dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
      tau: [T+1]  cumulative OU-time τ_k
      dtau: [T]   Δτ_k = τ_{k+1} - τ_k
    """
    t = torch.linspace(0., 1., T + 1, device=device, dtype=dtype)

    if warp.lower() in ("uniform", "const", "constant"):
        # Backward-compat: uniform τ_k = k * dt_uniform
        tau = torch.arange(T + 1, device=device, dtype=dtype) * dt_uniform
    elif warp.lower() == "cosine":
        s = 0.5 * (1.0 - torch.cos(math.pi * t))
        tau = tau_T * s
    elif warp.lower() in ("poly", "polynomial"):
        s = t.clamp_min(0).pow(poly_p)
        tau = tau_T * s
    elif warp.lower() in ("loglinear", "log-linear", "log_lin", "log"):
        eps = torch.tensor(log_eps, device=device, dtype=dtype).clamp_min(1e-12)
        s = -torch.log1p(-(1 - eps) * t) / (-torch.log(eps))
        tau = tau_T * s
    else:
        raise ValueError(f"Unknown warp schedule: {warp}")

    dtau = tau[1:] - tau[:-1]
    # Guard perfect monotonicity numerically
    tau, _ = torch.cummax(tau, dim=0)
    dtau = torch.clamp(dtau, min=torch.finfo(dtype).eps)
    return tau, dtau


@dataclass
class GraphDDPMSchedule:
    # Core
    T: int = 1000
    c: float = 1.0
    sigma: float = 1.0
    gamma: float = 1e-3
    # Training
    train_max_t: int = 50
    use_steady_state_init: bool = True  # x_kstart ~ N(0, sigma^2 L_gamma^{-1})
    # Time warp
    warp: str = "uniform"               # 'uniform' | 'cosine' | 'poly' | 'loglinear'
    tau_T: float = 1.0                  # total OU-time when warp != 'uniform'
    poly_p: float = 1.0                 # exponent for 'poly'
    log_eps: float = 1e-6               # epsilon for 'loglinear'
    dt: float = 1e-2                    # used when warp='uniform'
    # Reverse mean type
    mean_type: Literal["score", "posterior_eps"] = "score"
    # Numerical
    jitter_factor: float = 1e-6


class GraphGaussianDDPM(nn.Module):
    def __init__(self, L: torch.Tensor, denoise_fn: nn.Module, schedule: GraphDDPMSchedule):
        """
        Args:
          L: Laplacian [N,N], symmetric PSD
          denoise_fn: ε-model  ε̂ = f(x_k, adj, k)  -> [B,N,D]
          schedule: config
        """
        super().__init__()
        assert L.dim() == 2 and L.shape[0] == L.shape[1], "L must be [N,N]"
        N = L.shape[0]
        self.N = N
        self.cfg = schedule

        # Build L_gamma (PD)
        I = torch.eye(N, dtype=L.dtype, device=L.device)
        Lg = L + schedule.gamma * I
        L_gamma = (Lg + Lg.T) * 0.5  # symmetrize
        self.register_buffer("I", I)
        self.register_buffer("L_gamma", L_gamma)
        self.register_buffer("eps_num", torch.tensor(1e-12, dtype=L.dtype, device=L.device))

        c, sigma = schedule.c, schedule.sigma

        # Time-warp
        tau, dtau = _build_warp(
            T=schedule.T,
            warp=schedule.warp,
            tau_T=schedule.tau_T,
            poly_p=schedule.poly_p,
            log_eps=schedule.log_eps,
            dt_uniform=schedule.dt,
            device=L.device,
            dtype=L.dtype,
        )
        self.register_buffer("tau", tau)      # [T+1]
        self.register_buffer("dtau", dtau)    # [T]

        # Precompute per-step exact linear OU operators
        #   A_k = exp(-c L_gamma Δτ_k)               [T,N,N]
        #   Q_k = σ^2 (I - A_k^2) L_gamma^{-1}       [T,N,N]
        #   H_k = exp(-c L_gamma τ_k)                [T+1,N,N] (cumulative)
        A_steps = []
        Q_steps = []
        H_list = [I]
        for k in range(schedule.T):
            Ak = torch.matrix_exp(-c * L_gamma * dtau[k])   # [N,N]
            A_steps.append(Ak)
            Ak2 = Ak @ Ak
            Qk = sigma**2 * torch.linalg.solve(L_gamma, (I - Ak2))
            Q_steps.append((Qk + Qk.T) * 0.5)
            H_list.append(Ak @ H_list[-1])

        A_steps = torch.stack(A_steps, dim=0)               # [T,N,N]
        Q_steps = torch.stack(Q_steps, dim=0)               # [T,N,N]
        H_pows = torch.stack(H_list, dim=0)                 # [T+1,N,N]

        self.register_buffer("A_steps", A_steps)
        self.register_buffer("Q_steps", Q_steps)
        self.register_buffer("H_pows", H_pows)

        # 0->k marginals:
        #   Σ_k = σ^2 (I - H_k^2) L_gamma^{-1}
        S_list = []
        Bk_list = []  # chol(Σ_k)
        for k in range(schedule.T + 1):
            Hk = H_pows[k]
            Sk = I - (Hk @ Hk)
            Sk = (Sk + Sk.T) * 0.5
            # clamp tiny negatives introduced by round-off
            evals = torch.linalg.eigvalsh(Sk)
            lam_min = torch.min(evals)
            if lam_min < 0:
                bump = (-lam_min + 1e-12)
                Sk = Sk + bump * self.I
            S_list.append(Sk)
            
            if k == 0:
                Bk = torch.zeros_like(L_gamma)
            else:
                Sigmak = sigma**2 * torch.linalg.solve(L_gamma, S_list[-1])
                Sigmak = (Sigmak + Sigmak.T) * 0.5
                try:
                    # Bk = torch.linalg.cholesky(Sigmak, upper=False)
                    Bk = self._safe_cholesky(Sigmak, name="Sigma_k")
                except RuntimeError:
                    eps = schedule.jitter_factor * torch.trace(Sigmak) / N
                    Bk = torch.linalg.cholesky(Sigmak + eps * I, upper=False)
            Bk_list.append(Bk)

        S_pows = torch.stack(S_list, dim=0)                 # [T+1,N,N]
        Bk_lowers = torch.stack(Bk_list, dim=0)             # [T+1,N,N]
        self.register_buffer("S_pows", S_pows)
        self.register_buffer("Bk_lowers", Bk_lowers)

        # Per-step sqrt for Q_k   (for reverse noise)
        Bdelta_list = []
        for k in range(schedule.T):
            Qk = Q_steps[k]
            try:
                # Bk = torch.linalg.cholesky(Qk, upper=False)
                Bk = self._safe_cholesky(Qk, name="Q_k")
            except RuntimeError:
                eps = schedule.jitter_factor * torch.trace(Qk) / N
                Bk = torch.linalg.cholesky(Qk + eps * I, upper=False)
            Bdelta_list.append(Bk)
        B_delta_steps = torch.stack(Bdelta_list, dim=0)     # [T,N,N]
        self.register_buffer("B_delta_steps", B_delta_steps)

        # DDPM posterior coefficients and covariance (for k>=1)
        #   B_post_k = (I - A_{k-1}^2) H_{k-1} S_k^{-1}
        #   C_post_k = A_{k-1} S_{k-1} S_k^{-1}
        #   ˜Σ_post_k = σ^2 L_gamma^{-1} (I - A_{k-1}^2) S_{k-1} S_k^{-1}
        Bpost_list, Cpost_list, Bpostchol_list = [], [], []
        for k in range(1, schedule.T + 1):
            Akm1 = A_steps[k - 1]
            Hkm1 = H_pows[k - 1]
            Skm1 = S_pows[k - 1]
            Sk = S_pows[k]

            # Solve X = Sk^{-1} V via linear solve for numerical stability
            # (a) B_post_k
            V = (I - Akm1 @ Akm1) @ Hkm1

            ridge = self.cfg.jitter_factor * torch.trace(Sk) / self.N
            Sk_stable = (Sk + Sk.transpose(0,1)) * 0.5 + ridge * self.I
            # Y = Sk^{-1} V  =>  Sk Y = V
            Y = torch.linalg.solve(Sk_stable, V)
            Bpost = Y
            # (b) C_post_k
            V2 = Akm1 @ Skm1
            Y2 = torch.linalg.solve(Sk_stable, V2)
            Cpost = Y2

            # (c) posterior covariance
            V3 = (I - Akm1 @ Akm1) @ Skm1
            Y3 = torch.linalg.solve(Sk_stable, V3)
            Sig_post = sigma**2 * torch.linalg.solve(L_gamma, Y3)
            Sig_post = (Sig_post + Sig_post.T) * 0.5
            try:
                # Btil = torch.linalg.cholesky(Sig_post, upper=False)
                Btil = self._safe_cholesky(Sig_post, name="Sigma_post")
            except RuntimeError:
                eps = schedule.jitter_factor * torch.trace(Sig_post) / N
                Btil = torch.linalg.cholesky(Sig_post + eps * I, upper=False)

            Bpost_list.append(Bpost)
            Cpost_list.append(Cpost)
            Bpostchol_list.append(Btil)

        B_post_steps = torch.stack(Bpost_list, dim=0)         # [T,N,N]   index k-1 -> coeffs for step k
        C_post_steps = torch.stack(Cpost_list, dim=0)         # [T,N,N]
        B_post_chol_steps = torch.stack(Bpostchol_list, dim=0)# [T,N,N]
        self.register_buffer("B_post_steps", B_post_steps)
        self.register_buffer("C_post_steps", C_post_steps)
        self.register_buffer("B_post_chol_steps", B_post_chol_steps)

        # Cholesky for stationary init: L_gamma = U^T U
        try:
            U = torch.linalg.cholesky(L_gamma, upper=True)
        except RuntimeError:
            eps = schedule.jitter_factor * torch.trace(L_gamma) / N
            U = torch.linalg.cholesky(L_gamma + eps * I, upper=True)
        self.register_buffer("U_chol_upper", U)

        self.denoise_fn = denoise_fn

    # ------------------------- Small helpers -------------------------
    def _mat_apply(self, M: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
        """Apply [N,N] M to batch X [B,N,D] -> [B,N,D]."""
        return torch.einsum("nm,bmd->bnd", M, X)

    def _apply_by_step(self, M_steps: torch.Tensor, k: torch.Tensor, X: torch.Tensor, step_offset: int = 0) -> torch.Tensor:
        """
        Apply per-sample matrix M_steps[j] to X[b], where j = k_b + step_offset.
        M_steps shape:
          - [T+1,N,N] for H_pows/S_pows (use j=k)
          - [T,N,N]   for A_steps/Q_steps/B_delta_steps (use j=k-1)
        """
        j = (k + step_offset).clamp_min(0)
        Mk = M_steps.index_select(0, j)  # [B,N,N]
        return torch.einsum("bnm,bmd->bnd", Mk, X)

    def _solve_by_step(self, A_steps: torch.Tensor, k: torch.Tensor, rhs: torch.Tensor, step_offset: int = 0) -> torch.Tensor:
        """Solve A_j * x = rhs per sample, j = k + step_offset."""
        Bsz, N, D = rhs.shape
        j = (k + step_offset).clamp_min(0)
        Aj = A_steps.index_select(0, j)                       # [B,N,N]
        rhs2 = rhs.permute(0, 2, 1).reshape(Bsz * D, N, 1)    # [B*D,N,1]
        Aj_batch = Aj.unsqueeze(1).expand(Bsz, D, N, N).reshape(Bsz * D, N, N)
        x2 = torch.linalg.solve(Aj_batch, rhs2)
        x = x2.reshape(Bsz, D, N).permute(0, 2, 1)
        return x

    def _eps_from_e(self, e: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        """e_k = B_k eps  =>  eps = B_k^{-1} e_k   (triangular solves)."""
        Bsz, N, D = e.shape
        Bk = self.Bk_lowers.index_select(0, k)                 # [B,N,N], lower
        e2 = e.permute(0, 2, 1).reshape(Bsz * D, N, 1)
        Bk_batch = Bk.unsqueeze(1).expand(Bsz, D, N, N).reshape(Bsz * D, N, N)
        eps2 = torch.linalg.solve_triangular(Bk_batch, e2, upper=False)
        eps = eps2.reshape(Bsz, D, N).permute(0, 2, 1)
        return eps

    def _score_from_epshat(self, eps_hat: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        """
        s_theta = B_k^{-T} * eps_hat  (note: no minus sign here)
        Using μθ = A^{-1}(x - Q sθ) makes it equivalent to +Q * (conditional score).
        """
        Bsz, N, D = eps_hat.shape
        Bk = self.Bk_lowers.index_select(0, k)                 # [B,N,N], lower
        eps2 = eps_hat.permute(0, 2, 1).reshape(Bsz * D, N, 1)
        BkT_batch = Bk.transpose(1, 2).unsqueeze(1).expand(Bsz, D, N, N).reshape(Bsz * D, N, N)
        s2 = torch.linalg.solve_triangular(BkT_batch, eps2, upper=True)
        s = s2.reshape(Bsz, D, N).permute(0, 2, 1)
        return s

    def _safe_cholesky(self, M: torch.Tensor, name: str, *, force_psd: bool = True) -> torch.Tensor:
        """
        Returns lower-triangular L such that L L^T ≈ M.
        Strategy:
        1) Try Cholesky.
        2) If it fails, symmetrize and add adaptive jitter based on λ_min(M).
        3) Retry Cholesky on M + jitter * I.
        """
        I = self.I
        N = M.shape[-1]
        # 1) quick try
        try:
            return torch.linalg.cholesky(M, upper=False)
        except RuntimeError:
            pass

        # 2) symmetrize (kills tiny asymmetry)
        Msym = (M + M.transpose(-2, -1)) * 0.5

        # eigen floor
        evals = torch.linalg.eigvalsh(Msym)
        lam_min = torch.min(evals)
        lam_max = torch.max(evals)

        # base jitter from config
        base = self.cfg.jitter_factor
        # absolute scale from trace (or max diag if trace is ~0)
        scale = torch.trace(Msym) / M.shape[-1]
        if scale.abs() < 1e-30:
            scale = torch.max(torch.diagonal(Msym))

        # if λ_min < 0, add at least (-λ_min + tiny)
        need = torch.clamp(-lam_min + 1e-12, min=0.0)
        # also add a relative bit vs spectrum
        rel = base * torch.clamp(lam_max, min=1.0)

        jitter = torch.clamp(need, min=0.0) + torch.clamp(base * scale, min=0.0) + rel

        try:
            return torch.linalg.cholesky(Msym + jitter * I, upper=False)
        except RuntimeError as e:
            # As a last resort, increase jitter x10 and try again
            jitter2 = 10.0 * jitter
            return torch.linalg.cholesky(Msym + jitter2 * I, upper=False)

    # ------------------------- Forward process -------------------------
    @torch.no_grad()
    def q_sample(self, x0: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Draw x_k ~ N(H_k x0, Σ_k) using B_k:
            x_k = H_k x0 + B_k z,  z~N(0,I)
        Returns: x_k, mean(H_k x0), e_k = x_k - mean
        """
        x0 = x0.to(dtype=self.L_gamma.dtype, device=self.L_gamma.device)
        assert x0.dim() == 3
        mean = self._apply_by_step(self.H_pows, k, x0, step_offset=0)     # H_k x0
        z = torch.randn_like(x0)
        Bk = self.Bk_lowers.index_select(0, k)
        e_k = torch.einsum("bnm,bmd->bnd", Bk, z)
        x_k = mean + e_k
        return x_k, mean, e_k

    # ------------------------- Training loss: ε-pred -------------------------
    def loss_epsilon_matching(
        self,
        eps_model: Optional[nn.Module],
        x0: torch.Tensor,
        adj: torch.Tensor,
        k: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        L_eps = E || ε - ε̂(x_k, adj, k) ||^2
        where x_k = H_k x0 + B_k ε,   ε ~ N(0,I).
        """
        Bsz = x0.shape[0]
        device = x0.device
        if k is None:
            k = torch.randint(low=1, high=self.cfg.train_max_t + 1, size=(Bsz,), device=device)

        with torch.no_grad():
            xk, _, e_k = self.q_sample(x0, k=k)
            eps = self._eps_from_e(e_k, k)                                  # whiten

        eps_model = self.denoise_fn if eps_model is None else eps_model
        eps_hat = eps_model(xk, adj, k)
        loss = F.mse_loss(eps_hat.float(), eps.float())
        return loss

    # ------------------------- Reverse mean(s) -------------------------
    @torch.no_grad()
    def p_mean(self, xk: torch.Tensor, k: torch.Tensor, eps_model: Optional[nn.Module], adj: torch.Tensor) -> torch.Tensor:
        """
        mean_type == "score":
            μθ = A_{k-1}^{-1} ( x_k - Q_{k-1} sθ ),  sθ = B_k^{-T} ε̂
        mean_type == "posterior_eps":
            x0_hat = H_k^{-1} ( x_k - B_k ε̂ )
            μθ = B_post_k x0_hat + C_post_k x_k
        """
        eps_model = self.denoise_fn if eps_model is None else eps_model
        eps_hat = eps_model(xk, adj, k)                                # [B,N,D]

        if self.cfg.mean_type == "score":
            s_theta = self._score_from_epshat(eps_hat, k)              # [B,N,D]
            rhs = xk - self._apply_by_step(self.Q_steps, k, s_theta, step_offset=-1)
            mu = self._solve_by_step(self.A_steps, k, rhs, step_offset=-1)   # A_{k-1}^{-1}(...)
            return mu

        elif self.cfg.mean_type == "posterior_eps":
            # x0_hat via ε̂
            Bk = self.Bk_lowers.index_select(0, k)                     # [B,N,N]
            Beps = torch.einsum("bnm,bmd->bnd", Bk, eps_hat)
            rhs = xk - Beps
            x0_hat = self._solve_by_step(self.H_pows, k, rhs, step_offset=0)  # H_k^{-1}(...)

            # μθ = B_post_k x0_hat + C_post_k x_k
            part1 = self._apply_by_step(self.B_post_steps, k, x0_hat, step_offset=-1)
            part2 = self._apply_by_step(self.C_post_steps, k, xk, step_offset=-1)
            mu = part1 + part2
            return mu

        else:
            raise ValueError(f"Unknown mean_type: {self.cfg.mean_type}")

    @torch.no_grad()
    def p_sample(self, xk: torch.Tensor, k: torch.Tensor, eps_model: Optional[nn.Module], adj: torch.Tensor, add_noise: bool=True) -> torch.Tensor:
        mu = self.p_mean(xk, k=k, eps_model=eps_model, adj=adj)
        if not add_noise:
            return mu

        if self.cfg.mean_type == "score":
            B = self.B_delta_steps.index_select(0, (k - 1).clamp_min(0))
        else:  # "posterior_eps"
            B = self.B_post_chol_steps.index_select(0, (k - 1).clamp_min(0))
        z = torch.randn_like(xk)
        return mu + torch.einsum("bnm,bmd->bnd", B, z)

    # ------------------------- Sampling loop -------------------------
    @torch.no_grad()
    def sample(
        self,
        B: int,
        D: int,
        eps_model: Optional[nn.Module],
        adj: torch.Tensor,
        deterministic_last: bool=True,
        start_k: Optional[int] = None,
        x_start: Optional[torch.Tensor] = None,
        capture_steps: Optional[Tuple[int, ...]] = None,
    ) -> torch.Tensor | Tuple[torch.Tensor, dict]:
        device = self.L_gamma.device
        dtype = self.L_gamma.dtype
        eps_model = self.denoise_fn if eps_model is None else eps_model

        # Determine start step
        k_start = int(self.cfg.train_max_t if start_k is None else start_k)
        assert 1 <= k_start <= self.cfg.T

        # Initialize x_{k_start}
        if x_start is not None:
            x = x_start.to(device=device, dtype=dtype)
            B_eff, _, D_eff = x.shape
        else:
            assert B is not None and D is not None, "B and D are required when x_start is None"
            B_eff, D_eff = int(B), int(D)
            if self.cfg.use_steady_state_init:
                # x ~ N(0, σ^2 L_gamma^{-1}) via chol(L_gamma)=U (upper): x = σ * U^{-1} z
                z = torch.randn(B_eff, self.N, D_eff, device=device, dtype=dtype)
                U = self.U_chol_upper
                z2 = z.permute(0, 2, 1).reshape(B_eff * D_eff, self.N, 1)
                U_batch = U.unsqueeze(0).expand(B_eff * D_eff, -1, -1)
                y2 = torch.linalg.solve_triangular(U_batch, z2, upper=True)   # U y = z
                x = self.cfg.sigma * y2.reshape(B_eff, D_eff, self.N).permute(0, 2, 1)
            else:
                x = torch.randn(B_eff, self.N, D_eff, device=device, dtype=dtype)

        # Optional trajectory capture
        snapshots = {}
        capture_set = set(capture_steps) if capture_steps is not None else None

        # Reverse loop: k_start → 1
        for step in range(k_start, 0, -1):
            k_vec = torch.full((B_eff,), step, device=device, dtype=torch.long)
            add_noise = not (deterministic_last and step == 1)
            x = self.p_sample(x, k=k_vec, eps_model=eps_model, adj=adj, add_noise=add_noise)
            if capture_set is not None and step in capture_set:
                snapshots[int(step)] = x.detach().cpu().clone()

        return (x, snapshots) if capture_set is not None else x
