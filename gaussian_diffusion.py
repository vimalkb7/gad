from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F


# ===============================================================
# Graph-Aware Gaussian DDPM (discrete) — SIMPLE (no spectral eigendecomp)
#
# SDE (OU on graphs; using sqrt(2c) * sigma convention):
#   d x_t = -c (L + gamma I) x_t dt + sqrt(2c) * sigma * dW_t
#
# Exact Δt-discretization:
#   L_gamma     = L + gamma I
#   tilde_alpha = expm(-c * L_gamma * Δt)                  (N x N)
#   alpha       = tilde_alpha @ tilde_alpha = expm(-2c * L_gamma * Δt)
#   SigmaΔ      = sigma^2 * (I - alpha) @ L_gamma^{-1}     (N x N)
#
# k-step marginals (t_k = k Δt):
#   H_k   = tilde_alpha^k = expm(-c * L_gamma * t_k)
#   Σ_k   = sigma^2 * (I - alpha^k) @ L_gamma^{-1}
#
# Reverse (DDPM-style) mean using a SCORE (built from epsilon predictor):
#   s_theta(x_k,k) = B_k^{-T} * eps_theta(x_k, adj, k)
#   mu_theta = tilde_alpha^{-1} ( x_k - SigmaΔ @ s_theta )
#   x_{k-1}  = mu_theta + BΔ z,  z~N(0,I),  BΔ BΔ^T = SigmaΔ.
#
# Training objective (Noise-matching / epsilon-prediction):
#   Write x_k = H_k x0 + e_k with e_k ~ N(0, Σ_k) and e_k = B_k eps, eps~N(0,I).
#   Predict eps_theta(x_k, adj, k) and minimize:
#     L_eps = E[ || eps - eps_theta(x_k, adj, k) ||^2 ]
#
# Implementation: No eigenbasis anywhere. We use matrix exponentials,
# linear solves, and Cholesky factorizations (for ΣΔ and Σ_k).
# ===============================================================


@dataclass
class GraphDDPMSchedule:
    T: int = 1000
    dt: float = 1e-2
    c: float = 1.0
    sigma: float = 1.0
    gamma: float = 1e-3
    use_steady_state_init: bool = True  # x_T ~ N(0, sigma^2 L_gamma^{-1}) via Cholesky


class GraphGaussianDDPM(nn.Module):
    def __init__(self, L: torch.Tensor, denoise_fn: nn.Module, schedule: GraphDDPMSchedule):
        """
        Args:
          L: Laplacian [N,N], symmetric PSD
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
        L_gamma = (Lg + Lg.T) * 0.5  # soft symmetrization
        self.register_buffer("L_gamma", L_gamma)

        # One-step matrices
        dt, c, sigma = schedule.dt, schedule.c, schedule.sigma
        tilde_alpha = torch.matrix_exp(-c * L_gamma * dt)           # [N,N]
        alpha = tilde_alpha @ tilde_alpha                           # [N,N]
        # SigmaΔ = sigma^2 * (I - alpha) @ L_gamma^{-1}
        Sigma_delta = sigma**2 * torch.linalg.solve(L_gamma, (I - alpha))  # [N,N]

        # Reverse noise factor (Cholesky of ΣΔ)
        try:
            B_delta = torch.linalg.cholesky((Sigma_delta + Sigma_delta.T) * 0.5, upper=False)
        except RuntimeError:
            eps = 1e-6 * torch.trace(Sigma_delta) / N
            B_delta = torch.linalg.cholesky((Sigma_delta + Sigma_delta.T) * 0.5 + eps * I, upper=False)

        # Cache buffers
        self.register_buffer("I", I)
        self.register_buffer("tilde_alpha", tilde_alpha)
        self.register_buffer("alpha", alpha)
        self.register_buffer("Sigma_delta", Sigma_delta)
        self.register_buffer("B_delta", B_delta)
        self.register_buffer("eps_num", torch.tensor(1e-12, dtype=L.dtype, device=L.device))

        # Precompute powers H_k = tilde_alpha^k and alpha^k
        T = schedule.T
        H_list = [I]
        A_list = [I]
        for _ in range(T):
            H_list.append(tilde_alpha @ H_list[-1])   # tilde_alpha^k
            A_list.append(alpha @ A_list[-1])         # alpha^k
        self.register_buffer("H_pows", torch.stack(H_list, dim=0))       # [T+1,N,N]
        self.register_buffer("alpha_pows", torch.stack(A_list, dim=0))   # [T+1,N,N]

        # Precompute B_k (lower Cholesky of Σ_k) for k=0..T
        # Σ_k = sigma^2 * (I - alpha^k) @ L_gamma^{-1}
        Bk_list = []
        for k in range(T + 1):
            Ak = I - self.alpha_pows[k]                         # [N,N]
            Sigma_k = sigma**2 * torch.linalg.solve(L_gamma, Ak)
            Sigma_k = (Sigma_k + Sigma_k.T) * 0.5
            if k == 0:
                Bk = torch.zeros_like(Sigma_k)                  # Σ_0 = 0
            else:
                try:
                    Bk = torch.linalg.cholesky(Sigma_k, upper=False)
                except RuntimeError:
                    eps = 1e-6 * torch.trace(Sigma_k) / N
                    Bk = torch.linalg.cholesky(Sigma_k + eps * I, upper=False)
            Bk_list.append(Bk)
        self.register_buffer("Bk_lowers", torch.stack(Bk_list, dim=0))   # [T+1,N,N]

        # For stationary init: x_T ~ N(0, sigma^2 L_gamma^{-1}) => x = sigma * U^{-1} z, U=chol(L_gamma)
        try:
            U = torch.linalg.cholesky(L_gamma, upper=True)  # L_gamma = U^T U
        except RuntimeError:
            eps = 1e-6 * torch.trace(L_gamma) / N
            U = torch.linalg.cholesky(L_gamma + eps * I, upper=True)
        self.register_buffer("U_chol_upper", U)

        self.denoise_fn = denoise_fn

    # ------------------------- Helpers -------------------------
    def _apply(self, M: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
        """Apply an [N,N] matrix M to batch X [B,N,D]: returns [B,N,D]."""
        return torch.einsum("nm,bmd->bnd", M, X)

    def _apply_k(self, M_pows: torch.Tensor, k: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
        """Apply per-sample matrix M_pows[k_b] to X[b]. M_pows: [T+1,N,N]; k: [B]."""
        Mk = M_pows[k]                      # [B,N,N]
        Mk = Mk.to(dtype=X.dtype, device=X.device)
        return torch.einsum("bnm,bmd->bnd", Mk, X)

    def _eps_from_e(self, e: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        """
        Compute whitened noise eps from e_k using B_k (lower):  e_k = B_k eps  =>  solve B_k eps = e_k
        """
        Bsz, N, D = e.shape
        Bk = self.Bk_lowers.index_select(0, k)           # [B,N,N], lower
        e2 = e.permute(0,2,1).reshape(Bsz*D, N, 1)       # [B*D, N, 1]
        Bk_batch = Bk.unsqueeze(1).expand(Bsz, D, N, N).reshape(Bsz*D, N, N)
        eps2 = torch.linalg.solve_triangular(Bk_batch, e2, upper=False)  # Bk * eps = e
        eps = eps2.reshape(Bsz, D, N).permute(0,2,1)     # [B,N,D]
        return eps

    def _score_from_epshat(self, eps_hat: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        """
        s_theta = B_k^{-T} eps_hat  i.e., solve B_k^T s = eps_hat
        """
        Bsz, N, D = eps_hat.shape
        Bk = self.Bk_lowers.index_select(0, k)              # [B,N,N] (lower)
        eps2 = eps_hat.permute(0,2,1).reshape(Bsz*D, N, 1)  # [B*D,N,1]
        BkT_batch = Bk.transpose(1,2).unsqueeze(1).expand(Bsz, D, N, N).reshape(Bsz*D, N, N)
        s2 = torch.linalg.solve_triangular(BkT_batch, eps2, upper=True)  # Bk^T s = eps_hat
        s = s2.reshape(Bsz, D, N).permute(0,2,1)              # [B,N,D]
        return s

    # ------------------------- Forward process -------------------------
    @torch.no_grad()
    def q_sample(self, x0: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Draw x_k ~ N(H_k x0, Σ_k) using B_k:
            x_k = H_k x0 + B_k z, z~N(0,I)
        Returns: x_k, mean(H_k x0), e_k = x_k - mean
        """
        x0 = x0.to(dtype=self.H_pows.dtype, device=self.H_pows.device)
        assert x0.dim() == 3
        Bsz, N, D = x0.shape
        mean = self._apply_k(self.H_pows, k, x0)        # [B,N,D]

        z = torch.randn_like(x0)                        # [B,N,D]
        Bk = self.Bk_lowers.index_select(0, k)          # [B,N,N]
        e_k = torch.einsum("bnm,bmd->bnd", Bk, z)       # [B,N,D]
        x_k = mean + e_k
        return x_k, mean, e_k

    # ------------------------- Diagnostic Functions -------------------------
    @torch.no_grad()
    def reverse_from_k(self, xk: torch.Tensor, k: int, eps_model: Optional[nn.Module],
                    adj: torch.Tensor, deterministic_last: bool=True) -> torch.Tensor:
        """
        Reverse DDPM from an intermediate time k back to 0 on a batch.
        Assumes one integer k for the whole batch.
        """
        device = xk.device
        eps_model = self.denoise_fn if eps_model is None else eps_model
        for step in reversed(range(1, k + 1)):
            kk = torch.full((xk.size(0),), step, device=device, dtype=torch.long)
            add_noise = not (deterministic_last and step == 1)
            xk = self.p_sample(xk, k=kk, eps_model=eps_model, adj=adj, add_noise=add_noise)
        return xk

    @torch.no_grad()
    def sweep_noisy_reconstruction(self, x0: torch.Tensor, adj: torch.Tensor,
                                eps_model: Optional[nn.Module], ks: list[int]) -> dict:
        """
        For each k in ks: sample x_k ~ q(x_k|x_0), then reconstruct \hat{x}_0 by reversing k→0.
        Returns dict with k, per-k MSE, and the intermediate tensors for inspection.
        """
        device = x0.device
        out = {'k': [], 'mse': [], 'x0_hat_list': [], 'xk_list': []}
        for k in ks:
            kk = torch.full((x0.size(0),), k, device=device, dtype=torch.long)
            xk, _, _ = self.q_sample(x0, kk)
            x0_hat = self.reverse_from_k(xk, k=k, eps_model=eps_model, adj=adj, deterministic_last=True)
            mse = torch.mean((x0_hat - x0) ** 2).detach()
            out['k'].append(k)
            out['mse'].append(mse)
            out['x0_hat_list'].append(x0_hat.detach())
            out['xk_list'].append(xk.detach())
        if len(out['mse']) > 0:
            out['mse'] = torch.stack(out['mse'])
        return out

    # ------------------------- Training loss: epsilon prediction -------------------------
    def loss_epsilon_matching(
        self,
        eps_model: Optional[nn.Module],
        x0: torch.Tensor,
        adj: torch.Tensor,
        k: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        L_eps = E || eps - eps_theta(x_k, adj, k) ||^2
        """
        Bsz = x0.shape[0]
        device = x0.device
        if k is None:
            k = torch.randint(low=1, high=self.cfg.T + 1, size=(Bsz,), device=device)

        with torch.no_grad():
            xk, _, e_k = self.q_sample(x0, k=k)         # [B,N,D]
            eps = self._eps_from_e(e_k, k)              # [B,N,D]

        eps_model = self.denoise_fn if eps_model is None else eps_model
        eps_hat = eps_model(xk, adj, k)                 # [B,N,D]
        loss = F.mse_loss(eps_hat, eps)
        return loss

    # ------------------------- Reverse step -------------------------
    @torch.no_grad()
    def p_mean(self, xk: torch.Tensor, k: torch.Tensor, eps_model: Optional[nn.Module], adj: torch.Tensor) -> torch.Tensor:
        """
        mu_theta = tilde_alpha^{-1} ( x_k - SigmaΔ @ s_theta ),  s_theta = B_k^{-T} eps_hat
        """
        eps_model = self.denoise_fn if eps_model is None else eps_model
        eps_hat = eps_model(xk, adj, k)                    # [B,N,D]
        s_theta = self._score_from_epshat(eps_hat, k)      # [B,N,D]
        rhs = xk - self._apply(self.Sigma_delta, s_theta)  # [B,N,D]

        # Solve tilde_alpha * mu = rhs  (batched over B*D)
        Bsz, N, D = rhs.shape
        rhs2 = rhs.permute(0,2,1).reshape(Bsz*D, N, 1)
        A = self.tilde_alpha.unsqueeze(0).expand(Bsz*D, N, N).contiguous()
        mu2 = torch.linalg.solve(A, rhs2)
        mu = mu2.reshape(Bsz, D, N).permute(0,2,1)
        return mu

    @torch.no_grad()
    def p_sample(self, xk: torch.Tensor, k: torch.Tensor, eps_model: Optional[nn.Module], adj: torch.Tensor, add_noise: bool=True) -> torch.Tensor:
        mu = self.p_mean(xk, k=k, eps_model=eps_model, adj=adj)
        if add_noise:
            z = torch.randn_like(xk)
            x_prev = mu + self._apply(self.B_delta, z)
        else:
            x_prev = mu
        return x_prev

    # ------------------------- Sampling loop -------------------------
    @torch.no_grad()
    def sample(self, B: int, D: int, eps_model: Optional[nn.Module], adj: torch.Tensor, deterministic_last: bool=True) -> torch.Tensor:
        device = self.L_gamma.device
        # Init x_T
        if self.cfg.use_steady_state_init:
            # x_T ~ N(0, sigma^2 L_gamma^{-1}) via chol(L_gamma)=U (upper): x = sigma * U^{-1} z
            z = torch.randn(B, self.N, D, device=device, dtype=self.L_gamma.dtype)
            U = self.U_chol_upper
            z2 = z.permute(0,2,1).reshape(B*D, self.N, 1)
            U_batch = U.unsqueeze(0).expand(B*D, -1, -1)
            y2 = torch.linalg.solve_triangular(U_batch, z2, upper=True)   # U y = z
            x = self.cfg.sigma * y2.reshape(B, D, self.N).permute(0,2,1)
        else:
            x = torch.randn(B, self.N, D, device=device, dtype=self.L_gamma.dtype)

        eps_model = self.denoise_fn if eps_model is None else eps_model

        # Reverse steps
        for step in reversed(range(1, self.cfg.T + 1)):
            k = torch.full((B,), step, device=device, dtype=torch.long)
            add_noise = not (deterministic_last and step == 1)
            x = self.p_sample(x, k=k, eps_model=eps_model, adj=adj, add_noise=add_noise)
        return x
