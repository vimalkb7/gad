# gaussian_diffusion_x0_pred.py
# =====================================
# Graph-aware Gaussian DDPM with x0-prediction — eigenbasis-stable version
# Implements the exact discrete-time chain induced by the warped graph heat SDE.
# Matches the math (Euler–Maruyama note → exact OU Gaussians):
#   Forward (transition):  x_k = A_k x_{k-1} + η_k,   η_k ~ N(0, Q_k)
#   Marginal:              x_k|x_0 ~ N(H_k x_0, Σ_k)
#   Posterior (plug-in):   p(x_{k-1}|x_k, \hat x_0) = N( μθ, \widetilde Σ_k )
# All matrix ops are done mode-wise in the eigen-basis of L_γ for stability.
#
# Shapes follow the original code:
#   X: [B, N, D]
#   k: [B] long, with 1..T indicating the current discrete step (k)
#
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import math
import torch
from torch import nn
import torch.nn.functional as F


# ------------------------- Schedules -------------------------
def _build_warp(
    T: int,
    warp: str,
    tau_T: float,
    poly_p: float,
    log_eps: float,
    dt_uniform: float,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
      tau:  [T+1]  cumulative OU-time τ_k (τ_0 = 0)
      dtau: [T]    Δτ_k = τ_k - τ_{k-1}, for k=1..T
    """
    t = torch.linspace(0.0, 1.0, T + 1, device=device, dtype=dtype)

    if warp.lower() in ("uniform", "const", "constant"):
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

    # Monotone & strictly positive increments
    tau, _ = torch.cummax(tau, dim=0)
    dtau = (tau[1:] - tau[:-1]).clamp_min(torch.finfo(dtype).eps)
    return tau, dtau


@dataclass
class GraphDDPMSchedule:
    # Core
    T: int = 1000
    c: float = 1.0        # OU drift scale
    sigma: float = 1.0
    gamma: float = 1e-3   # shift for L_gamma = L + gamma I
    # Training
    train_max_t: int = 50
    use_steady_state_init: bool = True  # x_kstart ~ N(0, sigma^2 L_gamma^{-1})
    # Time warp
    warp: str = "loglinear"             # 'uniform' | 'cosine' | 'poly' | 'loglinear'
    tau_T: float = 1.0                  # total OU-time when warp != 'uniform'
    poly_p: float = 1.0                 # exponent for 'poly'
    log_eps: float = 1e-6               # epsilon for 'loglinear'
    dt: float = 1e-2                    # used when warp='uniform'
    # Numerical
    jitter_factor: float = 1e-6         # used in clamp_min and inverses


class GraphGaussianDDPM_X0(nn.Module):
    """
    Graph-aware DDPM (x0-prediction), eigenbasis-stable implementation.

    Args
    ----
    L : torch.Tensor [N,N]
        Graph Laplacian, symmetric PSD.
    denoise_fn : nn.Module
        x0-predictor with interface: forward(X, t) -> x0_hat, X:[B,N,D], t:[B] or scalar.
    schedule : GraphDDPMSchedule
        Diffusion/sampling config.
    """

    def __init__(self, L: torch.Tensor, denoise_fn: nn.Module, schedule: GraphDDPMSchedule):
        super().__init__()
        assert L.dim() == 2 and L.shape[0] == L.shape[1], "L must be [N,N]"
        N = L.shape[0]
        self.N = N
        self.cfg = schedule

        # Build L_gamma (strictly PD): symmetrize L and add gamma I
        I = torch.eye(N, dtype=L.dtype, device=L.device)
        gamma = torch.as_tensor(schedule.gamma, dtype=L.dtype, device=L.device)
        Lg = (L + L.T) * 0.5 + gamma * I
        self.register_buffer("I", I)
        self.register_buffer("L_gamma", Lg)

        # Eigendecomposition once (mode-wise diagonal ops)
        lam, U = torch.linalg.eigh(Lg)  # lam: [N] > 0, U: [N,N]
        self.register_buffer("lam", lam)
        self.register_buffer("U", U)

        c, sigma = schedule.c, schedule.sigma

        # Time-warp grid
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
        self.register_buffer("tau", tau)    # [T+1], τ_k
        self.register_buffer("dtau", dtau)  # [T],   Δτ_k

        # Probabilities over k ∈ {1..train_max_t} proportional to Δτ_k
        p_k = dtau[:schedule.train_max_t]
        p_k = p_k / p_k.sum().clamp_min(self.cfg.jitter_factor)
        self.register_buffer("p_k", p_k)   # [train_max_t]

        # ----------------------------------------
        # Mode-wise scalars (exact, diagonal in eigenbasis)
        # ----------------------------------------
        #   H_k   = exp(-c * τ_k * Lγ)     -> diag(h_k,i)
        #   A_k   = exp(-c * Δτ_k * Lγ)    -> diag(a_k,i)
        #   Σ_k   = σ^2 (I - H_k^2) Lγ^-1  -> diag(v_k,i)
        #   Q_k   = σ^2 (I - A_k^2) Lγ^-1  -> diag(q_k,i)

        lam_col  = lam.view(1, N)             # [1,N]
        tau_col  = tau.view(-1, 1)            # [T+1,1]
        dtau_col = dtau.view(-1, 1)           # [T,1]

        H_modes = torch.exp(-c * tau_col * lam_col)         # [T+1, N]  (h_k,i)
        A_steps = torch.exp(-c * dtau_col * lam_col)        # [T,   N]  (a_k,i)

        inv_lam = (1.0 / lam).view(1, N)                    # [1,N]
        Sigma_modes = (sigma ** 2) * (1.0 - H_modes ** 2) * inv_lam     # [T+1,N] v_k,i
        Q_steps     = (sigma ** 2) * (1.0 - A_steps ** 2) * inv_lam     # [T,  N] q_k,i

        # Save forward quantities
        tiny_marg = max(self.cfg.jitter_factor, torch.finfo(Sigma_modes.dtype).tiny)
        self.register_buffer("H_modes", H_modes)         # h_k,i
        self.register_buffer("A_steps", A_steps)         # a_k,i
        self.register_buffer("Sigma_modes", Sigma_modes) # v_k,i
        self.register_buffer("Q_steps", Q_steps)         # q_k,i
        self.register_buffer("std_marg", torch.sqrt(Sigma_modes.clamp_min(tiny_marg)))

        # Loss weights: inverse of mean marginal variance across modes at step k
        w_k = Sigma_modes[:schedule.train_max_t].mean(dim=1).clamp_min(self.cfg.jitter_factor)  # [train_max_t]
        self.register_buffer("w_k_inv", 1.0 / w_k)  # [train_max_t]

        # ----------------------------------------
        # Exact Gaussian posterior (mode-wise)
        #   \tilde v_k^{-1} = a_k^2 / q_k + 1 / v_{k-1}
        #   μθ = B_k * \hat x0 + C_k * x_k
        # where
        #   B_k = \tilde v_k * (h_{k-1} / v_{k-1}),   C_k = \tilde v_k * (a_k / q_k)
        # ----------------------------------------
        eps = max(self.cfg.jitter_factor, torch.finfo(L.dtype).tiny)
        v_prev = self.Sigma_modes[:-1, :]           # [T,N]
        h_prev = self.H_modes[:-1, :]               # [T,N]
        a_step = self.A_steps                       # [T,N]
        q_step = self.Q_steps                       # [T,N]

        inv_v_prev = 1.0 / v_prev.clamp_min(eps)    # [T,N]
        a_over_q   = a_step / q_step.clamp_min(eps) # [T,N]
        h_over_v   = h_prev * inv_v_prev            # [T,N]

        inv_v_post = (a_step * a_over_q) + inv_v_prev     # [T,N]
        v_post     = 1.0 / inv_v_post                      # [T,N]

        B_modes = h_over_v / inv_v_post                    # [T,N]
        C_modes = a_over_q / inv_v_post                    # [T,N]
        tiny_post = max(self.cfg.jitter_factor, torch.finfo(v_post.dtype).tiny)
        std_post = torch.sqrt(v_post.clamp_min(tiny_post))

        self.register_buffer("B_modes", B_modes)
        self.register_buffer("C_modes", C_modes)
        self.register_buffer("std_post", std_post)

        self.denoise_fn = denoise_fn

    # ------------------------- Utilities: spectral transforms -------------------------
    def _to_modes(self, X: torch.Tensor) -> torch.Tensor:
        """X [B,N,D] -> Y [B,N,D], where Y = U^T X (apply per feature)."""
        return torch.einsum("nm,bmd->bnd", self.U.transpose(0, 1), X)

    def _from_modes(self, Y: torch.Tensor) -> torch.Tensor:
        """Y [B,N,D] -> X [B,N,D], where X = U Y."""
        return torch.einsum("nm,bmd->bnd", self.U, Y)
    
    def _match_model_io(self, x: torch.Tensor, t: torch.Tensor, model: nn.Module):
        """Cast (x,t) -> model's dtype/device, return (x_m, t_m) and a caster back to diffusion dtype/device."""
        mparam = next(model.parameters())
        mdev, mdtype = mparam.device, mparam.dtype
        x_m = x.to(device=mdev, dtype=mdtype)

        # keep k as same shape, but let the model decide the dtype it wants
        t_m = t.to(device=mdev, dtype=mdtype)

        # caster to bring model outputs back to diffusion dtype/device
        def to_diff_dtype(y: torch.Tensor):
            return y.to(device=self.lam.device, dtype=self.lam.dtype)

        return x_m, t_m, to_diff_dtype

    # ------------------------- Forward transition sampling (exact q(x_k|x_{k-1})) ---------
    @torch.no_grad()
    def q_step_sample(self, x_prev: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Draw x_k from the exact transition: x_k = A_k x_{k-1} + η_k, with η_k ~ N(0, Q_k).
        Uses per-mode diagonalization for stability. k in 1..T (indexing A_steps[k-1]).
        Returns: x_k, noise_sample (in node space).
        """
        device, dtype = self.lam.device, self.lam.dtype
        x_prev = x_prev.to(device=device, dtype=dtype)

        idx = (k - 1).clamp_min(0)                  # [B]
        a_k = self.A_steps.index_select(0, idx)     # [B,N]
        q_k = self.Q_steps.index_select(0, idx)     # [B,N]
        std_q = torch.sqrt(q_k.clamp_min(self.cfg.jitter_factor))  # [B,N]

        y_prev = self._to_modes(x_prev)             # [B,N,D]
        y_det  = y_prev * a_k.unsqueeze(-1)         # [B,N,D]
        z      = torch.randn_like(y_prev)
        y_eta  = std_q.unsqueeze(-1) * z            # [B,N,D]
        y_next = y_det + y_eta
        x_next = self._from_modes(y_next)
        eta    = self._from_modes(y_eta)
        return x_next, eta

    # ------------------------- Full forward chain (transition-based) ----------------------
    @torch.no_grad()
    def forward_chain(self, x0: torch.Tensor) -> torch.Tensor:
        """
        Simulate x_0 -> x_T via exact transitions q(x_k|x_{k-1}).
        Returns [B, T+1, N, D] trajectory.
        """
        device, dtype = self.lam.device, self.lam.dtype
        B, N, D = x0.shape
        traj = torch.empty(B, self.cfg.T + 1, N, D, device=device, dtype=dtype)
        traj[:, 0] = x0.to(device=device, dtype=dtype)
        x = traj[:, 0]
        for step in range(1, self.cfg.T + 1):
            k_vec = torch.full((B,), step, device=device, dtype=torch.long)
            x, _ = self.q_step_sample(x, k=k_vec)
            traj[:, step] = x
        return traj

    # ------------------------- Stationary covariance helper -------------------------------
    @torch.no_grad()
    def stationary_cov_modes(self) -> torch.Tensor:
        """Return diag(σ^2 / λ_i) as [N] (mode variances at τ=∞)."""
        return (self.cfg.sigma ** 2) * (1.0 / self.lam)

    # ------------------------- Forward marginal sampling (eq. (3)) -------------------------
    @torch.no_grad()
    def q_sample(self, x0: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Draw x_k ~ N(H_k x0, Σ_k) in eigenbasis:
            y_k = h_k .* y0 + sqrt(v_k) .* ξ,  ξ~N(0,I)
            x_k = U y_k
        Returns: x_k, mean(H_k x0), e_k = x_k - mean
        """
        device, dtype = self.lam.device, self.lam.dtype
        x0 = x0.to(dtype=dtype, device=device)

        hk   = self.H_modes.index_select(0, k)      # [B,N]
        stdk = self.std_marg.index_select(0, k)     # [B,N]

        y0     = self._to_modes(x0)                 # [B,N,D]
        y_mean = y0 * hk.unsqueeze(-1)              # [B,N,D]
        z      = torch.randn_like(y0)
        yk     = y_mean + stdk.unsqueeze(-1) * z

        x_mean = self._from_modes(y_mean)
        xk     = self._from_modes(yk)
        e_k    = xk - x_mean
        return xk, x_mean, e_k

    # ------------------------- Training loss: x0-pred (MSE) -------------------------
    # def loss_x0_matching(
    #     self,
    #     x0_model: Optional[nn.Module],
    #     x0: torch.Tensor,
    #     k: Optional[torch.Tensor] = None,
    # ) -> torch.Tensor:
    #     """
    #     L = E_k || x0 - x0_hat(x_k, k) ||^2,  where x_k is sampled exactly from the forward marginal.
    #     """
    #     Bsz = x0.shape[0]
    #     device = self.lam.device
    #     if k is None:
    #         # Sample steps uniformly from {1,...,train_max_t}
    #         k = torch.randint(low=1, high=self.cfg.train_max_t + 1, size=(Bsz,), device=device)

    #     with torch.no_grad():
    #         xk, _, _ = self.q_sample(x0, k=k)

    #     x0_model = self.denoise_fn if x0_model is None else x0_model
    #     x_in, t_in, back = self._match_model_io(xk, k, x0_model)
    #     x0_hat = back(x0_model(x_in, t_in))
    #     return F.mse_loss(x0_hat, x0.to(dtype=self.lam.dtype, device=self.lam.device))


    def loss_x0_matching(
        self,
        x0_model: Optional[nn.Module],
        x0: torch.Tensor,
        k: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        L = E_k [ w_k * || x0 - x0_hat(x_k, k) ||^2 ],
        with k sampled ∝ Δτ_k (self.p_k) and w_k = 1/mean(Σ_k) to balance noise levels.
        """
        Bsz = x0.shape[0]
        device = self.lam.device

        # sample k ∈ {1..train_max_t} with p(k) ∝ Δτ_k
        if k is None:
            # torch.multinomial expects probs sum to 1
            idx = torch.multinomial(self.p_k, num_samples=Bsz, replacement=True)  # 0..train_max_t-1
            k = idx + 1  # 1..train_max_t
            k = k.to(device=device, dtype=torch.long)

        with torch.no_grad():
            xk, _, _ = self.q_sample(x0, k=k)

        x0_model = self.denoise_fn if x0_model is None else x0_model
        x_in, t_in, back = self._match_model_io(xk, k, x0_model)
        x0_hat = back(x0_model(x_in, t_in))

        # per-step weight (broadcast to [B,1,1])
        w = self.w_k_inv.index_select(0, k - 1).view(-1, 1, 1)
        return ((x0_hat - x0) ** 2 * w).mean()


    # ------------------------- Optional: plug-in score at step k -------------------------
    @torch.no_grad()
    def score_from_x0hat(self, xk: torch.Tensor, k: torch.Tensor, x0_model: Optional[nn.Module] = None) -> torch.Tensor:
        """
        \hat ∇_x log p(x_k) = - Σ_k^{-1} (x_k - H_k \hat x0)
        """
        x0_model = self.denoise_fn if x0_model is None else x0_model
        x_in, t_in, back = self._match_model_io(xk, k, x0_model)
        x0_hat = back(x0_model(x_in, t_in))

        yk   = self._to_modes(xk)
        y0h  = self._to_modes(x0_hat)
        hk   = self.H_modes.index_select(0, k)             # [B,N]
        vik  = self.Sigma_modes.index_select(0, k)         # [B,N]
        invv = (1.0 / vik.clamp_min(self.cfg.jitter_factor)).unsqueeze(-1)  # [B,N,1]

        resid_modes = yk - y0h * hk.unsqueeze(-1)          # [B,N,D]
        score_modes = - invv * resid_modes                 # [B,N,D]
        return self._from_modes(score_modes)

    # ------------------------- Reverse mean (DDPM with \hat x0) -----------------
    @torch.no_grad()
    def p_mean(self, xk: torch.Tensor, k: torch.Tensor, x0_model: Optional[nn.Module]) -> torch.Tensor:
        """
        μθ = B_k * \hat x0 + C_k * x_k, built per-mode and rotated back.
        Uses arrays at index (k-1) for steps k=1..T.
        """
        x0_model = self.denoise_fn if x0_model is None else x0_model
        x_in, t_in, back = self._match_model_io(xk, k, x0_model)
        x0_hat = back(x0_model(x_in, t_in))

        yk  = self._to_modes(xk)           # [B,N,D]
        y0h = self._to_modes(x0_hat)       # [B,N,D]

        idx = (k - 1).clamp_min(0)         # steps 1..T -> indices 0..T-1
        Bm  = self.B_modes.index_select(0, idx)  # [B,N]
        Cm  = self.C_modes.index_select(0, idx)  # [B,N]

        y_mean = y0h * Bm.unsqueeze(-1) + yk * Cm.unsqueeze(-1)
        return self._from_modes(y_mean)

    # ------------------------- One reverse step with exact posterior variance ------------
    @torch.no_grad()
    def p_sample(self, xk: torch.Tensor, k: torch.Tensor, x0_model: Optional[nn.Module], add_noise: bool = True) -> torch.Tensor:
        """
        Draw x_{k-1} ~ N( μθ(x_k,k), \widetilde Σ_k ) in eigenbasis:
          μθ modes:  y_mean = B_k .* y0_hat + C_k .* y_k
          var modes: v_post_k  (std_post_k = sqrt(v_post_k))
        """
        x0_model = self.denoise_fn if x0_model is None else x0_model
        x_in, t_in, back = self._match_model_io(xk, k, x0_model)
        x0_hat = back(x0_model(x_in, t_in))

        yk  = self._to_modes(xk)
        y0h = self._to_modes(x0_hat)

        idx = (k - 1).clamp_min(0)
        Bm  = self.B_modes.index_select(0, idx)     # [B,N]
        Cm  = self.C_modes.index_select(0, idx)     # [B,N]
        st  = self.std_post.index_select(0, idx)    # [B,N]

        y_mean = y0h * Bm.unsqueeze(-1) + yk * Cm.unsqueeze(-1)
        if not add_noise:
            y_prev = y_mean
        else:
            z = torch.randn_like(yk)
            y_prev = y_mean + st.unsqueeze(-1) * z

        return self._from_modes(y_prev)

    # ------------------------- Sampling loop ---------------------------------------------
    @torch.no_grad()
    def sample(
        self,
        B: int,
        D: int,
        x0_model: Optional[nn.Module],
        deterministic_last: bool = True,
        start_k: Optional[int] = None,
        x_start: Optional[torch.Tensor] = None,
        capture_steps: Optional[Tuple[int, ...]] = None,
    ) -> torch.Tensor | Tuple[torch.Tensor, dict]:
        """
        Reverse chain using the exact posterior (variance and mean), with x0 plug-in.
        Starts from either provided x_start (at step start_k) or stationary prior.
        """
        device, dtype = self.lam.device, self.lam.dtype
        x0_model = self.denoise_fn if x0_model is None else x0_model

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
                # Stationary: y ~ N(0, diag(σ^2/λ)), x = U y
                std_inf = torch.sqrt(((self.cfg.sigma ** 2) * (1.0 / self.lam)).clamp_min(self.cfg.jitter_factor))  # [N]
                z = torch.randn(B_eff, self.N, D_eff, device=device, dtype=dtype)
                y = z * std_inf.view(1, -1, 1)
                x = self._from_modes(y)
            else:
                x = torch.randn(B_eff, self.N, D_eff, device=device, dtype=dtype)

        # Optional trajectory capture
        snapshots = {}
        capture_set = set(capture_steps) if capture_steps is not None else None

        # Reverse loop: k_start → 1
        for step in range(k_start, 0, -1):
            k_vec = torch.full((B_eff,), step, device=device, dtype=torch.long)
            add_noise = not (deterministic_last and step == 1)
            x = self.p_sample(x, k=k_vec, x0_model=x0_model, add_noise=add_noise)
            if capture_set is not None and step in capture_set:
                snapshots[int(step)] = x.detach().cpu().clone()

        return (x, snapshots) if capture_set is not None else x
