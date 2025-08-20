# forward_pass.py  — SBM-aligned version (fixed topology, per-community Gaussian signals)
import os
import math
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional

import torch
import numpy as np
import matplotlib.pyplot as plt

# ====== NEW: import your SBM dataset API ======
from graph_dataset import build_sbm_dataset, CommunitySpec

torch.set_printoptions(threshold=float('inf'))
torch.set_default_dtype(torch.float64)

# ------------------------------------------------------------
# Exact Graph-OU forward (Euler–Maruyama evolution, exact-one-step operators precomputed)
# ------------------------------------------------------------

@torch.no_grad()
def forward_diffusion_exact(
    X_sub: torch.Tensor,     # [N, D]
    A_sub: torch.Tensor,     # [N, N]
    T: int,
    c: float,
    gamma: float,
    sigma: float,
    return_trajectory: bool = False,
):
    """
    Evolves: x_{k+1} = x_k + dt*(-c L_gamma x_k) + sqrt(2c)*sigma*sqrt(dt)*z_k
    Also returns exact one-step matrices (for theory/diagnostics).
    """
    assert X_sub.dim() == 2, "X_sub must be [N, D]"
    N, D = X_sub.shape
    device, dtype = X_sub.device, X_sub.dtype

    A_sub = A_sub.to(device=device, dtype=dtype)
    I = torch.eye(N, device=device, dtype=dtype)

    # Laplacian and L_gamma
    Dg = torch.diag(A_sub.sum(dim=1))
    L  = Dg - A_sub
    L_gamma = L + gamma * I

    dt = 1.0 / T
    G = sigma * math.sqrt(2.0 * c)

    # Precompute exact one-step matrices for theory
    tilde_alpha = torch.matrix_exp(-c * L_gamma * dt)        # [N,N]
    alpha       = tilde_alpha @ tilde_alpha                  # [N,N]
    Sigma_delta = sigma**2 * torch.linalg.solve(L_gamma, (I - alpha))  # [N,N]

    # Simulate trajectory with EM
    x = X_sub.clone()
    if return_trajectory:
        traj = torch.empty(T + 1, N, D, dtype=dtype, device=device)
        traj[0] = x

    for k in range(T):
        z = torch.randn(N, D, device=device, dtype=dtype)
        x = x + (-c * (L_gamma @ x)) * dt + (G * math.sqrt(dt) * z)
        if return_trajectory:
            traj[k + 1] = x

    if return_trajectory:
        return traj, (L, L_gamma, tilde_alpha, alpha, Sigma_delta)
    else:
        return x, (L, L_gamma, tilde_alpha, alpha, Sigma_delta)

# ------------------------------------------------------------
# Empirical mean & covariance over samples
# ------------------------------------------------------------

def empirical_mean_and_cov_over_samples(X_batch: torch.Tensor):
    """
    X_batch: [R, N, D] for a *fixed timestep k* (R trajectories)
    Returns:
      mean_k: [N, D]
      cov_k : [N, N]  (pooling features as i.i.d. replicates)
    """
    R, N, D = X_batch.shape
    mean_k = X_batch.mean(dim=0)  # [N, D]
    X_flat = X_batch.permute(0, 2, 1).reshape(R * D, N)      # [R*D, N]
    X_centered = X_flat - X_flat.mean(dim=0, keepdim=True)
    cov_k = (X_centered.T @ X_centered) / (R * D - 1)        # [N, N]
    cov_k = 0.5 * (cov_k + cov_k.T)
    return mean_k, cov_k

# ------------------------------------------------------------
# Time-series errors (vs final theoretical stats) — accepts c,gamma for labels
# ------------------------------------------------------------

def compute_time_series_errors_against_final(
    trajectories: torch.Tensor,           # [R, T+1, N, D]
    L_gamma: torch.Tensor,
    alpha: torch.Tensor,
    sigma: float,
    T: int,
    c: float,
    gamma: float,
    out_dir: str = "."
) -> dict:
    """
    For each timestep k=0..T, compute empirical mean & covariance across R samples,
    then compare to the *theoretical target at final step T only*:
        μ_T = 0
        Σ_T = sigma^2 * (I - alpha^T) @ L_gamma^{-1}
    Saves plots:
      - mean_error_vs_time_*.png
      - cov_fro_error_vs_time_*.png
    """
    device, dtype = trajectories.device, trajectories.dtype
    R, TP1, N, D = trajectories.shape
    assert TP1 == T + 1

    I = torch.eye(N, device=device, dtype=dtype)
    alpha_T = torch.linalg.matrix_power(alpha, T)
    Sigma_T = sigma**2 * torch.linalg.solve(L_gamma, (I - alpha_T))
    Sigma_T = 0.5 * (Sigma_T + Sigma_T.T)

    mu_T = torch.zeros(N * D, device=device, dtype=dtype)

    mean_err_l2 = np.zeros(TP1)
    cov_err_fro  = np.zeros(TP1)
    cov_err_rel  = np.zeros(TP1)

    Sigma_T_fro = torch.norm(Sigma_T, p='fro') + 1e-12

    for k in range(TP1):
        Xk = trajectories[:, k]                     # [R, N, D]
        mean_k, cov_k = empirical_mean_and_cov_over_samples(Xk)

        mean_diff = mean_k.reshape(-1) - mu_T
        mean_err_l2[k] = float(torch.norm(mean_diff, p=2).item())

        cov_diff = (cov_k - Sigma_T)
        fro = torch.norm(cov_diff, p='fro')
        cov_err_fro[k] = float(fro.item())
        cov_err_rel[k] = float((fro / Sigma_T_fro).item())

    os.makedirs(out_dir, exist_ok=True)

    param_str = f"T={T}, c={c:.3g}, gamma={gamma:.3g}, sigma={sigma:.3g}"

    # 1) Mean L2 error vs time
    plt.figure()
    plt.plot(np.arange(TP1), mean_err_l2, label=r"$\|\mathbb{E}[\mathbf{x}_k]-\mu_T\|_2$")
    plt.xlabel("timestep k")
    plt.ylabel(r"$\|\mathbb{E}[\mathbf{x}_k]-\mu_T\|_2$")
    plt.title("Mean error vs final theoretical mean (at step T)")
    plt.legend(title=param_str, loc="best", frameon=True)
    mean_plot_path = os.path.join(
        out_dir, f"mean_error_vs_time_T{T}_c{c:.3g}_g{gamma:.3g}_s{sigma:.3g}.png"
    )
    plt.tight_layout()
    plt.savefig(mean_plot_path, dpi=150)
    plt.close()

    # 2) Covariance Fro and relative Fro vs time
    plt.figure()
    plt.plot(np.arange(TP1), cov_err_fro,
             label=r"$\|\Sigma_k - \Sigma_T\|_F$")
    plt.plot(np.arange(TP1), cov_err_rel,
             label=r"$\|\Sigma_k - \Sigma_T\|_F / \|\Sigma_T\|_F$")
    plt.xlabel("timestep k")
    plt.ylabel("error")
    plt.title("Covariance error vs final theoretical covariance (at step T)")
    plt.legend(title=param_str, loc="best", frameon=True)
    cov_plot_path = os.path.join(
        out_dir, f"cov_fro_error_vs_time_T{T}_c{c:.3g}_g{gamma:.3g}_s{sigma:.3g}.png"
    )
    plt.tight_layout()
    plt.savefig(cov_plot_path, dpi=150)
    plt.close()

    return {
        "mean_err_l2": mean_err_l2,
        "cov_err_fro": cov_err_fro,
        "cov_err_rel": cov_err_rel,
        "mean_plot_path": mean_plot_path,
        "cov_plot_path": cov_plot_path,
        "Sigma_T": Sigma_T,
    }

# ------------------------------------------------------------
# Theoretical finite-k covariance Σ_k (reference)
# ------------------------------------------------------------

def theoretical_covariance_exact(L_gamma: torch.Tensor, alpha: torch.Tensor, sigma: float, k: int):
    """
    Σ_k = sigma^2 * (I - alpha^k) @ L_gamma^{-1}
    """
    N = L_gamma.size(0)
    device, dtype = L_gamma.device, L_gamma.dtype
    I = torch.eye(N, device=device, dtype=dtype)
    alpha_k = torch.linalg.matrix_power(alpha, k)            # [N,N]
    Sigma_k = sigma**2 * torch.linalg.solve(L_gamma, (I - alpha_k))
    Sigma_k = 0.5 * (Sigma_k + Sigma_k.T)                    # symmetrize
    return Sigma_k

# ============================================================
# SBM-consistent X0 samplers
# ============================================================

def _make_X0_batch_sbm(
    R: int,
    sizes: Tuple[int, int],
    d_feat: int,
    comm0: CommunitySpec,
    comm1: CommunitySpec,
    seed: int = 42
) -> torch.Tensor:
    """
    Vectorized sampler for initial X0 from the dataset's per-community Gaussians.
    Returns: [R, N, D] with N=sum(sizes), D=d_feat, dtype=float64
    """
    n1, n2 = sizes
    rng = np.random.default_rng(seed)
    X0_c0 = rng.normal(loc=comm0.mean, scale=comm0.std, size=(R, n1, d_feat))
    X0_c1 = rng.normal(loc=comm1.mean, scale=comm1.std, size=(R, n2, d_feat))
    X0 = np.concatenate([X0_c0, X0_c1], axis=1).astype(np.float64)
    return torch.tensor(X0, dtype=torch.float64)

# ------------------------------------------------------------
# Vectorized EM simulation for a batch of trajectories
# ------------------------------------------------------------

def _simulate_trajectories_batch(
    X0_batch: torch.Tensor,   # [R, N, D]
    A_sub: torch.Tensor,      # [N, N]
    T: int,
    c: float,
    gamma: float,
    sigma: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Vectorized Euler–Maruyama simulation for R trajectories in one pass.
    Returns:
      traj: [R, T+1, N, D]
      L_gamma: [N, N]
    """
    assert X0_batch.dim() == 3, "X0_batch must be [R, N, D]"
    R, N, D = X0_batch.shape
    device, dtype = X0_batch.device, X0_batch.dtype

    A = A_sub.to(device=device, dtype=dtype)
    I = torch.eye(N, device=device, dtype=dtype)
    Dg = torch.diag(A.sum(dim=1))
    L  = Dg - A
    Lg = L + gamma * I

    dt = 1.0 / T
    G  = sigma * math.sqrt(2.0 * c)

    traj = torch.empty((R, T+1, N, D), dtype=dtype, device=device)
    traj[:, 0] = X0_batch

    x = X0_batch
    for k in range(T):
        z = torch.randn(R, N, D, device=device, dtype=dtype)
        drift  = -c * (Lg @ x)
        x = x + drift * dt + (G * math.sqrt(dt)) * z
        traj[:, k+1] = x

    return traj, Lg

def _final_targets(L_gamma: torch.Tensor, sigma: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Steady-state targets for OU:
      Σ∞ = σ^2 (Lγ)^{-1},  μ∞=0
    """
    N = L_gamma.size(0)
    device, dtype = L_gamma.device, L_gamma.dtype
    I = torch.eye(N, device=device, dtype=dtype)
    Sig_final = sigma**2 * torch.linalg.solve(L_gamma, I)
    Sig_final = 0.5 * (Sig_final + Sig_final.T)
    mu_final  = torch.zeros(N, dtype=dtype, device=device)
    return Sig_final, mu_final

def _empirical_mean_cov_at_k(Xk: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Xk: [R, N, D]
    Returns: (mean_k [N,D], cov_k [N,N])
    """
    R, N, D = Xk.shape
    mean_k = Xk.mean(dim=0)
    X_flat = Xk.permute(0, 2, 1).reshape(R * D, N)
    Xc = X_flat - X_flat.mean(dim=0, keepdim=True)
    cov_k = (Xc.T @ Xc) / (R * D - 1)
    cov_k = 0.5 * (cov_k + cov_k.T)
    return mean_k, cov_k

def _curvature_penalty(y: np.ndarray) -> float:
    if len(y) < 3:
        return 0.0
    d2 = y[:-2] - 2.0 * y[1:-1] + y[2:]
    return float(np.mean(d2**2))

def _line_fit_mse_to_target(y: np.ndarray) -> float:
    """
    Normalize to [0,1] and penalize MSE to the target straight line 1 - t.
    """
    eps = 1e-12
    y = y.copy()
    y -= y[-1]                      # remove final offset
    denom = max(y[0], eps)
    y /= denom                      # normalized error: 1 at k=0, ~0 at k=T
    t = np.linspace(0.0, 1.0, len(y))
    target = 1.0 - t                # perfect straight line with slope -1 (over [0,1])
    return float(np.mean((y - target) ** 2))

def _final_relative_residual(y: np.ndarray) -> float:
    """
    Relative final error: y_T / (y_0 + eps). Small => reached final target well.
    """
    eps = 1e-12
    return float(y[-1] / (y[0] + eps))

@dataclass
class SearchConfig:
    R_eval: int = 64
    T_eval: int = 1000
    w_line: float = 1.0
    w_curv: float = 0.2
    w_final: float = 2.0
    # NEW: default to SBM signal model
    x0_mode: str = "sbm"            # 'sbm' or 'zeros'
    sizes: Tuple[int, int] = (5, 5) # communities
    comm0: CommunitySpec = CommunitySpec(mean=-3.0, std=0.25)
    comm1: CommunitySpec = CommunitySpec(mean=+3.5, std=0.30)
    # Grids
    c_grid: Optional[List[float]] = None
    gamma_grid: Optional[List[float]] = None
    sigma_grid: Optional[List[float]] = None
    # RNG & output
    seed: int = 42
    out_dir: Optional[str] = None
    # SBM topology parameters (for completeness; topology built outside)
    p_intra: float = 0.85
    p_inter: float = 0.05

def _make_X0_batch(
    R: int,
    n_nodes: int,
    d_feat: int,
    mode: str,
    seed: int,
    sizes: Tuple[int, int],
    comm0: CommunitySpec,
    comm1: CommunitySpec
) -> torch.Tensor:
    if mode == "zeros":
        return torch.zeros((R, n_nodes, d_feat), dtype=torch.float64)
    # default: "sbm"
    return _make_X0_batch_sbm(R, sizes, d_feat, comm0, comm1, seed=seed)

def _evaluate_params_once(
    adj: torch.Tensor,
    n_nodes: int,
    d_feat: int,
    c: float, gamma: float, sigma: float,
    cfg: SearchConfig,
    device: str = "cpu",
) -> Dict[str, float]:
    """
    Simulate trajectories, compute error curves vs steady-state target,
    and return the scalar objective + breakdown.
    """
    X0_batch = _make_X0_batch(
        cfg.R_eval, n_nodes, d_feat,
        mode=cfg.x0_mode, seed=cfg.seed,
        sizes=cfg.sizes, comm0=cfg.comm0, comm1=cfg.comm1
    ).to(device)

    traj, Lg = _simulate_trajectories_batch(X0_batch, adj.to(device), cfg.T_eval, c, gamma, sigma)

    # Steady-state targets
    Sig_final, _ = _final_targets(Lg, sigma)

    TP1 = cfg.T_eval + 1
    cov_err = np.zeros(TP1, dtype=np.float64)
    mean_err = np.zeros(TP1, dtype=np.float64)

    for k in range(TP1):
        Xk = traj[:, k]                                # [R, N, D]
        mean_k, cov_k = _empirical_mean_cov_at_k(Xk)   # [N,D], [N,N]
        mean_err[k] = float(torch.norm(mean_k.reshape(-1), p=2).item())
        cov_err[k]  = float(torch.norm(cov_k - Sig_final, p='fro').item())

    mse_line_cov  = _line_fit_mse_to_target(cov_err)
    mse_line_mean = _line_fit_mse_to_target(mean_err)
    curv_cov  = _curvature_penalty(cov_err)
    curv_mean = _curvature_penalty(mean_err)
    fin_cov   = _final_relative_residual(cov_err)
    fin_mean  = _final_relative_residual(mean_err)

    obj = (cfg.w_line  * (mse_line_cov + mse_line_mean)
         + cfg.w_curv  * (curv_cov + curv_mean)
         + cfg.w_final * (fin_cov + fin_mean))

    # Spectral radius of exact one-step tilde_alpha with dt=1/T_eval
    dt = 1.0 / cfg.T_eval
    with torch.no_grad():
        tilde_alpha = torch.matrix_exp(-c * Lg * dt)
        rho = np.max(np.abs(np.linalg.eigvals(tilde_alpha.detach().cpu().numpy())))
        rho = float(np.real(rho))

    return dict(
        obj=obj,
        mse_line_cov=mse_line_cov, mse_line_mean=mse_line_mean,
        curv_cov=curv_cov, curv_mean=curv_mean,
        fin_cov=fin_cov, fin_mean=fin_mean,
        rho_tilde_alpha=rho
    )

def search_optimal_params(
    adj: torch.Tensor,
    n_nodes: int,
    d_feat: int,
    cfg: Optional[SearchConfig] = None,
    device: str = "cpu",
) -> Tuple[Tuple[float,float,float], Dict[str, float]]:
    """
    Grid-search over (c, gamma, sigma) to minimize the linearity+final-match objective.
    Returns:
      best_tuple: (c*, gamma*, sigma*)
      best_report: metrics dict for the winner
    """
    if cfg is None:
        cfg = SearchConfig()

    # Default log-spaced grids
    if cfg.c_grid is None:
        cfg.c_grid = list(10.0 ** np.linspace(-2, 1, 7))      # 0.01 .. 10
    if cfg.gamma_grid is None:
        cfg.gamma_grid = list(10.0 ** np.linspace(-2, 0.7, 7)) # 0.01 .. ~5
    if cfg.sigma_grid is None:
        cfg.sigma_grid = list(10.0 ** np.linspace(-1, 1, 7))   # 0.1  .. 10

    best_tuple = None
    best_obj = float("inf")
    best_report = None
    tried = 0

    for c in cfg.c_grid:
        for gamma in cfg.gamma_grid:
            for sigma in cfg.sigma_grid:
                tried += 1
                rep = _evaluate_params_once(adj, n_nodes, d_feat, c, gamma, sigma, cfg, device=device)
                if rep["obj"] < best_obj:
                    best_tuple = (c, gamma, sigma)
                    best_obj = rep["obj"]
                    best_report = rep

                print(f"[{tried}] c={c:.4g}, gamma={gamma:.4g}, sigma={sigma:.4g}  "
                      f"obj={rep['obj']:.4e} | "
                      f"line(cov)={rep['mse_line_cov']:.2e}, line(mean)={rep['mse_line_mean']:.2e}, "
                      f"final(cov)={rep['fin_cov']:.2e}, final(mean)={rep['fin_mean']:.2e}, "
                      f"curv(cov)={rep['curv_cov']:.2e}, curv(mean)={rep['curv_mean']:.2e}, "
                      f"rho(tilde_alpha)={rep['rho_tilde_alpha']:.4f}")

    print("\n=== BEST PARAMS ===")
    print(f"c*={best_tuple[0]:.6g}, gamma*={best_tuple[1]:.6g}, sigma*={best_tuple[2]:.6g}")
    print("Objective breakdown:", best_report)

    # Optional: plots for the winner
    if cfg.out_dir is not None:
        os.makedirs(cfg.out_dir, exist_ok=True)
        X0_batch = _make_X0_batch(
            cfg.R_eval, n_nodes, d_feat, mode=cfg.x0_mode, seed=cfg.seed,
            sizes=cfg.sizes, comm0=cfg.comm0, comm1=cfg.comm1
        )
        traj, Lg = _simulate_trajectories_batch(X0_batch, adj, cfg.T_eval, *best_tuple)
        Sig_final, _ = _final_targets(Lg, best_tuple[2])

        cov_err, mean_err = [], []
        for k in range(cfg.T_eval + 1):
            Xk = traj[:, k]
            mean_k, cov_k = _empirical_mean_cov_at_k(Xk)
            cov_err.append(float(torch.norm(cov_k - Sig_final, p='fro').item()))
            mean_err.append(float(torch.norm(mean_k.reshape(-1), p=2).item()))
        cov_err = np.array(cov_err); mean_err = np.array(mean_err)

        def _norm_curve(y):
            y = y.copy()
            y -= y[-1]
            y /= max(y[0], 1e-12)
            return y

        cov_norm = _norm_curve(cov_err)
        mean_norm = _norm_curve(mean_err)
        t = np.linspace(0.0, 1.0, len(cov_norm))
        target = 1.0 - t

        plt.figure()
        plt.plot(t, cov_norm, label="cov error (norm.)")
        plt.plot(t, target, linestyle="--", label="target 1 - t")
        plt.xlabel("normalized time t")
        plt.ylabel("normalized error")
        plt.title(f"Cov trajectory vs target | c={best_tuple[0]:.3g}, gamma={best_tuple[1]:.3g}, sigma={best_tuple[2]:.3g}")
        plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(cfg.out_dir, "best_cov_norm_vs_target.png"), dpi=150); plt.close()

        plt.figure()
        plt.plot(t, mean_norm, label="mean error (norm.)")
        plt.plot(t, target, linestyle="--", label="target 1 - t")
        plt.xlabel("normalized time t")
        plt.ylabel("normalized error")
        plt.title(f"Mean trajectory vs target | c={best_tuple[0]:.3g}, gamma={best_tuple[1]:.3g}, sigma={best_tuple[2]:.3g}")
        plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(cfg.out_dir, "best_mean_norm_vs_target.png"), dpi=150); plt.close()

    return best_tuple, best_report

# ------------------------------------------------------------
# Main — build SBM topology, search params, long run diagnostics
# ------------------------------------------------------------

if __name__ == "__main__":
    # ===== SBM topology & signal specs (match your dataset) =====
    d_feat  = 10
    sizes   = (5, 5)
    p_intra = 0.85
    p_inter = 0.05
    seed    = 42

    # Community Gaussians (WELL-SEPARATED)
    comm0 = CommunitySpec(mean=-3.0, std=0.25)
    comm1 = CommunitySpec(mean=+3.5, std=0.30)

    # Build ONE fixed SBM topology + one sample of features (we only need A & N)
    Adj, X0_sample, community = build_sbm_dataset(
        sizes=sizes,
        d_feat=d_feat,
        p_intra=p_intra,
        p_inter=p_inter,
        seed=seed,
        comm0=comm0,
        comm1=comm1,
        return_pyg=False
    )
    Adj = Adj.to(torch.float64)  # [N,N]
    N = Adj.shape[0]

    # ==== 1) Search best (c, gamma, sigma) for near-linear decay & steady-state match ====
    cfg = SearchConfig(
        R_eval=64,
        T_eval=1000,
        w_line=1.0,
        w_curv=0.2,
        w_final=2.0,
        # grids
        c_grid     = list(10.0 ** np.linspace(-2, 1, 7)),     # 0.01..10
        gamma_grid = list(10.0 ** np.linspace(-2, 0.7, 7)),   # 0.01..~5
        sigma_grid = list(10.0 ** np.linspace(-1, 1, 7)),     # 0.1..10
        # SBM signal model for X0
        x0_mode="sbm",
        sizes=sizes,
        comm0=comm0,
        comm1=comm1,
        seed=seed,
        out_dir="./ParamSearchResults_SBM",
        p_intra=p_intra,
        p_inter=p_inter
    )

    (c_star, gamma_star, sigma_star), report = search_optimal_params(
        adj=Adj, n_nodes=N, d_feat=d_feat, cfg=cfg, device="cpu"
    )

    print("\nSelected hyperparameters:")
    print("c*     =", c_star)
    print("gamma* =", gamma_star)
    print("sigma* =", sigma_star)
    print("Report =", report)

    # ==== 2) Re-simulate longer trajectories with the best params & save diagnostics ====
    R_long = 300
    T_long = 1000
    out_dir = "./ForwardDiagnostics_SBM"
    os.makedirs(out_dir, exist_ok=True)

    # Sample a single X0 from SBM Gaussians for the long run (N, D)
    X0_long = _make_X0_batch_sbm(1, sizes, d_feat, comm0, comm1, seed=seed)[0]  # [N,D]

    all_traj = []
    cache = None
    for i in range(R_long):
        traj_i, cache = forward_diffusion_exact(
            X_sub=X0_long, A_sub=Adj, T=T_long, c=c_star, gamma=gamma_star, sigma=sigma_star,
            return_trajectory=True
        )
        all_traj.append(traj_i)              # [T+1, N, D]
        if (i + 1) % 50 == 0:
            print(f"Generated {i+1}/{R_long} trajectories")

    all_traj = torch.stack(all_traj, dim=0)  # [R, T+1, N, D]
    _, L_gamma, tilde_alpha, alpha, _ = cache

    ts_res = compute_time_series_errors_against_final(
        trajectories=all_traj,
        L_gamma=L_gamma,
        alpha=alpha,
        sigma=sigma_star,
        T=T_long,
        c=c_star,
        gamma=gamma_star,
        out_dir=out_dir
    )
    print(f"Saved mean error curve to: {ts_res['mean_plot_path']}")
    print(f"Saved covariance error curves to: {ts_res['cov_plot_path']}")
