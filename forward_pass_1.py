# forward_timewarp_minimal.py
# -----------------------------------------------------------------------------
# Time‑warp OU forward process on graphs (no hyperparameter search).
# - Builds a simple SBM graph (internal; no external deps)
# - Simulates forward process with exact OU increments under a warp τ(t)
# - Plots:
#     (1) mean/cov error trajectories vs final theoretical Σ_T (finite τ_T),
#     (2) τ_k vs k
# - Prints final-step ||μ_T^emp - 0||_2 and ||Σ_T^emp - Σ_T^theory||_F,
#   and also reports numbers vs Σ_∞ = σ^2 L_γ^{-1} for reference.
# -----------------------------------------------------------------------------

import os
import math
from dataclasses import dataclass
from typing import Tuple, List, Dict, Optional

import numpy as np
import torch
import matplotlib.pyplot as plt

torch.set_default_dtype(torch.float64)
torch.set_printoptions(precision=6, sci_mode=False, threshold=10000)

# =============================================================================
# Utilities: SBM graph, Laplacian, τ-schedules
# =============================================================================

def build_simple_sbm_adjacency(
    sizes: Tuple[int, int],
    p_intra: float = 0.85,
    p_inter: float = 0.05,
    seed: int = 42,
) -> torch.Tensor:
    """
    Build a 2-block SBM adjacency (undirected, no self-loops) without networkx.
    """
    rng = np.random.default_rng(seed)
    n1, n2 = sizes
    N = n1 + n2
    A = np.zeros((N, N), dtype=np.float64)

    # Block 1
    mask = rng.random((n1, n1)) < p_intra
    A[:n1, :n1] = np.triu(mask, 1)
    A[:n1, :n1] += A[:n1, :n1].T

    # Block 2
    mask = rng.random((n2, n2)) < p_intra
    A[n1:, n1:] = np.triu(mask, 1)
    A[n1:, n1:] += A[n1:, n1:].T

    # Cross-block
    mask = rng.random((n1, n2)) < p_inter
    A[:n1, n1:] = mask
    A[n1:, :n1] = mask.T

    np.fill_diagonal(A, 0.0)
    return torch.from_numpy(A)

def sym_norm_laplacian(A: torch.Tensor) -> torch.Tensor:
    """
    L_sym = I - D^{-1/2} A D^{-1/2}. Handles isolated nodes robustly.
    """
    A = A.to(dtype=torch.get_default_dtype())
    N = A.shape[0]
    deg = A.sum(dim=1)
    inv_sqrt_deg = torch.where(deg > 0, deg.rsqrt(), torch.zeros_like(deg))
    Dm12 = torch.diag(inv_sqrt_deg)
    I = torch.eye(N, dtype=A.dtype, device=A.device)
    return I - Dm12 @ A @ Dm12

def build_tau(T: int, schedule: str = "loglinear", tau_T: float = 5.0, poly_p: float = 0.5) -> np.ndarray:
    """
    Return τ_k for k=0..T with τ_0=0 and τ_T given.
    Schedules (normalized t∈[0,1]):
      • cosine:    s(t) = 0.5 * (1 - cos(pi * t))
      • poly(p):   s(t) = t**p
      • loglinear: s(t) = -log(1 - (1 - eps) * t)/-log(eps)  (near-linear)
    """
    t = np.linspace(0.0, 1.0, T + 1)
    if schedule == "cosine":
        s = 0.5 * (1.0 - np.cos(np.pi * t))
    elif schedule == "poly":
        s = np.power(t, poly_p)
    elif schedule in ("uniform", "linear"):                 
        s = t                                               
    elif schedule == "loglinear":
        eps = 1e-2
        s = -np.log(1.0 - (1.0 - eps) * t + 1e-12) / -np.log(eps)
    else:
        raise ValueError(f"Unknown schedule: {schedule}")
    tau = tau_T * s
    tau = np.maximum.accumulate(tau)  # guard monotonicity
    tau[0] = 0.0
    return tau


# =============================================================================
# L_gamma^{-1} utilities (stable inverse + heatmap)
# =============================================================================

def stable_spd_inverse(M: torch.Tensor) -> torch.Tensor:
    """
    Numerically stable inverse of an SPD matrix using Cholesky.
    Returns a symmetrized result.
    """
    chol = torch.linalg.cholesky(M)                  # M = R R^T
    Minv = torch.cholesky_inverse(chol)              # (R R^T)^{-1}
    Minv = 0.5 * (Minv + Minv.T)                     # symmetrize (paranoia)
    return Minv

def save_matrix_heatmap(
    M: torch.Tensor,
    out_dir: str,
    fname: str,
    title: str,
    center_zero: bool = True
) -> str:
    """
    Save a heatmap for matrix M to disk and return the path.
    """
    os.makedirs(out_dir, exist_ok=True)
    M_np = M.detach().cpu().numpy()
    vmax = np.abs(M_np).max()
    vmin = -vmax if center_zero else M_np.min()
    vmax =  vmax if center_zero else M_np.max()

    plt.figure()
    im = plt.imshow(M_np, interpolation="nearest", aspect="auto",
                    vmin=vmin, vmax=vmax, cmap="coolwarm")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.title(title)
    plt.xlabel("column")
    plt.ylabel("row")
    path = os.path.join(out_dir, fname)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()
    return path



# =============================================================================
# Time-warp forward process (exact OU increments)
# =============================================================================

def _precompute_step_ops(Lg: torch.Tensor, c: float, sigma: float, dtaus: np.ndarray,
                         eps: float = 1e-10) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """
    For each Δτ_k, compute:
      A_k = exp(-c Lg Δτ_k),
      chol(Q_k)  with  Q_k = σ^2 (I - A_k^2) Lg^{-1}.
    """
    device, dtype = Lg.device, Lg.dtype
    N = Lg.size(0)
    I = torch.eye(N, device=device, dtype=dtype)

    A_list: List[torch.Tensor] = []
    chol_list: List[torch.Tensor] = []

    # Solve with Lg (spd) instead of explicit inverse; symmetrize and jitter for PD
    for dτ in dtaus:
        dτ_t = torch.tensor(float(dτ), device=device, dtype=dtype)
        A = torch.matrix_exp(-c * Lg * dτ_t)                  # [N,N]
        A2 = A @ A
        Q = sigma**2 * torch.linalg.solve(Lg, (I - A2))       # [N,N]
        Q = 0.5 * (Q + Q.T)
        Q = Q + eps * I                                       # PD guard
        chol = torch.linalg.cholesky(Q)
        A_list.append(A)
        chol_list.append(chol)
    return A_list, chol_list

@torch.no_grad()
def simulate_timewarp_batch(
    X0_batch: torch.Tensor,   # [R, N, D]
    A_adj: torch.Tensor,      # [N, N] adjacency
    T: int,
    c: float,
    gamma: float,
    sigma: float,
    tau: np.ndarray,          # length T+1
) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    """
    Vectorized exact simulator for the time‑warp OU process:
      x_{k+1} = A_k x_k + ξ_k,  A_k = exp(-c L_γ Δτ_k), ξ_k ~ N(0, Q_k)
    Returns: (traj [R, T+1, N, D], L_gamma [N,N], tau [T+1])
    """
    assert X0_batch.dim() == 3, "X0_batch must be [R, N, D]"
    R, N, D = X0_batch.shape
    device, dtype = X0_batch.device, X0_batch.dtype

    A = A_adj.to(device=device, dtype=dtype)
    L_sym = sym_norm_laplacian(A)
    Lg = L_sym + gamma * torch.eye(N, device=device, dtype=dtype)

    dtaus = np.diff(tau)  # length T
    step_As, step_chols = _precompute_step_ops(Lg, c, sigma, dtaus)

    traj = torch.empty((R, T + 1, N, D), dtype=dtype, device=device)
    x = X0_batch.clone()
    traj[:, 0] = x

    for k in range(T):
        Ak = step_As[k]
        Ck = step_chols[k]
        # Deterministic propagation
        x = torch.einsum('ij,rjd->rid', Ak, x)
        # Correlated OU noise increment
        z = torch.randn(R, N, D, device=device, dtype=dtype)
        xi = torch.einsum('ij,rjd->rid', Ck, z)
        x = x + xi
        traj[:, k + 1] = x

    return traj, Lg, tau

# =============================================================================
# Theory & stats
# =============================================================================

def theoretical_cov_timewarp(Lg: torch.Tensor, sigma: float, c: float, tau_k: float) -> torch.Tensor:
    """Σ_k = σ^2 (I - exp(-2 c Lg τ_k)) Lg^{-1} (symmetrized)."""
    device, dtype = Lg.device, Lg.dtype
    N = Lg.size(0)
    I = torch.eye(N, device=device, dtype=dtype)
    E = torch.matrix_exp(-2.0 * c * Lg * torch.tensor(float(tau_k), device=device, dtype=dtype))
    Sig = sigma**2 * torch.linalg.solve(Lg, (I - E))
    Sig = 0.5 * (Sig + Sig.T)
    return Sig

def theoretical_cov_stationary(Lg: torch.Tensor, sigma: float) -> torch.Tensor:
    """Σ_∞ = σ^2 Lγ^{-1} (symmetrized)."""
    Sig_inf = sigma**2 * torch.linalg.inv(Lg)
    Sig_inf = 0.5 * (Sig_inf + Sig_inf.T)
    return Sig_inf

def empirical_mean_and_cov_over_samples(X_batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    X_batch: [R, N, D] → mean over samples, and node-cov over flattened samples.
    Returns:
      mean_k: [N, D]
      cov_k : [N, N] (across R*D samples)
    """
    R, N, D = X_batch.shape
    mean_k = X_batch.mean(dim=0)
    X_flat = X_batch.permute(0, 2, 1).reshape(R * D, N)
    Xc = X_flat - X_flat.mean(dim=0, keepdim=True)
    cov_k = (Xc.T @ Xc) / (R * D - 1)
    cov_k = 0.5 * (cov_k + cov_k.T)
    return mean_k, cov_k

# =============================================================================
# Plotting & diagnostics
# =============================================================================

def plot_tau_curve(tau: np.ndarray, out_dir: str, label: str = "") -> str:
    os.makedirs(out_dir, exist_ok=True)
    x = np.arange(len(tau))
    plt.figure()
    plt.plot(x, tau, label=r"$\tau_k$")
    plt.xlabel("timestep k")
    plt.ylabel(r"$\tau_k$")
    title = "Warp clock τ vs steps"
    if label:
        title += f" | {label}"
    plt.title(title)
    plt.legend(loc="best", frameon=True)
    path = os.path.join(out_dir, "tau_vs_steps.png")
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()
    return path

def compute_time_series_errors_against_final_timewarp(
    trajectories: torch.Tensor,           # [R, T+1, N, D]
    L_gamma: torch.Tensor,
    sigma: float,
    c: float,
    tau: np.ndarray,
    out_dir: str = ".",
    label: str = "",
) -> Dict[str, object]:
    """
    For each k, compare empirical mean/cov to final theoretical target at step T only:
      mean_err_l2[k] = ||E[x_k] - μ_T||_2   with μ_T = 0
      cov_err_fro[k] = ||Σ_k - Σ_T||_F
      cov_err_rel[k] = cov_err_fro / ||Σ_T||_F
    Saves plots and returns arrays.
    """
    device, dtype = trajectories.device, trajectories.dtype
    R, TP1, N, D = trajectories.shape
    assert TP1 == len(tau)

    Sigma_T = theoretical_cov_timewarp(L_gamma, sigma, c, tau[-1])
    Sigma_T_fro = torch.norm(Sigma_T, p='fro') + 1e-12
    mu_T = torch.zeros(N * D, device=device, dtype=dtype)

    mean_err_l2 = np.zeros(TP1)
    cov_err_fro  = np.zeros(TP1)
    cov_err_rel  = np.zeros(TP1)

    for k in range(TP1):
        Xk = trajectories[:, k]
        mean_k, cov_k = empirical_mean_and_cov_over_samples(Xk)
        mean_diff = mean_k.reshape(-1) - mu_T
        mean_err_l2[k] = float(torch.norm(mean_diff, p=2).item())
        cov_diff = (cov_k - Sigma_T)
        fro = torch.norm(cov_diff, p='fro')
        cov_err_fro[k] = float(fro.item())
        cov_err_rel[k] = float((fro / Sigma_T_fro).item())

    os.makedirs(out_dir, exist_ok=True)
    t_axis = np.arange(TP1)
    title_suffix = f"T={TP1-1}, c={c:.3g}, sigma={sigma:.3g} | {label}".strip()

    plt.figure()
    plt.plot(t_axis, mean_err_l2, label=r"$\|E[x_k]-\mu_T\|_2$")
    plt.xlabel("timestep k"); plt.ylabel("mean L2 error");
    plt.title("Mean error vs final theoretical mean");
    plt.legend(title=title_suffix, loc="best", frameon=True)
    mean_plot_path = os.path.join(out_dir, f"timewarp_mean_error_vs_time.png")
    plt.tight_layout(); plt.savefig(mean_plot_path, dpi=150); plt.close()

    plt.figure()
    plt.plot(t_axis, cov_err_fro, label=r"$\|\Sigma_k-\Sigma_T\|_F$")
    plt.plot(t_axis, cov_err_rel, label=r"rel. Fro error")
    plt.xlabel("timestep k"); plt.ylabel("error");
    plt.title("Covariance error vs final theoretical covariance");
    plt.legend(title=title_suffix, loc="best", frameon=True)
    cov_plot_path = os.path.join(out_dir, f"timewarp_cov_error_vs_time.png")
    plt.tight_layout(); plt.savefig(cov_plot_path, dpi=150); plt.close()

    return {
        "mean_err_l2": mean_err_l2,
        "cov_err_fro": cov_err_fro,
        "cov_err_rel": cov_err_rel,
        "mean_plot_path": mean_plot_path,
        "cov_plot_path": cov_plot_path,
        "Sigma_T": Sigma_T,
    }

# =============================================================================
# X0 samplers (SBM features)
# =============================================================================

@dataclass
class CommunitySpec:
    mean: float
    std: float

def make_X0_batch_sbm(R: int, sizes: Tuple[int, int], d_feat: int,
                      comm0: CommunitySpec, comm1: CommunitySpec, seed: int = 42) -> torch.Tensor:
    n1, n2 = sizes
    rng = np.random.default_rng(seed)
    X0_c0 = rng.normal(loc=comm0.mean, scale=comm0.std, size=(R, n1, d_feat))
    X0_c1 = rng.normal(loc=comm1.mean, scale=comm1.std, size=(R, n2, d_feat))
    X0 = np.concatenate([X0_c0, X0_c1], axis=1).astype(np.float64)
    return torch.tensor(X0, dtype=torch.float64)

# =============================================================================
# Entry: single-run diagnostics for fixed hyperparameters
# =============================================================================

@torch.no_grad()
def run_fixed_params_and_plot(
    *,
    # Graph & signal
    sizes: Tuple[int, int] = (5, 5),
    p_intra: float = 0.85,
    p_inter: float = 0.05,
    d_feat: int = 10,
    R: int = 256,
    seed: int = 42,
    comm0: CommunitySpec = CommunitySpec(mean=-3.0, std=0.25),
    comm1: CommunitySpec = CommunitySpec(mean=+3.5, std=0.30),
    # Hyperparameters
    T: int = 1000,
    c: float = 1.0,
    gamma: float = 0.1,
    sigma: float = 1.0,
    warp: str = "loglinear",     # 'loglinear' | 'cosine' | 'poly'
    tau_T: float = 5.0,
    poly_p: float = 0.5,
    # Output
    out_dir: str = "./Timewarp_FixedParam_Run",
) -> Dict[str, object]:

    # --- Build graph & initial batch
    Adj = build_simple_sbm_adjacency(sizes, p_intra=p_intra, p_inter=p_inter, seed=seed)
    X0_batch = make_X0_batch_sbm(R, sizes, d_feat, comm0, comm1, seed=seed)

    # --- Build τ and simulate
    tau = build_tau(T, schedule=warp, tau_T=tau_T, poly_p=poly_p)
    traj, L_gamma, tau = simulate_timewarp_batch(X0_batch, Adj, T, c, gamma, sigma, tau)

    # --- Compute and save L_gamma^{-1} + print diagnostics
    Lgamma_inv = stable_spd_inverse(L_gamma)
    inv_path = save_matrix_heatmap(
        Lgamma_inv,
        out_dir=out_dir,
        fname="Lgamma_inv_heatmap.png",
        title=r"$L_\gamma^{-1}$ heatmap",
        center_zero=True
    )

    # Print the matrix (small graphs) and some quick stats
    N = L_gamma.size(0)
    with torch.no_grad():
        print("\n--- L_gamma and its inverse diagnostics ---")
        print(f"Matrix size N={N}")
        # Print the full inverse for small N to keep logs readable
        if N <= 20:
            print("L_gamma^{-1} =\n", Lgamma_inv)
        else:
            print("L_gamma^{-1} (not printed in full since N>20)")
        eigvals = torch.linalg.eigvalsh(L_gamma)
        cond_num = (eigvals.max() / eigvals.min()).item()
        print(f"eig(L_gamma) min={eigvals.min().item():.6e}, max={eigvals.max().item():.6e}")
        print(f"Estimated condition number κ(L_gamma) ≈ {cond_num:.6e}")
        print(f"Saved L_gamma^{-1} heatmap: {inv_path}\n")


    # --- Time-series diagnostics vs final finite-τ_T theory
    label = f"gamma={gamma:.3g}, sigma={sigma:.3g}, warp={warp}, tau_T={tau_T}"
    ts = compute_time_series_errors_against_final_timewarp(
        trajectories=traj, L_gamma=L_gamma, sigma=sigma, c=c, tau=tau,
        out_dir=out_dir, label=label
    )

    # --- τ curve plot
    tau_plot_path = plot_tau_curve(tau, out_dir, label=f"warp={warp}, tau_T={tau_T}")

    # --- Final-step empirical stats
    X_T = traj[:, -1]
    mean_T_emp, cov_T_emp = empirical_mean_and_cov_over_samples(X_T)

    # --- Theoretical references
    Sig_T   = theoretical_cov_timewarp(L_gamma, sigma, c, tau[-1])
    Sig_inf = theoretical_cov_stationary(L_gamma, sigma)

    mu_T_theory = torch.zeros_like(mean_T_emp).reshape(-1)

    mean_l2_vs_T   = float(torch.norm(mean_T_emp.reshape(-1) - mu_T_theory, p=2).item())
    cov_fro_vs_T   = float(torch.norm(cov_T_emp - Sig_T, p='fro').item())
    cov_fro_vs_inf = float(torch.norm(cov_T_emp - Sig_inf, p='fro').item())

    rel_cov_vs_T   = float((torch.norm(cov_T_emp - Sig_T, p='fro') / (torch.norm(Sig_T, p='fro') + 1e-12)).item())
    rel_cov_vs_inf = float((torch.norm(cov_T_emp - Sig_inf, p='fro') / (torch.norm(Sig_inf, p='fro') + 1e-12)).item())

    # --- Print requested norms
    print("\n=== Fixed-parameter diagnostics ===")
    print(f"Graph: SBM sizes={sizes}, p_intra={p_intra}, p_inter={p_inter}")
    print(f"T={T}, c={c}, gamma={gamma}, sigma={sigma} | warp={warp}, tau_T={tau_T}")
    print(f"τ_T = {tau[-1]:.6g}")
    print(f"Final-step mean  ||μ_T^emp - 0||_2:           {mean_l2_vs_T:.6e}")
    print(f"Final-step cov   ||Σ_T^emp - Σ_T||_F:          {cov_fro_vs_T:.6e}  (rel={rel_cov_vs_T:.3%})")
    print(f"Final-step cov   ||Σ_T^emp - Σ_∞||_F:          {cov_fro_vs_inf:.6e} (rel={rel_cov_vs_inf:.3%})")
    print(f"Saved τ plot:                     {tau_plot_path}")
    print(f"Saved mean error curve:           {ts['mean_plot_path']}")
    print(f"Saved covariance error curves:    {ts['cov_plot_path']}\n")

    return dict(
        tau=tau,
        tau_plot=tau_plot_path,
        trajectories=traj,
        L_gamma=L_gamma,
        Sigma_T=Sig_T,
        Sigma_inf=Sig_inf,
        mean_err_l2_ts=ts["mean_err_l2"],
        cov_err_fro_ts=ts["cov_err_fro"],
        cov_err_rel_ts=ts["cov_err_rel"],
        final_mean_l2_vs_T=mean_l2_vs_T,
        final_cov_fro_vs_T=cov_fro_vs_T,
        final_cov_fro_vs_inf=cov_fro_vs_inf,
        rel_cov_vs_T=rel_cov_vs_T,
        rel_cov_vs_inf=rel_cov_vs_inf,
        mean_plot_path=ts["mean_plot_path"],
        cov_plot_path=ts["cov_plot_path"],
        L_gamma_inv=Lgamma_inv,
        L_gamma_inv_heatmap=inv_path,
    )

# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    # ------ Set your hyperparameters here ------
    sizes   = (5, 5)
    p_intra = 0.85
    p_inter = 0.20
    d_feat  = 10
    R       = 1024
    seed    = 42

    # Forward process hyperparameters
    T      = 1000
    c      = 1.6
    gamma  = 1.0
    sigma  = 0.01

    # Time-warp scheduler
    warp   = "loglinear"   # 'loglinear' | 'cosine' | 'poly'
    tau_T  = 3
    poly_p = 0.5

    out_dir = "/home/nfs/vkumarasamybal/Graph_Aware_Graph_Signal_Diffusion/DDPM_Graph_Trainer_Time_Wrap_SBM_Dataset_till_t_max/Output/Timewarp_FixedParam_Run_25"

    # ------ Run ------
    run_fixed_params_and_plot(
        sizes=sizes, p_intra=p_intra, p_inter=p_inter, d_feat=d_feat,
        R=R, seed=seed,
        comm0=CommunitySpec(mean=-3.0, std=0.25),
        comm1=CommunitySpec(mean=+3.5, std=0.30),
        T=T, c=c, gamma=gamma, sigma=sigma,
        warp=warp, tau_T=tau_T, poly_p=poly_p,
        out_dir=out_dir
    )
