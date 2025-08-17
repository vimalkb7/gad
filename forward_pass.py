import os
import torch
import numpy as np
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import networkx as nx
import math

torch.set_printoptions(threshold=float('inf'))

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def build_er_graph(n: int, p_edge: float, seed: int = 0):
    if seed is not None:
        np.random.seed(seed)
    G = nx.erdos_renyi_graph(n, p_edge, seed=seed)
    A = nx.to_numpy_array(G, dtype=float)
    return torch.tensor(A, dtype=torch.float64)

def sample_x0_from_gmm_single(n_nodes: int, d_feat: int,
                              weights=np.array([0.3, 0.4, 0.3]),
                              means=np.array([-5.0, 0.0, 4.0]),
                              stds=np.array([1.0, 0.8, 1.2]),
                              seed: int = 42) -> torch.Tensor:
    if seed is not None:
        np.random.seed(seed)
    K = len(weights)
    comps = np.random.choice(K, size=(n_nodes, d_feat), p=weights)
    mu   = means[comps]
    sd   = stds[comps]
    X0   = mu + sd * np.random.randn(n_nodes, d_feat)
    return torch.tensor(X0, dtype=torch.float64)

# ------------------------------------------------------------
# Exact Graph-OU forward (matches gaussian_diffusion.py)
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
    Exact Δt discretization per step (precomputes operators),
    but *evolves* with the equivalent Euler–Maruyama SDE:
        x_{k+1} = x_k + dt * (-c L_gamma x_k) + sqrt(2c)*sigma*sqrt(dt)*z_k
    Also returns the one-step exact matrices for theory checks.
    If return_trajectory=True, returns the whole path [T+1, N, D].
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
    # dt = 0.0002
    G = sigma * math.sqrt(2.0 * c)

    # Precompute exact one-step matrices for theory
    tilde_alpha = torch.matrix_exp(-c * L_gamma * dt)        # [N,N]
    alpha       = tilde_alpha @ tilde_alpha                  # [N,N]
    Sigma_delta = sigma**2 * torch.linalg.solve(L_gamma, (I - alpha))  # [N,N]

    # Simulate trajectory with EM (matches continuous SDE)
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
# Helper function for graph plots
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

    # Pool features as additional i.i.d. replicates for a stable node covariance
    X_flat = X_batch.permute(0, 2, 1).reshape(R * D, N)      # [R*D, N]
    X_centered = X_flat - X_flat.mean(dim=0, keepdim=True)
    cov_k = (X_centered.T @ X_centered) / (R * D - 1)        # [N, N]
    cov_k = 0.5 * (cov_k + cov_k.T)
    return mean_k, cov_k


def compute_time_series_errors_against_final(
    trajectories: torch.Tensor,           # [R, T+1, N, D]
    L_gamma: torch.Tensor,
    alpha: torch.Tensor,
    sigma: float,
    T: int,
    out_dir: str = "."
) -> dict:
    """
    For each timestep k=0..T, compute empirical mean & covariance across R samples,
    then compare to the *theoretical target at final step T only*:
        μ_T = 0                                (assuming x0 = 0)
        Σ_T = sigma^2 * (I - alpha^T) @ L_gamma^{-1}
    Saves plots:
      - mean_error_vs_time.png
      - cov_fro_error_vs_time.png
    """
    device, dtype = trajectories.device, trajectories.dtype
    R, TP1, N, D = trajectories.shape
    assert TP1 == T + 1

    I = torch.eye(N, device=device, dtype=dtype)
    alpha_T = torch.linalg.matrix_power(alpha, T)
    Sigma_T = sigma**2 * torch.linalg.solve(L_gamma, (I - alpha_T))
    Sigma_T = 0.5 * (Sigma_T + Sigma_T.T)

    # μ_T = 0 when x0 = 0
    mu_T = torch.zeros(N * D, device=device, dtype=dtype)

    mean_err_l2 = np.zeros(TP1)
    cov_err_fro  = np.zeros(TP1)
    cov_err_rel  = np.zeros(TP1)

    Sigma_T_fro = torch.norm(Sigma_T, p='fro') + 1e-12

    for k in range(TP1):
        Xk = trajectories[:, k]                     # [R, N, D]
        mean_k, cov_k = empirical_mean_and_cov_over_samples(Xk)

        # mean error vs μ_T = 0
        mean_diff = mean_k.reshape(-1) - mu_T
        mean_err_l2[k] = float(torch.norm(mean_diff, p=2).item())

        # covariance Fro error vs Σ_T
        cov_diff = (cov_k - Sigma_T)
        fro = torch.norm(cov_diff, p='fro')
        cov_err_fro[k] = float(fro.item())
        cov_err_rel[k] = float((fro / Sigma_T_fro).item())


    # --- Plots ---
    import matplotlib.pyplot as plt
    os.makedirs(out_dir, exist_ok=True)

    # Build a clean param string once
    param_str = f"T={T}, c={c:.3g}, γ={float(L_gamma.diag().mean().sub((L_gamma - torch.diag(L_gamma.diag())).mean()*0)+gamma):.3g}, σ={sigma:.3g}"
    # ↑ the γ in L_gamma is your scalar `gamma`; shown directly for clarity:
    param_str = f"T={T}, c={c:.3g}, γ={gamma:.3g}, σ={sigma:.3g}"

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
# Theoretical finite-k covariance Σ_k and check
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

def check_distribution_exact(
    samples,    # list of [N, D]
    L_gamma: torch.Tensor,
    alpha: torch.Tensor,
    sigma: float,
    T: int
) -> dict:
    """
    Compare empirical mean/covariance of x_T to Σ_T from exact formulas.
    Assumes x_0 = 0 so mean is 0 and Σ_T is as above.
    """
    X = torch.stack(samples, dim=0)  # [R, N, D]
    R, N, D = X.shape

    # Empirical mean per node/feature (should be ~0 if x0=0)
    empirical_mean = X.mean(dim=0)         # [N, D]
    pixel_wise_mean = empirical_mean.mean(dim=0)  # [D]
    max_abs_mean = empirical_mean.abs().max().item()

    # Empirical covariance across nodes, pooling features as i.i.d. trials
    X_flat = X.permute(0, 2, 1).reshape(R * D, N)     # [R*D, N]
    X_centered = X_flat - X_flat.mean(dim=0, keepdim=True)
    print("X_centered:")
    print(X_centered.shape)
    empirical_cov = (X_centered.T @ X_centered) / (R * D - 1)  # [N, N]

    # Theoretical Σ_T
    theoretical_cov = theoretical_covariance_exact(L_gamma, alpha, sigma, T)

    fro_error = torch.norm(empirical_cov - theoretical_cov, p='fro').item()
    return {
        'pixel_wise_mean': pixel_wise_mean,
        'max_abs_mean':    max_abs_mean,
        'empirical_cov':   empirical_cov,
        'theoretical_cov': theoretical_cov,
        'fro_error':       fro_error
    }


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

if __name__ == "__main__":
    # Config
    n_nodes = 5
    d_feat  = 10
    R       = 100
    seed    = 42

    T     = 10000
    c     = 2
    gamma = 0.3
    sigma = 6

    out_dir = "/tudelft.net/staff-bulk/ewi/insy/MMC/vimal/Results/Graph_Aware_Graph_Signal_Diffusion/Forward/Exp15"
    os.makedirs(out_dir, exist_ok=True)



    # Graph
    adj  = build_er_graph(n=n_nodes, p_edge=0.5, seed=seed)


    X0 = sample_x0_from_gmm_single(n_nodes = n_nodes, d_feat = d_feat)

    # --- Simulate R trajectories, storing all timesteps ---
    all_traj = []
    cache = None
    for i in range(R):
        traj_i, cache = forward_diffusion_exact(
            X_sub=X0, A_sub=adj, T=T, c=c, gamma=gamma, sigma=sigma,
            return_trajectory=True
        )
        all_traj.append(traj_i)              # [T+1, N, D]
        if (i + 1) % 50 == 0:
            print(f"Generated {i+1}/{R} trajectories")

    all_traj = torch.stack(all_traj, dim=0)  # [R, T+1, N, D]

    # Unpack theory bits
    _, L_gamma, tilde_alpha, alpha, Sigma_delta = cache

    # --- Your existing final-time check (kept intact) ---
    final_samples = [all_traj[i, -1] for i in range(R)]      # list of [N,D]
    results = check_distribution_exact(
        final_samples, L_gamma=L_gamma, alpha=alpha, sigma=sigma, T=T
    )

    print(f"Max absolute empirical mean: {results['max_abs_mean']:.4e}")
    print(f"Pixel-wise empirical mean (first 10 dims): {results['pixel_wise_mean'][:10]}")
    print("Empirical covariance matrix:")
    print(results['empirical_cov'])
    print("Theoretical covariance matrix (exact Σ_T):")
    print(results['theoretical_cov'])
    print(f"Frobenius norm error (emp vs theory): {results['fro_error']:.4e}")

    rel = torch.norm(results['empirical_cov'] - results['theoretical_cov'], p='fro') / \
          (torch.norm(results['theoretical_cov'], p='fro') + 1e-12)
    print("Relative Frobenius error:", rel.item())

    rho = np.max(np.abs(np.linalg.eigvals(tilde_alpha.detach().cpu().numpy())))
    print("Spectral radius ρ(tilde_alpha):", float(np.real(rho)))

    Sig_emp = results['empirical_cov'].to(torch.float64).clone()
    Sig_th  = results['theoretical_cov'].to(torch.float64).clone()
    eps = 1e-8
    I = torch.eye(Sig_emp.size(0), dtype=Sig_emp.dtype, device=Sig_emp.device)
    Sig_emp = Sig_emp + eps * I
    Sig_th  = Sig_th  + eps * I
    k_dim = Sig_emp.size(0)
    KL = 0.5 * ( torch.trace(torch.linalg.solve(Sig_th, Sig_emp)) - k_dim
                 - torch.logdet(Sig_emp) + torch.logdet(Sig_th) )
    print("KL(emp || theory):", KL.item())

    # --- NEW: Time-series error vs final-theoretical stats ---
    ts_res = compute_time_series_errors_against_final(
        trajectories=all_traj,
        L_gamma=L_gamma,
        alpha=alpha,
        sigma=sigma,
        T=T,
        out_dir=out_dir
    )
    print(f"Saved mean error curve to: {ts_res['mean_plot_path']}")
    print(f"Saved covariance error curves to: {ts_res['cov_plot_path']}")