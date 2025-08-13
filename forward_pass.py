import os
import torch
import numpy as np
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import networkx as nx

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
                              seed: int = None) -> torch.Tensor:
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
):
    """
    Exact Δt discretization per step:
      x_{k+1} = tilde_alpha x_k + B_delta z_k,
      tilde_alpha = expm(-c L_gamma Δt),
      alpha       = tilde_alpha @ tilde_alpha,
      Sigma_delta = sigma^2 * (I - alpha) @ L_gamma^{-1},
      B_delta B_delta^T = Sigma_delta.
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

    # Exact one-step matrices (matrix exponential)
    tilde_alpha = torch.matrix_exp(-c * L_gamma * dt)        # [N,N]
    alpha       = tilde_alpha @ tilde_alpha                  # [N,N]

    # Sigma_delta = sigma^2 * (I - alpha) @ L_gamma^{-1}
    # Use linear solve for stability
    Sigma_delta = sigma**2 * torch.linalg.solve(L_gamma, (I - alpha))

    # Cholesky for B_delta (lower)
    S = 0.5 * (Sigma_delta + Sigma_delta.T)  # symmetrize
    try:
        B_delta = torch.linalg.cholesky(S, upper=False)
    except RuntimeError:
        eps = 1e-6 * torch.trace(S) / N
        B_delta = torch.linalg.cholesky(S + eps * I, upper=False)

    # Iterate
    x = X_sub.clone()
    for _ in range(T):
        z = torch.randn(N, D, device=device, dtype=dtype)
        x = tilde_alpha @ x + B_delta @ z

    return x, (L, L_gamma, tilde_alpha, alpha, Sigma_delta)

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
    R       = 500
    seed    = 42

    T     = 600
    c     = 10.0
    gamma = 0.3
    sigma = 0.5

    # Graph & Laplacian
    adj  = build_er_graph(n=n_nodes, p_edge=0.5, seed=seed)
    Dm = torch.diag(adj.sum(dim=1))
    L  = Dm - adj

    # Generate R terminal samples with the EXACT integrator
    final_samples = []
    # Use x0 = 0 so theory (Σ_T) matches the MC exactly
    X0_zero = torch.zeros((n_nodes, d_feat), dtype=torch.float64)
    cache = None
    for i in range(R):
        x_T, cache = forward_diffusion_exact(
            X_sub=X0_zero, A_sub=adj, T=T, c=c, gamma=gamma, sigma=sigma
        )
        final_samples.append(x_T)
        if (i+1) % 50 == 0:
            print(f"Generated {i+1}/{R} samples")

    # Compare distributions (exact formulas)
    _, L_gamma, tilde_alpha, alpha, Sigma_delta = cache
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

    # Relative Frobenius error
    rel = torch.norm(results['empirical_cov'] - results['theoretical_cov'], p='fro') / \
          (torch.norm(results['theoretical_cov'], p='fro') + 1e-12)
    print("Relative Frobenius error:", rel.item())

    # Stability of exact one-step operator: ρ(tilde_alpha) < 1
    rho = np.max(np.abs(np.linalg.eigvals(tilde_alpha.detach().cpu().numpy())))
    print("Spectral radius ρ(tilde_alpha):", float(np.real(rho)))

    # KL divergence between Gaussians N(0, Σ_emp) and N(0, Σ_theory)
    Sig_emp = results['empirical_cov'].to(torch.float64).clone()
    Sig_th  = results['theoretical_cov'].to(torch.float64).clone()
    eps = 1e-8
    I = torch.eye(Sig_emp.size(0), dtype=Sig_emp.dtype, device=Sig_emp.device)
    Sig_emp = Sig_emp + eps * I
    Sig_th  = Sig_th  + eps * I
    k = Sig_emp.size(0)
    KL = 0.5 * ( torch.trace(torch.linalg.solve(Sig_th, Sig_emp)) - k
                 - torch.logdet(Sig_emp) + torch.logdet(Sig_th) )
    print("KL(emp || theory):", KL.item())
