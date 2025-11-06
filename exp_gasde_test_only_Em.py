# exp_gasde_test.py
# =================
# Simple Euler–Maruyama reverse-SDE sampler ONLY (no Heun/trapezoid, no Langevin, no clipping).

import numpy as np
import torch
import math

from models import DenoiserGNN
from sde import GASDE
from signal_metrics import (
    total_variation,
    spectral_centroid,   # unchanged
    degree_correlation,  # unchanged
    mmd_distance,
)

# ---------- Paths ----------
PATH_L_NORM   = "/home/nfs/vkumarasamybal/Code3/Dataset_Processed/Exp3/L_sym.npy"
PATH_L        = "/home/nfs/vkumarasamybal/Code3/Dataset/L.npy"
PATH_X_TRAIN  = "/home/nfs/vkumarasamybal/Code3/Dataset_Processed/Exp3/x_train.npy"
PATH_X_TEST   = "/home/nfs/vkumarasamybal/Code3/Dataset_Processed/Exp3/x_test.npy"

PATH_CKPT     = "/tudelft.net/staff-bulk/ewi/insy/MMC/vimal/Results2/GASDE/Exp1/gnn_gasde_1.pth"
PATH_STATS    = "/tudelft.net/staff-bulk/ewi/insy/MMC/vimal/Results2/GASDE/Exp1/exp_gasde_standardization_stats.pt"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------- Data ----------
L_norm = np.load(PATH_L_NORM)
L_norm = (L_norm + L_norm.T) * 0.5  # enforce symmetry
L      = np.load(PATH_L)
L      = (L + L.T) * 0.5            # symmetrize combinatorial Laplacian

x0       = np.load(PATH_X_TRAIN)
x0_test  = np.load(PATH_X_TEST)
B_eval, N = x0_test.shape

# ---------- Match training schedule ----------
gamma  = 0.8
S_heat = 7.0
sigma  = 1.0
T      = 1.0
alpha  = 4.0
c_min  = 0.1

L_gamma = L_norm + gamma * np.eye(L_norm.shape[0], dtype=L_norm.dtype)
L_gamma_ten = torch.as_tensor(L_gamma, dtype=torch.float32, device=device)

sde = GASDE(
    L_gamma=L_gamma_ten, S=S_heat, sigma=sigma, T=T,
    alpha=alpha, c_min=c_min, device=device
)

# ---------- Model ----------
S_op = torch.as_tensor(L_norm, dtype=torch.float32, device=device)
net = DenoiserGNN(
    S=S_op,
    Ks=[5, 5, 5],
    t_dim=64,
    C=24,
    activation="silu",
    use_layernorm=True,
    use_residual=True
).to(device)
state = torch.load(PATH_CKPT, map_location=device)
net.load_state_dict(state)
net.eval()

# ---------- Train-time normalization ----------
stats = torch.load(PATH_STATS, map_location=device)
mu  = stats["mu"].to(device=device, dtype=torch.float32)   # [1, N]
std = stats["std"].to(device=device, dtype=torch.float32)  # [1, N]

# ---------- Score adapter (SC/DC functions unchanged) ----------
@torch.no_grad()
def score_fn_adapter(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """
    score = -Sigma(t)^{-1} [ x - H(t) * x0_hat ], computed in eigenbasis.
    """
    if t.dim() == 0:
        t = t.repeat(x.shape[0])

    x = x.to(device=device, dtype=sde.dtype)
    t = t.to(device=device, dtype=sde.dtype)

    x0_hat = net(x, t)

    U   = sde.U.to(x)
    lam = sde.lam.to(x).clamp_min(1e-8)
    s   = sde._s_of_t(t).to(x)  # [B]
    sigma2 = (sde.sigma ** 2)

    # Project to eigenbasis
    z_t   = x @ U
    z_hat = x0_hat @ U

    decay    = torch.exp(- s[:, None] * lam[None, :])           # [B, N]
    mu_eig   = decay * z_hat                                    # [B, N]
    diff_eig = z_t - mu_eig

    var_eig = sigma2 * (1.0 - torch.exp(-2.0 * s[:, None] * lam[None, :])) / lam[None, :]
    inv_var = 1.0 / var_eig.clamp_min(torch.finfo(var_eig.dtype).eps)

    score_eig = - diff_eig * inv_var                            # [B, N]
    score = score_eig @ U.T                                      # [B, N]
    return score

# ---------- Pure Euler–Maruyama (reverse SDE) on a uniform-in-s grid ----------
@torch.no_grad()
def sample_reverse_sde_em_in_s(
    sde: GASDE, score_fn, shape, steps: int = 3000
) -> torch.Tensor:
    """
    Euler–Maruyama sampler of the reverse SDE on a grid uniform in heat s.
    x_{k+1} = x_k + (b - g^2 * score) * dt + g * sqrt(|dt|) * N(0, I)
    """
    dev = sde.device
    B, N = shape
    x = sde.prior_sampling(shape).to(dtype=sde.dtype)

    S_tot, T = sde.S, sde.T

    def inv_s_to_t(s_target: torch.Tensor) -> torch.Tensor:
        # invert s(t) with two Newton steps
        t = (s_target / max(S_tot, 1e-8)) * T
        for _ in range(2):
            u = (t / T).clamp(0, 1)
            s_t  = c_min * t + (S_tot - c_min * T) * (u ** (alpha + 1.0))
            dsdt = c_min + (S_tot - c_min * T) * (alpha + 1.0) * (u ** alpha) / T
            t = (t - (s_t - s_target) / dsdt).clamp(0, T)
        return t

    # descending grid in s (S -> 0), mapped to t
    s_grid = torch.linspace(S_tot, 0.0, steps + 1, device=dev, dtype=sde.dtype)
    t_grid = inv_s_to_t(s_grid)

    rng = torch.Generator(device=dev)
    for k in range(steps):
        t_k, t_k1 = t_grid[k], t_grid[k + 1]
        dt = (t_k1 - t_k)  # negative
        tB = torch.full((B,), float(t_k.item()), device=dev, dtype=sde.dtype)

        drift_fwd_k, diffusion_k = sde.sde(x, tB)          # diffusion_k: [B]
        score_k = score_fn(x, tB)                           # [B, N]
        gk = diffusion_k.view(B, 1)                         # [B, 1]
        drift_rev_k = drift_fwd_k - (gk ** 2) * score_k     # [B, N]

        xi = torch.randn((B, N), generator=rng, device=dev, dtype=x.dtype)
        x = x + drift_rev_k * dt + gk * (dt.abs().sqrt()) * xi

    return x

# ---------- Helpers for SC stability (preprocess only inputs) ----------
@torch.no_grad()
def _per_sample_zscore(X: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    m = X.mean(dim=1, keepdim=True)
    s = X.std(dim=1, keepdim=True).clamp_min(eps)
    return (X - m) / s

@torch.no_grad()
def normalized_tv(X: torch.Tensor, L_mat) -> torch.Tensor:
    """
    Scale-invariant TV: (x^T L x) / (x^T x), per sample.
    """
    L_t = torch.as_tensor(L_mat, dtype=X.dtype, device=X.device)
    Xc = X - X.mean(dim=1, keepdim=True)
    num = (Xc @ L_t * Xc).sum(dim=1)
    den = (Xc ** 2).sum(dim=1).clamp_min(1e-12)
    return num / den

# ---------- One evaluation ----------
@torch.no_grad()
def evaluate_once(B_eval: int, N: int, steps: int = 4000):
    # 1) Sample in normalized space (PURE EM)
    X_gen_norm = sample_reverse_sde_em_in_s(
        sde, score_fn_adapter, shape=(B_eval, N), steps=steps
    )

    # 2) De-normalize to original scale
    X_gen = X_gen_norm.to(torch.float32) * std + mu

    # 3) Real signals (original scale)
    X_real = torch.as_tensor(x0_test, dtype=torch.float32, device=device)

    # 4) TV
    tv_gen  = total_variation(X_gen,  L)
    tv_real = total_variation(X_real, L)

    # 5) Spectral centroid — unchanged function, stabilized inputs
    X_sc_gen  = _per_sample_zscore(X_gen).to(torch.float64)
    X_sc_real = _per_sample_zscore(X_real).to(torch.float64)
    sc_gen  = spectral_centroid(X_sc_gen,  L_norm)   # unchanged function
    sc_real = spectral_centroid(X_sc_real, L_norm)

    # 6) Degree correlation — unchanged function
    Xg_center = X_gen  - X_gen.mean(dim=1, keepdim=True)
    Xr_center = X_real - X_real.mean(dim=1, keepdim=True)
    d_vec = np.diag(L)
    dc_gen  = degree_correlation(Xg_center, d_vec, "mean")  # unchanged function
    dc_real = degree_correlation(Xr_center, d_vec, "mean")

    # 7) MMD distances (distributional match)
    dist_tv = mmd_distance(tv_real, tv_gen, kernel="rbf", num_sigma=5, sigma_scale=1.0)
    dist_sc = mmd_distance(sc_real, sc_gen, kernel="rbf", num_sigma=5, sigma_scale=1.0)
    dist_dc = mmd_distance(dc_real, dc_gen, kernel="rbf", num_sigma=5, sigma_scale=1.0)

    return dist_tv.item(), dist_sc.item(), dist_dc.item(), tv_real, tv_gen, X_real, X_gen

# ---------- Run multiple trials ----------
tvs, scs, dcs = [], [], []
n_trials = 10
last_tv_real = last_tv_gen = None
last_X_real = last_X_gen = None

for _ in range(n_trials):
    dist_tv, dist_sc, dist_dc, tv_real, tv_gen, X_real, X_gen = evaluate_once(
        B_eval=B_eval, N=N, steps=5000
    )
    tvs.append(dist_tv)
    scs.append(dist_sc)      # <-- fixed typo (was `sc`)
    dcs.append(dist_dc)
    last_tv_real, last_tv_gen = tv_real, tv_gen
    last_X_real, last_X_gen = X_real, X_gen

print(f"TV - Mean: {np.mean(tvs):.6f} - Std: {np.std(tvs):.6f}")
print(f"SC - Mean: {np.mean(scs):.6f} - Std: {np.std(scs):.6f}")
print(f"DC - Mean: {np.mean(dcs):.6f} - Std: {np.std(dcs):.6f}")

# Diagnostics from the last run
def tv_summary(name, tv_vals):
    m, s = float(tv_vals.mean()), float(tv_vals.std())
    print(f"{name} TV: mean={m:.6f}, std={s:.6f}")
    return m, s

m_r, s_r = tv_summary("Real", last_tv_real)
m_g, s_g = tv_summary("Gen ", last_tv_gen)
print(f"TV ratio Gen/Real = {m_g/m_r:.3f}  -> >1 rougher, <1 smoother")

nTV_real = normalized_tv(last_X_real, L)
nTV_gen  = normalized_tv(last_X_gen,  L)
print(f"nTV Real: mean={float(nTV_real.mean()):.6f}, std={float(nTV_real.std()):.6f}")
print(f"nTV Gen : mean={float(nTV_gen.mean()):.6f}, std={float(nTV_gen.std()):.6f}")
print(f"nTV ratio Gen/Real = {float(nTV_gen.mean()/nTV_real.mean()):.3f}")

@torch.no_grad()
def cumulative_spectral_energy(X, L_norm):
    Ls = torch.as_tensor(L_norm, dtype=X.dtype, device=X.device)
    lam, U = torch.linalg.eigh(Ls)
    Z = X @ U
    ps = (Z ** 2).mean(0)
    cum = torch.cumsum(ps, 0) / ps.clamp_min(1e-12).sum()
    return lam.cpu().numpy(), cum.cpu().numpy()

lam, cum_real = cumulative_spectral_energy(last_X_real, L_norm)
_,   cum_gen  = cumulative_spectral_energy(last_X_gen,  L_norm)
print("cum_real:", cum_real)
print("cum_gen :", cum_gen)
