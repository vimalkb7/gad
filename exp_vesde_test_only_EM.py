# exp_vesde_test_only_EM_stable.py
# ================================
# Reverse-SDE sampler using ONLY Euler–Maruyama (no Heun, no Langevin, no clipping)
# Stabilized with:
#   - sigma_cap (clamp the schedule)
#   - per-interval substeps (micro-EM)
#   - dt_max_abs guard
# Robust metric implementations to avoid NaNs.

import os
import math
import numpy as np
import torch

from models import DenoiserGNN
from sde import VESDE
from signal_metrics import mmd_distance  # we reuse only MMD from your utils

# ---------- Paths ----------
PATH_L_NORM   = "/home/nfs/vkumarasamybal/Code3/Dataset_Processed/Exp3/L_sym.npy"
PATH_L        = "/home/nfs/vkumarasamybal/Code3/Dataset/L.npy"
PATH_X_TRAIN  = "/home/nfs/vkumarasamybal/Code3/Dataset_Processed/Exp3/x_train.npy"
PATH_X_TEST   = "/home/nfs/vkumarasamybal/Code3/Dataset_Processed/Exp3/x_test.npy"

PATH_CKPT     = "/tudelft.net/staff-bulk/ewi/insy/MMC/vimal/Results2/VESDE/Exp1/gnn_vesde_1.pth"
PATH_STATS    = "/tudelft.net/staff-bulk/ewi/insy/MMC/vimal/Results2/VESDE/Exp1/vesde_standardization_stats.pt"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------- Data ----------
L_norm = np.load(PATH_L_NORM)
L_norm = (L_norm + L_norm.T) * 0.5   # enforce symmetry
L      = np.load(PATH_L)
L      = (L + L.T) * 0.5             # enforce symmetry

x0      = np.load(PATH_X_TRAIN)
x0_test = np.load(PATH_X_TEST)

B_eval, N_nodes = x0_test.shape

# ---------- VE-SDE schedule (match training hyperparams) ----------
sigma_min = 0.01
sigma_max = 50.0
T = 1.0

sde = VESDE(N=N_nodes, sigma_min=sigma_min, sigma_max=sigma_max, T=T)

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

# ---------- Train-time normalization (same stats as training) ----------
if os.path.exists(PATH_STATS):
    stats = torch.load(PATH_STATS, map_location=device)
    mu  = stats["mu"].to(device).to(torch.float32)   # [1,N]
    std = stats["std"].to(device).to(torch.float32)  # [1,N]
else:
    X0_train = torch.as_tensor(x0, dtype=torch.float32, device=device)
    mu  = X0_train.mean(dim=0, keepdim=True)
    std = X0_train.std(dim=0, keepdim=True).clamp_min(1e-6)

# ---------- Helpers ----------
def _expand_like(v, x):
    while v.dim() < x.dim():
        v = v.unsqueeze(-1)
    return v

@torch.no_grad()
def score_fn_adapter(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """VE-DSM: net predicts the score directly."""
    if t.dim() == 0:
        t = t.repeat(x.shape[0])
    return net(x, t)

# sigma <-> t mapping for VE schedule: sigma(t) = sigma_min * (sigma_max/sigma_min)^t
def _sigma_to_t(s, smin, smax):
    log_smin  = math.log(float(smin))
    log_ratio = math.log(float(smax) / float(smin))
    return (torch.log(s) - log_smin) / log_ratio

def _karras_sigmas(smin, smax, steps, rho=7.0, device=None, dtype=torch.float32):
    i = torch.linspace(0, 1, steps + 1, device=device, dtype=dtype)
    # monotonically decreasing sigma sequence
    return (smax**(1/rho) + i * ((smin**(1/rho)) - smax**(1/rho)))**rho

# ---------- EM-only sampler (stabilized) ----------
@torch.no_grad()
def sample_vesde_em_only(
    sde: VESDE, score_fn, shape, steps: int = 750,
    sigma_cap: float = 10.0,      # cap the sigma schedule
    substeps: int = 8,            # micro-EM steps inside each interval
    dt_max_abs: float = 1e-3      # guard on |dt| per micro-step
) -> torch.Tensor:
    """
    Reverse-SDE Euler–Maruyama ONLY, stabilized via:
      - clamped sigma schedule (sigma_cap)
      - per-interval substeps (micro-EM)
      - |dt| cap per micro-step
    No Heun, no Langevin, no clipping of x/scores.
    """
    B, N = shape
    dev = device
    dtype = torch.float32

    # base Karras schedule, then clamp to sigma_cap
    sigmas = _karras_sigmas(sde.sigma_min, sde.sigma_max, steps, rho=7.0, device=dev, dtype=dtype)
    sigmas = torch.clamp(sigmas, max=float(sigma_cap))

    # initialize from the (capped) start sigma
    s0 = sigmas[0]
    x = s0 * torch.randn(B, N, device=dev, dtype=dtype)

    for k in range(steps):
        # interval endpoints in sigma, and their mapped times
        s_cur, s_next = sigmas[k], sigmas[k+1]
        t_cur  = _sigma_to_t(s_cur,  sde.sigma_min, sde.sigma_max)
        t_next = _sigma_to_t(s_next, sde.sigma_min, sde.sigma_max)

        # integrate from t_cur -> t_next in 'substeps' micro-steps
        for j in range(substeps):
            # linear micro-time (descending)
            tau0 = t_cur + (t_next - t_cur) * (j     / substeps)
            tau1 = t_cur + (t_next - t_cur) * ((j+1) / substeps)
            dt   = (tau1 - tau0).item()  # negative number

            # |dt| guard
            if abs(dt) > dt_max_abs:
                dt = -abs(dt_max_abs) if dt < 0 else abs(dt_max_abs)

            # reverse SDE at current state/time
            drift_rev, diff = sde.reverse_sde(x, torch.tensor(tau0, device=dev, dtype=dtype), score_fn, probability_flow=False)

            # EM update
            noise = torch.randn_like(x)
            x = x + drift_rev * dt + _expand_like(diff, x) * noise * math.sqrt(abs(dt))

            # cheap finite guard
            if not torch.isfinite(x).all():
                x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    return x

# ---------- SAFE METRICS ----------
@torch.no_grad()
def safe_total_variation(X: torch.Tensor, L_np: np.ndarray) -> torch.Tensor:
    """TV(x) = x^T L x per sample, robust to non-finites."""
    L_t = torch.as_tensor(L_np, dtype=X.dtype, device=X.device)
    X   = torch.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    L_t = torch.nan_to_num(L_t, nan=0.0, posinf=0.0, neginf=0.0)
    tv = (X @ L_t) * X
    tv = torch.nan_to_num(tv, nan=0.0, posinf=0.0, neginf=0.0)
    return tv.sum(dim=1)

@torch.no_grad()
def safe_spectral_centroid(X: torch.Tensor, L_norm_np: np.ndarray) -> torch.Tensor:
    """Spectral centroid on per-sample z-scored signals, robust."""
    eps = 1e-12
    m = X.mean(dim=1, keepdim=True)
    s = X.std(dim=1, keepdim=True).clamp_min(eps)
    Xz = (X - m) / s

    Ls = torch.as_tensor((L_norm_np + L_norm_np.T) * 0.5, dtype=X.dtype, device=X.device)
    lam, U = torch.linalg.eigh(Ls)
    lam = lam.clamp_min(0.0)  # kill tiny negatives

    Z = Xz @ U
    power_ps = (Z**2)
    denom_ps = power_ps.sum(dim=1, keepdim=True).clamp_min(eps)
    weights_ps = power_ps / denom_ps
    sc = (weights_ps * lam.view(1, -1)).sum(dim=1)
    sc = torch.nan_to_num(sc, nan=0.0, posinf=0.0, neginf=0.0)
    return sc

@torch.no_grad()
def safe_degree_correlation(X: torch.Tensor, d_vec_np: np.ndarray) -> torch.Tensor:
    """Mean-centered degree correlation, robust to NaNs."""
    Xc = X - X.mean(dim=1, keepdim=True)
    Xc = torch.nan_to_num(Xc, nan=0.0, posinf=0.0, neginf=0.0)
    d = torch.as_tensor(d_vec_np, dtype=X.dtype, device=X.device)
    num = (Xc * d.view(1, -1)).sum(dim=1)
    den = (Xc.norm(dim=1) * d.norm()).clamp_min(1e-12)
    dc = num / den
    dc = torch.nan_to_num(dc, nan=0.0, posinf=0.0, neginf=0.0)
    return dc

# ---------- Evaluation ----------
tvs, scs, dcs = [], [], []
num_trials = 10

X_real_orig = torch.as_tensor(x0_test, dtype=torch.float32, device=device)
X_real_orig = torch.nan_to_num(X_real_orig, nan=0.0, posinf=0.0, neginf=0.0)

for run in range(num_trials):
    X_gen_norm = sample_vesde_em_only(
        sde, score_fn_adapter, shape=(B_eval, N_nodes),
        steps=5000, sigma_cap=10.0, substeps=8, dt_max_abs=1e-3
    )

    print(f"[Run {run+1}] finite:", torch.isfinite(X_gen_norm).all().item(),
          "| min/max:", float(torch.nan_to_num(X_gen_norm).min()),
          float(torch.nan_to_num(X_gen_norm).max()))

    X_gen = X_gen_norm * std + mu
    X_gen = torch.nan_to_num(X_gen, nan=0.0, posinf=0.0, neginf=0.0)

    tv_gen  = safe_total_variation(X_gen, L)
    tv_real = safe_total_variation(X_real_orig, L)

    sc_gen  = safe_spectral_centroid(X_gen,  L_norm)
    sc_real = safe_spectral_centroid(X_real_orig, L_norm)

    d_vec = np.diag(L)
    dc_gen  = safe_degree_correlation(X_gen,  d_vec)
    dc_real = safe_degree_correlation(X_real_orig, d_vec)

    # MMDs
    def _finite(v): return torch.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
    dist_tv = mmd_distance(_finite(tv_real), _finite(tv_gen), kernel="rbf", num_sigma=5, sigma_scale=1.0)
    dist_sc = mmd_distance(_finite(sc_real), _finite(sc_gen), kernel="rbf", num_sigma=5, sigma_scale=1.0)
    dist_dc = mmd_distance(_finite(dc_real), _finite(dc_gen), kernel="rbf", num_sigma=5, sigma_scale=1.0)

    tvs.append(float(dist_tv))
    scs.append(float(dist_sc))
    dcs.append(float(dist_dc))

print(f"TV - Mean: {np.mean(tvs):.6f} - Std: {np.std(tvs):.6f}")
print(f"SC - Mean: {np.mean(scs):.6f} - Std: {np.std(scs):.6f}")
print(f"DC - Mean: {np.mean(dcs):.6f} - Std: {np.std(dcs):.6f}")
