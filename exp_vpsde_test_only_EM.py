# exp_vpsde_test_only_EM.py
# ==================
# Reverse-SDE sampler using ONLY Euler–Maruyama (no Heun, no Langevin, no clipping)

import os, random, math
import numpy as np
import torch

from models import DenoiserGNN
from sde import VPSDE
from signal_metrics import (
    total_variation,
    spectral_centroid,
    degree_correlation,
    mmd_distance,
)

# ------------------- knobs -------------------
SEED   = 79
STEPS  = 100
N_RUNS = 10
# ---------------------------------------------

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def seed_everything(seed=SEED):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False
    os.environ["PYTHONHASHSEED"] = str(seed)

seed_everything()

# ------------------- helpers -------------------
def _expand_like(v, x):
    while v.dim() < x.dim():
        v = v.unsqueeze(-1)
    return v

def zscore_1d(v: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    m = v.mean()
    s = v.std().clamp_min(eps)
    return (v - m) / s

def clip_sc_to_bounds(sc: torch.Tensor, lmin: float = 0.0, lmax: float = 2.0) -> torch.Tensor:
    return sc.clamp(min=lmin, max=lmax)

def t_grid_from_alpha_bar(sde: VPSDE, steps: int, power: float = 0.5, device=None):
    """
    Create a descending time grid T -> 0 using alpha_bar as the 's-analog'.
    power=0.5 => uniform in sqrt(alpha_bar), concentrating steps in the low-noise tail.
    """
    device = device or sde.device or "cpu"
    s, T = sde.s, sde.T
    u = torch.linspace(0, 1, steps+1, device=device)   # 0..1
    eps = sde.eps

    # target alpha_bar goes from ~1 down to eps
    a = eps + (1.0 - eps) * (1.0 - u**power)
    # invert cosine schedule: a = cos^2(theta), theta = 0.5*pi*(t/T + s)/(1+s)
    theta = torch.arccos(torch.sqrt(a)).clamp(0, math.pi/2)
    t_over_T = (2.0 * (1.0 + s) / math.pi) * theta - s
    t = (t_over_T * T).clamp(0.0, T)

    # descending (T -> 0)
    return torch.flip(t, dims=[0])

def _final_x0_from_net(x, net, device):
    t0 = torch.zeros(x.shape[0], device=device, dtype=x.dtype)
    return net(x, t0)

# ------------------- score adapter -------------------
@torch.no_grad()
def score_fn_adapter(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """
    VPSDE: if net predicts x0 in standardized space, the score is:
      ∇_x log p_t(x) = -(x - sqrt(alpha_bar(t))*x0_hat) / (1 - alpha_bar(t))
    """
    if t.dim() == 0:
        t = t.repeat(x.shape[0])
    x0_hat = net(x, t)                       # predicts x0 (standardized space)
    a_bar  = sde._alpha_bar(t)               # (B,)
    num = x - torch.sqrt(a_bar)[:, None] * x0_hat
    den = (1.0 - a_bar)[:, None].clamp_min(1e-8)
    return - num / den

# ------------------- EM-only sampler -------------------
@torch.no_grad()
def em_sample_vpsde(sde: VPSDE, score_fn, net, shape, steps: int = STEPS, device=device):
    """
    Reverse-SDE Euler–Maruyama ONLY:
      x_{k+1} = x_k + drift_rev(x_k, t_k) * dt + g(t_k) * sqrt(|dt|) * ξ,  ξ~N(0,I)
    - time grid: uniform in sqrt(alpha_bar)
    - one noise draw per step
    - no Heun/Langevin/clipping
    """
    x = sde.prior_sampling(shape, device=device)  # standardized space
    t_grid = t_grid_from_alpha_bar(sde, steps=steps, device=device)

    for k in range(steps):
        t  = t_grid[k]
        t1 = t_grid[k+1]
        dt = t1 - t  # negative step

        # Forward SDE pieces
        drift_fwd_k, diffusion_k = sde.sde(x, t)      # diffusion_k: [B]
        score_k = score_fn(x, t)                      # [B,N]

        # Reverse drift
        gk = diffusion_k.view(x.shape[0], 1)          # [B,1]
        drift_rev_k = drift_fwd_k - (gk ** 2) * score_k

        # EM update
        noise = torch.randn_like(x)
        x = x + drift_rev_k * dt + gk * (abs(float(dt)) ** 0.5) * noise

    # If the net predicts x0, return denoised x0 at t=0
    return _final_x0_from_net(x, net, device)

# ------------------- load stats/model/data -------------------
# Train-time standardization stats (mu, std) to de-normalize:
stats = torch.load(
    "/tudelft.net/staff-bulk/ewi/insy/MMC/vimal/Results2/VPSDE/Exp1/exp_vpsde_standardization_stats.pt",
    map_location="cpu"
)
mu, std = stats["mu"], stats["std"]

# Data & Laplacians
L_norm = np.load("/home/nfs/vkumarasamybal/Code3/Dataset_Processed/Exp3/L_sym.npy")  # for SC
L      = np.load("/home/nfs/vkumarasamybal/Code3/Dataset/L.npy")                      # for TV/DC
x0     = np.load("/home/nfs/vkumarasamybal/Code3/Dataset_Processed/Exp3/x_train.npy")
x0_test= np.load("/home/nfs/vkumarasamybal/Code3/Dataset_Processed/Exp3/x_test.npy")

B_eval, N = x0_test.shape

# Model & SDE
sde = VPSDE(N=N, s=0.008, T=1.0)
S   = torch.as_tensor(L_norm, dtype=torch.float32, device=device)
net = DenoiserGNN(
    S=S, Ks=[5, 5, 5], t_dim=64, C=24,
    activation="silu", use_layernorm=True, use_residual=True
).to(device)
net.load_state_dict(torch.load(
    "/tudelft.net/staff-bulk/ewi/insy/MMC/vimal/Results2/VPSDE/Exp1/gnn_vpsde_1.pth",
    map_location=device
))
net.eval()

mu  = mu.to(device)
std = std.to(device)

# ------------------- evaluation loop (MMD: TV, SC, DC) -------------------
tvs, scs, dcs = [], [], []

for _ in range(N_RUNS):
    # EM-only samples (standardized -> original)
    X_gen_std = em_sample_vpsde(sde, score_fn_adapter, net, shape=(B_eval, N), steps=STEPS, device=device)
    X_gen = X_gen_std * std + mu

    # Real signals (original scale)
    X_real = torch.as_tensor(x0_test, dtype=torch.float32, device=device)

    # TV with combinatorial Laplacian
    tv_gen = total_variation(X_gen, L)
    tv_real = total_variation(X_real, L)

    # Spectral centroid with normalized Laplacian (stabilize for MMD)
    sc_gen  = spectral_centroid(X_gen,  L_norm)
    sc_real = spectral_centroid(X_real, L_norm)
    sc_real_b = zscore_1d(clip_sc_to_bounds(sc_real))
    sc_gen_b  = zscore_1d(clip_sc_to_bounds(sc_gen))

    # Degree correlation
    d = np.diag(L)
    dc_gen  = degree_correlation(X_gen,  d, "mean")
    dc_real = degree_correlation(X_real, d, "mean")

    # MMDs
    dist_tv = mmd_distance(tv_real,  tv_gen,  kernel="rbf", num_sigma=5, sigma_scale=1.0)
    dist_sc = mmd_distance(sc_real_b, sc_gen_b, kernel="rbf", num_sigma=5, sigma_scale=1.0)
    dist_dc = mmd_distance(dc_real,  dc_gen,  kernel="rbf", num_sigma=5, sigma_scale=1.0)

    tvs.append(max(float(dist_tv), 0.0))
    scs.append(max(float(dist_sc), 0.0))
    dcs.append(max(float(dist_dc), 0.0))

print(f"TV - Mean: {np.mean(tvs):.6f} - Std: {np.std(tvs):.6f}")
print(f"SC - Mean: {np.mean(scs):.6f} - Std: {np.std(scs):.6f}")
print(f"DC - Mean: {np.mean(dcs):.6f} - Std: {np.std(dcs):.6f}")
