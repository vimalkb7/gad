# exp_vesde.py
# =============

import random, os
import numpy as np
import math

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from models import DenoiserGNN
from sde import VESDE


# ---------------- Utilities ----------------
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


# ---------------- Training / Eval ----------------
def _sample_t_ve(sde, B, device, dtype):
    """
    Sample t by sampling log-sigma ~ Uniform[log sigma_min, log sigma_max],
    then map to t in [0,1] such that sigma(t) = sigma_min * (sigma_max/sigma_min)^t.
    """
    log_smin = math.log(float(sde.sigma_min))
    log_ratio = math.log(float(sde.sigma_max) / float(sde.sigma_min))
    u = torch.rand(B, device=device, dtype=dtype)
    log_sigma = log_smin + u * log_ratio
    t = (log_sigma - log_smin) / log_ratio
    return t.clamp(0.0, 1.0)


def _tv_energy(x: torch.Tensor, L: torch.Tensor) -> torch.Tensor:
    """
    Per-sample TV = x^T L x, where L is a (sym.) Laplacian matrix on device.
    x: [B, N], L: [N, N]
    returns: [B] TV per sample
    """
    return (x @ L * x).sum(dim=1)


def train_step(batch_x0, device, net, opt, sde, L_tv_ten, w_tv=5e-3, grad_clip=1.0):
    # dtype/device consistency
    batch_x0 = batch_x0.to(device=device, dtype=torch.float32)
    B = batch_x0.shape[0]

    # time sampling (log-uniform in sigma)
    t = _sample_t_ve(sde, B, device, batch_x0.dtype)  # [B]

    # forward marginal: X_t = X_0 + sigma(t) * eps
    mean_t, std_t = sde.marginal_prob(batch_x0, t)    # mean_t = X0, std_t shape [B]
    eps = torch.randn_like(batch_x0)
    x_t = mean_t + std_t[:, None] * eps

    # ----- VE-DSM (score prediction) -----
    # true score:  -(x_t - x0)/sigma(t)^2
    target = -(x_t - batch_x0) / (std_t[:, None] ** 2)
    pred   = net(x_t, t)  # must output score with same shape as x_t
    dsm_loss = ((std_t[:, None] ** 2) * (pred - target) ** 2).mean()

    # ----- TV regularizer on the *denoised proxy*: x0_hat = x_t + sigma(t)^2 * s_theta(x_t,t)
    x0_hat = x_t + (std_t[:, None] ** 2) * pred
    tv_per_sample = _tv_energy(x0_hat, L_tv_ten)     # [B]
    tv_loss = tv_per_sample.mean()

    loss = dsm_loss + w_tv * tv_loss

    opt.zero_grad(set_to_none=True)
    loss.backward()
    if grad_clip is not None:
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=grad_clip)
    opt.step()
    return float(loss.item()), float(dsm_loss.item()), float(tv_loss.item())


@torch.no_grad()
def eval_step(batch_x0, device, net, sde, L_tv_ten, w_tv=5e-3):
    batch_x0 = batch_x0.to(device=device, dtype=torch.float32)
    B = batch_x0.shape[0]

    t = _sample_t_ve(sde, B, device, batch_x0.dtype)
    mean_t, std_t = sde.marginal_prob(batch_x0, t)
    eps = torch.randn_like(batch_x0)
    x_t = mean_t + std_t[:, None] * eps

    target = -(x_t - batch_x0) / (std_t[:, None] ** 2)
    pred   = net(x_t, t)
    dsm_loss = ((std_t[:, None] ** 2) * (pred - target) ** 2).mean()

    x0_hat = x_t + (std_t[:, None] ** 2) * pred
    tv_loss = _tv_energy(x0_hat, L_tv_ten).mean()

    loss = dsm_loss + w_tv * tv_loss
    return float(loss.item()), float(dsm_loss.item()), float(tv_loss.item())


@torch.no_grad()
def evaluate(loader, device, net, sde, L_tv_ten, w_tv=5e-3, n_batches=10):
    losses, dsm_losses, tv_losses = [], [], []
    for i, (batch_x0,) in enumerate(loader):
        loss, dsm, tv = eval_step(batch_x0, device, net, sde, L_tv_ten, w_tv=w_tv)
        losses.append(loss); dsm_losses.append(dsm); tv_losses.append(tv)
        if i + 1 >= n_batches:
            break
    denom = max(len(losses), 1)
    return (sum(losses)/denom, sum(dsm_losses)/denom, sum(tv_losses)/denom)


def train(train_loader, val_loader, device, net, opt, sde, L_tv_ten,
          epochs=1000, eval_every=10, w_tv=5e-3):
    train_curve, eval_curve = [], []
    for epoch in range(epochs):
        # ---- Train epoch ----
        running, running_dsm, running_tv, n = 0.0, 0.0, 0.0, 0
        for (batch_x0,) in train_loader:
            loss, dsm, tv = train_step(batch_x0, device, net, opt, sde, L_tv_ten, w_tv=w_tv)
            running += loss; running_dsm += dsm; running_tv += tv; n += 1
        avg_train = running / n if n > 0 else float('nan')
        avg_dsm   = running_dsm / n if n > 0 else float('nan')
        avg_tv    = running_tv / n if n > 0 else float('nan')
        train_curve.append((epoch + 1, avg_train, avg_dsm, avg_tv))

        # ---- Eval ----
        if (epoch + 1) % eval_every == 0:
            ev_loss, ev_dsm, ev_tv = evaluate(val_loader, device, net, sde, L_tv_ten, w_tv=w_tv, n_batches=10)
            eval_curve.append((epoch + 1, ev_loss, ev_dsm, ev_tv))
            print(f"[Eval]  Ep {epoch + 1:5d} | train={avg_train:.6f} (dsm={avg_dsm:.6f}, tv={avg_tv:.6f}) "
                  f"| eval={ev_loss:.6f} (dsm={ev_dsm:.6f}, tv={ev_tv:.6f})")
        else:
            eval_curve.append((epoch + 1, None, None, None))
            if (epoch + 1) % max(eval_every // 2, 1) == 0:
                print(f"[Train] Ep {epoch + 1:5d} | train={avg_train:.6f} (dsm={avg_dsm:.6f}, tv={avg_tv:.6f})")

    return train_curve, eval_curve


# ---------------- Main ----------------
if __name__ == "__main__":
    seed_everything(8)

    # --- Data ---
    PATH_L_NORM = "/home/nfs/vkumarasamybal/Code3/Dataset_Processed/Exp3/L_sym.npy"
    PATH_L_COMB = "/home/nfs/vkumarasamybal/Code3/Dataset/L.npy"  # optional; if missing we'll fall back to L_norm
    PATH_X_TRAIN = "/home/nfs/vkumarasamybal/Code3/Dataset_Processed/Exp3/x_train.npy"
    PATH_X_TEST  = "/home/nfs/vkumarasamybal/Code3/Dataset_Processed/Exp3/x_test.npy"

    L_norm = np.load(PATH_L_NORM)
    L_norm = (L_norm + L_norm.T) * 0.5  # symmetry for safety

    # Prefer combinatorial Laplacian for TV if available
    if os.path.exists(PATH_L_COMB):
        L_tv_np = np.load(PATH_L_COMB)
        L_tv_np = (L_tv_np + L_tv_np.T) * 0.5
    else:
        L_tv_np = L_norm  # fallback

    x0 = np.load(PATH_X_TRAIN)
    x0_test = np.load(PATH_X_TEST)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # VE-SDE hyperparams (standard)
    sigma_min = 0.01
    sigma_max = 50.0
    T = 1.0

    # tensors
    X0 = torch.as_tensor(x0, dtype=torch.float32)
    X0_test = torch.as_tensor(x0_test, dtype=torch.float32)

    # per-node (feature-wise) normalization with train stats
    mu  = X0.mean(dim=0, keepdim=True)                 # [1, N]
    std = X0.std(dim=0, keepdim=True).clamp_min(1e-6)  # [1, N]
    X0      = (X0 - mu) / std
    X0_test = (X0_test - mu) / std

    # dataloaders
    B = X0.shape[0]
    batch_size = max(1, min(128, B))
    dataset, dataset_test = TensorDataset(X0), TensorDataset(X0_test)
    loader      = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    loader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=True, drop_last=False)

    # graph operators
    S = torch.as_tensor(L_norm, dtype=torch.float32, device=device)     # for GNN
    L_tv_ten = torch.as_tensor(L_tv_np, dtype=torch.float32, device=device)  # for TV loss

    # model
    N_nodes = X0.shape[1]
    gnn = DenoiserGNN(
        S=S,
        Ks=[5, 5, 5],
        t_dim=64,
        C=24,
        activation="silu",
        use_layernorm=True,
        use_residual=True
    ).to(device)

    # SDE object — use the data’s node count
    sde = VESDE(N=N_nodes, sigma_min=sigma_min, sigma_max=sigma_max, T=T)

    # optimizer
    opt_gnn = torch.optim.AdamW(gnn.parameters(), lr=1e-3, weight_decay=1e-4)

    # train (w/ TV)
    train_curve, eval_curve = train(
        loader, loader_test, device, gnn, opt_gnn, sde, L_tv_ten,
        epochs=5_000, eval_every=20, w_tv=0
    )

    # save weights + normalization stats for test-time sampling
    results_dir = "/tudelft.net/staff-bulk/ewi/insy/MMC/vimal/Results2/VESDE/Exp1"
    os.makedirs(results_dir, exist_ok=True)
    torch.save(gnn.state_dict(), f"{results_dir}/gnn_vesde_1.pth")
    torch.save({"mu": mu.cpu(), "std": std.cpu()}, f"{results_dir}/vesde_standardization_stats.pt")
