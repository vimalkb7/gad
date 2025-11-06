# exp_vpsde.py
# ==============

import os, random
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F

from models import DenoiserGNN
from sde import VPSDE


# ---------------- Utilities ----------------
def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def _tv_energy(x: torch.Tensor, L: torch.Tensor) -> torch.Tensor:
    """
    Per-sample graph TV = x^T L x, with L a (sym.) Laplacian.
    x: [B, N], L: [N, N]  -> returns [B]
    """
    return (x @ L * x).sum(dim=1)


# ---------------- Train / Eval steps ----------------
def train_step(batch_x0, device, net, opt, sde, L_tv_ten, w_tv=5e-3, grad_clip=1.0):
    """
    GASDE-style training for VPSDE:
      - sample t ~ Uniform(0,1) biased to later times via pow(0.65)
      - draw x_t from exact VP marginal
      - predict x0_hat = net(x_t, t)
      - SNR-weighted MSE + TV( x0_hat )
    """
    batch_x0 = batch_x0.to(device=device, dtype=torch.float32)
    B = batch_x0.shape[0]

    # Bias toward later times (higher noise), similar spirit to GASDE's s^0.65
    t = torch.rand(B, device=device, dtype=batch_x0.dtype).pow(0.65)

    # Exact VP marginal
    mean_t, std_t = sde.marginal_prob(batch_x0, t)         # mean_t ~ ᾱ(t) x0, std_t ~ sqrt(1-ᾱ(t))
    eps = torch.randn_like(batch_x0)
    x_t = mean_t + std_t[:, None] * eps

    # Predict clean signal
    x0_hat = net(x_t, t)

    # ---- SNR-aware weight (stable): downweight very noisy times ----
    # A simple, bounded surrogate: w(t) = 1 / (1 + std(t)^2)
    w_t = 1.0 / (1.0 + (std_t[:, None] ** 2))

    # Reconstruction loss
    res = x0_hat - batch_x0
    mse = (w_t * (res ** 2)).mean()

    # TV regularizer on the predicted clean signal
    tv_pred = _tv_energy(x0_hat, L_tv_ten).mean()

    loss = mse + w_tv * tv_pred

    opt.zero_grad(set_to_none=True)
    loss.backward()
    if grad_clip is not None:
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=grad_clip)
    opt.step()

    return float(loss.item())


@torch.no_grad()
def eval_step(batch_x0, device, net, sde, L_tv_ten, w_tv=5e-3):
    batch_x0 = batch_x0.to(device=device, dtype=torch.float32)
    B = batch_x0.shape[0]
    t = torch.rand(B, device=device, dtype=batch_x0.dtype).pow(0.65)

    mean_t, std_t = sde.marginal_prob(batch_x0, t)
    eps = torch.randn_like(batch_x0)
    x_t = mean_t + std_t[:, None] * eps

    x0_hat = net(x_t, t)
    w_t = 1.0 / (1.0 + (std_t[:, None] ** 2))

    res = x0_hat - batch_x0
    mse = (w_t * (res ** 2)).mean()
    tv_pred = _tv_energy(x0_hat, L_tv_ten).mean()

    loss = mse + w_tv * tv_pred
    return float(loss.item())


@torch.no_grad()
def evaluate(loader, device, net, sde, L_tv_ten, w_tv=5e-3, n_batches=10):
    losses = []
    for i, (batch_x0,) in enumerate(loader):
        losses.append(eval_step(batch_x0, device, net, sde, L_tv_ten, w_tv=w_tv))
        if i + 1 >= n_batches:
            break
    return sum(losses) / max(len(losses), 1)


def train(train_loader, val_loader, device, net, opt, sde, L_tv_ten,
          epochs=1000, eval_every=10, w_tv=5e-3):
    train_losses, eval_losses = [], []
    for epoch in range(epochs):
        # ---- Train epoch ----
        total, n = 0.0, 0
        for (batch_x0,) in train_loader:
            total += train_step(batch_x0, device, net, opt, sde, L_tv_ten, w_tv=w_tv)
            n += 1
        avg_train = total / n if n > 0 else float('nan')
        train_losses.append(avg_train)

        # ---- Eval ----
        if (epoch + 1) % eval_every == 0:
            avg_eval = evaluate(val_loader, device, net, sde, L_tv_ten, w_tv=w_tv, n_batches=10)
            eval_losses.append((epoch + 1, avg_eval))
            print(f"[Eval] Epoch {epoch + 1:5d} | train_loss = {avg_train:.6f} | eval_loss = {avg_eval:.6f}")
        else:
            eval_losses.append((epoch + 1, None))
            if (epoch + 1) % max(eval_every // 2, 1) == 0:
                print(f"[Train] Epoch {epoch + 1:5d} | train_loss = {avg_train:.6f}")

    return train_losses, eval_losses


# ---------------- Main ----------------
if __name__ == "__main__":
    seed_everything(79)

    # ---- Data & Laplacians ----
    PATH_L_NORM = "/home/nfs/vkumarasamybal/Code3/Dataset_Processed/Exp3/L_sym.npy"
    PATH_L_COMB = "/home/nfs/vkumarasamybal/Code3/Dataset/L.npy"
    PATH_X_TRAIN = "/home/nfs/vkumarasamybal/Code3/Dataset_Processed/Exp3/x_train.npy"
    PATH_X_TEST  = "/home/nfs/vkumarasamybal/Code3/Dataset_Processed/Exp3/x_test.npy"

    L_norm = np.load(PATH_L_NORM)
    L_norm = (L_norm + L_norm.T) * 0.5  # symmetry for safety

    if os.path.exists(PATH_L_COMB):
        L_tv_np = np.load(PATH_L_COMB)
        L_tv_np = (L_tv_np + L_tv_np.T) * 0.5
    else:
        L_tv_np = L_norm  # fallback

    x0 = np.load(PATH_X_TRAIN)
    x0_test = np.load(PATH_X_TEST)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ---- VPSDE hyperparams ----
    s_vp = 0.008
    T = 1.0

    # ---- Tensors ----
    X0 = torch.as_tensor(x0, dtype=torch.float32)
    X0_test = torch.as_tensor(x0_test, dtype=torch.float32)

    # per-node normalization (fit on train, used for both train/test)
    mu  = X0.mean(dim=0, keepdim=True)
    std = X0.std(dim=0, keepdim=True).clamp_min(1e-6)
    X0      = (X0 - mu) / std
    X0_test = (X0_test - mu) / std

    # ---- Dataloaders ----
    B = X0.shape[0]
    batch_size = max(1, min(128, B))
    dataset, dataset_test = TensorDataset(X0), TensorDataset(X0_test)
    loader      = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    loader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=True, drop_last=False)

    # ---- Graph operators ----
    S = torch.as_tensor(L_norm,   dtype=torch.float32, device=device)      # for GNN filters
    L_tv_ten = torch.as_tensor(L_tv_np, dtype=torch.float32, device=device) # for TV loss

    # ---- Model ----
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

    # ---- SDE (use data's node count) ----
    sde = VPSDE(N=N_nodes, s=s_vp, T=T)

    # ---- Optimizer ----
    opt_gnn = torch.optim.AdamW(gnn.parameters(), lr=1e-3, weight_decay=1e-4)

    # ---- Train ----
    train_losses, eval_losses = train(
        loader, loader_test, device, gnn, opt_gnn, sde, L_tv_ten,
        epochs=5_000, eval_every=20, w_tv=5e-3
    )

    # ---- Save weights + normalization stats ----
    RESULTS_DIR = "/tudelft.net/staff-bulk/ewi/insy/MMC/vimal/Results2/VPSDE/Exp1"
    os.makedirs(RESULTS_DIR, exist_ok=True)

    torch.save(gnn.state_dict(), f"{RESULTS_DIR}/gnn_vpsde_1.pth")
    torch.save(
        {"mu": mu.cpu(), "std": std.cpu()},
        f"{RESULTS_DIR}/exp_vpsde_standardization_stats.pt",
    )
