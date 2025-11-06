# exp_gasde.py
# =================

import os, random, math
import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from models import DenoiserGNN, DenoiserMLP
from sde import GASDE
from utils import plot_heat_schedule


# --------- Helpers ---------
def inv_s_to_t(s_target, S_tot, T, alpha, c_min, iters=2):
    """
    Invert s(t) = c_min*t + (S_tot - c_min*T) * (t/T)^(alpha+1)
    with a few Newton steps. s_target is a tensor on the correct device/dtype.
    """
    # Good initializer: proportional to target fraction of total S
    t = (s_target / max(float(S_tot), 1e-8)) * T
    for _ in range(iters):
        u = (t / T).clamp(0, 1)
        s_t  = c_min * t + (S_tot - c_min * T) * (u ** (alpha + 1.0))
        dsdt = c_min + (S_tot - c_min * T) * (alpha + 1.0) * (u ** alpha) / T
        t = (t - (s_t - s_target) / dsdt).clamp(0, T)
    return t


def _degree_vector_from_L(L_np: np.ndarray) -> np.ndarray:
    # diagonal of combinatorial Laplacian is the degree
    return np.diag(L_np)

@torch.no_grad()
def _prep_degree_tensor(L_np: np.ndarray, device, dtype):
    d = _degree_vector_from_L(L_np)
    return torch.as_tensor(d, dtype=dtype, device=device)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


# --------- Training / Eval steps ---------
def train_step(batch_x0, device, net, opt, sde, L_ten_global, L_norm, d_ten):
    # Ensure dtype/device consistency
    batch_x0 = batch_x0.to(device=device, dtype=sde.dtype)
    B = batch_x0.shape[0]

    # Sample uniformly in s (slightly biased to later times); invert to t
    u = torch.rand(B, device=device, dtype=sde.dtype).pow(0.65)
    s = u * sde.S
    # Use the SAME schedule hyperparams as the SDE
    t = inv_s_to_t(s, sde.S, sde.T, alpha=sde.alpha, c_min=sde.c_min)

    # Exact marginal
    x_t   = sde.marginal_sampling(batch_x0, t)
    x0_hat = net(x_t, t)

    # SNR-aware weight: beta(t) = 2 * sigma^2 * c(t)
    c_t  = sde._c_of_t(t)  # already on device/dtype
    beta = 2.0 * (sde.sigma ** 2) * c_t
    w_t  = (1.0 / (1.0 + beta)).unsqueeze(1)  # [B,1]

    # Residual
    res = x0_hat - batch_x0

    # TV regularizer on prediction — use the SAME Laplacian as your TV metric (combinatorial L)
    tv_pred = (x0_hat @ L_ten_global * x0_hat).sum(dim=1).mean()

    # Base (SNR-weighted) MSE in node space
    mse = (w_t * (res ** 2)).mean()

    # Combine (you can re-enable spec/degree terms if desired)
    # loss = mse + 1e-3 * spec_loss + 5e-4 * tv_pred + dc_loss
    loss = mse + 5e-3 * tv_pred

    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
    opt.step()

    return loss.item()


@torch.no_grad()
def eval_step(batch_x0, device, net, sde, L_ten_global, L_norm, d_ten):
    batch_x0 = batch_x0.to(device=device, dtype=sde.dtype)
    B = batch_x0.shape[0]

    # Sample t the SAME way as train
    u = torch.rand(B, device=device, dtype=sde.dtype).pow(0.65)
    s = u * sde.S
    t = inv_s_to_t(s, sde.S, sde.T, alpha=sde.alpha, c_min=sde.c_min)

    x_t   = sde.marginal_sampling(batch_x0, t)
    x0_hat = net(x_t, t)

    c_t  = sde._c_of_t(t)
    beta = 2.0 * (sde.sigma ** 2) * c_t
    w_t  = (1.0 / (1.0 + beta)).unsqueeze(1)

    res = x0_hat - batch_x0
    tv_pred = (x0_hat @ L_ten_global * x0_hat).sum(dim=1).mean()
    mse = (w_t * (res ** 2)).mean()
    # loss = mse + 1e-3 * spec_loss + 5e-4 * tv_pred + dc_loss
    loss = mse + 5e-2 * tv_pred
    return loss.item()


@torch.no_grad()
def evaluate(loader, device, net, sde, L_ten_global, L_norm, d_ten, n_batches=10):
    losses = []
    for i, (batch_x0,) in enumerate(loader):
        losses.append(eval_step(batch_x0, device, net, sde, L_ten_global, L_norm, d_ten))
        if i + 1 >= n_batches:
            break
    return sum(losses) / max(len(losses), 1)


def train(train_loader, val_loader, device, net, opt, sde, L_ten_global, L_norm, d_ten,
          epochs=1000, eval_every=10):
    train_losses, eval_losses = [], []
    for epoch in range(epochs):
        # Train
        total, n = 0.0, 0
        for (batch_x0,) in train_loader:
            total += train_step(batch_x0, device, net, opt, sde, L_ten_global, L_norm, d_ten)
            n += 1
        avg_train = total / n if n > 0 else float('nan')
        train_losses.append(avg_train)

        # Eval
        if (epoch + 1) % eval_every == 0:
            avg_eval = evaluate(val_loader, device, net, sde, L_ten_global, L_norm, d_ten, n_batches=10)
            eval_losses.append((epoch + 1, avg_eval))
            print(f"[Eval] Epoch {epoch + 1:5d} | train_loss = {avg_train:.6f} | eval_loss = {avg_eval:.6f}")
        else:
            eval_losses.append((epoch + 1, None))
            if (epoch + 1) % max(eval_every // 2, 1) == 0:
                print(f"[Train] Epoch {epoch + 1:5d} | train_loss = {avg_train:.6f}")

    return train_losses, eval_losses


# --------- Main ---------
if __name__ == "__main__":
    seed_everything(79)

    # Data & Laplacians
    L_norm = np.load("/home/nfs/vkumarasamybal/Code3/Dataset_Processed/Exp3/L_sym.npy")
    x0     = np.load("/home/nfs/vkumarasamybal/Code3/Dataset_Processed/Exp3/x_train.npy")
    x0_test= np.load("/home/nfs/vkumarasamybal/Code3/Dataset_Processed/Exp3/x_test.npy")
    L_np   = np.load("/home/nfs/vkumarasamybal/Code3/Dataset/L.npy")

    # Be safe: enforce symmetry
    L_norm = (L_norm + L_norm.T) * 0.5
    L_np   = (L_np   + L_np.T)   * 0.5

    # SDE hyperparams (tune as needed)
    gamma = 0.8
    S     = 7.0
    sigma = 1.0
    T     = 1.0
    alpha = 4.0        # keep consistent across SDE & inv_s_to_t
    c_min = 0.1

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Tensors (dtype float32 end-to-end)
    d_ten = torch.as_tensor(np.diag(L_np), dtype=torch.float32, device=device)
    L_ten_global = torch.as_tensor(L_np, dtype=torch.float32, device=device)

    L_gamma = L_norm + gamma * np.eye(L_norm.shape[0])
    L_gamma_ten = torch.as_tensor(L_gamma, dtype=torch.float32, device=device)

    # SDE with explicit alpha, c_min for consistency
    sde = GASDE(L_gamma=L_gamma_ten, S=S, sigma=sigma, T=T, alpha=alpha, c_min=c_min, device=device)

    # Plot the heat schedule once
    RESULTS_DIR = "/tudelft.net/staff-bulk/ewi/insy/MMC/vimal/Results2/GASDE/Exp1"
    os.makedirs(RESULTS_DIR, exist_ok=True)
    plot_heat_schedule(
        sde,
        n_eigs='all',
        show_modes=True,
        results_directory=RESULTS_DIR
    )

    # Normalize data (per-node, using train stats)
    X0      = torch.as_tensor(x0, dtype=torch.float32)
    X0_test = torch.as_tensor(x0_test, dtype=torch.float32)
    mu  = X0.mean(dim=0, keepdim=True)
    std = X0.std(dim=0, keepdim=True).clamp_min(1e-6)
    X0 = (X0 - mu) / std
    X0_test = (X0_test - mu) / std

    # Dataloaders
    B = X0.shape[0]
    batch_size = max(1, min(128, B))
    dataset, dataset_test = TensorDataset(X0), TensorDataset(X0_test)
    loader      = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    loader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=True, drop_last=False)

    # Models
    N_nodes = X0.shape[1]
    mlp = DenoiserMLP(n_nodes=N_nodes, t_dim=64, hidden=256).to(device)

    S_ten = torch.as_tensor(L_norm, dtype=torch.float32, device=device)
    gnn = DenoiserGNN(
        S=S_ten,
        Ks=[5, 5, 5],
        t_dim=64,
        C=24,
        activation="silu",
        use_layernorm=True,
        use_residual=True
    ).to(device)

    # Optims
    opt_mlp = torch.optim.AdamW(mlp.parameters(), lr=1e-4, weight_decay=1e-4)
    opt_gnn = torch.optim.AdamW(gnn.parameters(), lr=1e-4, weight_decay=1e-4)

    # Train GNN
    losses_gnn, evals_gnn = train(
        loader, loader_test, device, gnn, opt_gnn, sde, L_ten_global, L_norm, d_ten,
        epochs=5_000, eval_every=20
    )

    # Save
    torch.save(gnn.state_dict(), f"{RESULTS_DIR}/gnn_gasde_1.pth")
    torch.save({"mu": mu.cpu(), "std": std.cpu()},
               f"{RESULTS_DIR}/exp_gasde_standardization_stats.pt")
