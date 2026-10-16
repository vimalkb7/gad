import os
import numpy as np

import torch

from src.data import load_molene, split_train_test, get_loaders
from src.models import DenoiserGNN
from src.sde import GASDE
from src.trainer import train, sample
from src.utils import seed_everything, evaluate_metrics
from src.metrics import spectral_centroid, mmd_distance


STEPS = [250, 500, 750, 1000]
N_REPS = 10

# All models share the heat schedule c(t), S and sigma of GAD and only differ in the drift:
#   GAD: -c(t) (L_norm + gamma I) x    VPD: -c(t) sigma^2 x (unit stationary variance)    VED: no drift
SDE = {"gamma": 0.2, "S": 25.0, "sigma": 0.5, "T": 1.0, "alpha": 4.0, "c_min": 0.1}

CONFIGS = {
    "ga": {"seed": 79, "lr": 1e-3},
    "vp": {"seed": 79, "lr": 1e-3},
    "ve": {"seed": 8,  "lr": 1e-3},
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def build_sde(name, L_norm):
    N = L_norm.shape[0]
    if name == "ga":
        L_drift = L_norm + SDE["gamma"] * np.eye(N)
    elif name == "vp":
        L_drift = (SDE["sigma"] ** 2) * np.eye(N)
    elif name == "ve":
        L_drift = np.zeros((N, N))
    else:
        raise ValueError(f"Unknown SDE '{name}'")

    L_drift_ten = torch.as_tensor(L_drift, dtype=torch.float32, device=device)
    return GASDE(
        L_gamma=L_drift_ten, S=SDE["S"], sigma=SDE["sigma"], T=SDE["T"],
        alpha=SDE["alpha"], c_min=SDE["c_min"], device=device
    )

def build_gnn(L_norm):
    S = torch.as_tensor(L_norm, dtype=torch.float32, device=device)
    return DenoiserGNN(
        S=S,
        Ks=[5, 5, 5],
        t_dim=64,
        C=24,
        activation="silu",
        use_layernorm=True,
        use_residual=True
    ).to(device)

def load_data():
    X, _, L, L_norm = load_molene()
    X0, X0_test = split_train_test(X, n_train=670, n_test=74)

    # Per-node standardization with train statistics
    mu = X0.mean(dim=0, keepdim=True)
    std = X0.std(dim=0, keepdim=True).clamp_min(1e-6)
    return X0, X0_test, L, L_norm, mu, std

def zscore(X):
    # Per-sample z-score; temperatures are dominated by the constant component
    m = X.mean(dim=1, keepdim=True)
    s = X.std(dim=1, keepdim=True).clamp_min(1e-6)
    return ((X - m) / s).to(torch.float64)

def evaluate_metrics_molene(X_gen, X_real, L, L_norm):
    dist_tv, _, dist_dc = evaluate_metrics(X_gen, X_real, L)

    sc_gen = spectral_centroid(zscore(X_gen), L_norm)
    sc_real = spectral_centroid(zscore(X_real), L_norm)
    dist_sc = mmd_distance(sc_real, sc_gen, kernel="rbf", num_sigma=5, sigma_scale=1.0)

    return dist_tv, dist_sc.item(), dist_dc


def run_molene_training(epochs=5_000, models=tuple(CONFIGS)):
    os.makedirs("checkpoints/molene", exist_ok=True)
    for name in models:
        seed_everything(CONFIGS[name]["seed"])

        X0, X0_test, _, L_norm, mu, std = load_data()
        loader, loader_test = get_loaders((X0 - mu) / std, (X0_test - mu) / std)

        sde = build_sde(name, L_norm)
        gnn = build_gnn(L_norm)
        opt_gnn = torch.optim.AdamW(gnn.parameters(), lr=CONFIGS[name]["lr"], weight_decay=1e-4)

        train(loader, loader_test, device, gnn, opt_gnn, sde, epochs=epochs, eval_every=epochs)
        torch.save(gnn.state_dict(), f"checkpoints/molene/{name}.pth")

def run_molene_experiments():
    os.makedirs("results", exist_ok=True)
    X0, X0_test, L, L_norm, mu, std = load_data()
    X_real = X0_test.to(device)
    mu, std = mu.to(device), std.to(device)
    B, N = X0_test.shape

    for name in CONFIGS:
        sde = build_sde(name, L_norm)
        gnn = build_gnn(L_norm)
        gnn.load_state_dict(torch.load(f"checkpoints/molene/{name}.pth", map_location=device))
        gnn.eval()

        res = np.zeros((len(STEPS), N_REPS, 3))  # MMD of TV, SC and DC
        for i, steps in enumerate(STEPS):
            for j in range(N_REPS):
                seed_everything(j)
                X_gen = sample(sde, gnn, shape=(B, N), steps=steps, device=device) * std + mu
                res[i, j] = evaluate_metrics_molene(X_gen, X_real, L, L_norm)

        np.save(f"results/molene_{name}.npy", res)
    np.save("results/molene_steps.npy", np.array(STEPS))
