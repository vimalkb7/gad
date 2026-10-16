import os
import numpy as np

import torch

from src.data import load_sbm, get_loaders
from src.models import DenoiserGNN
from src.sde import GASDE, VPSDE, VESDE
from src.trainer import train, sample
from src.utils import seed_everything, evaluate_metrics


STEPS = [100, 200, 300, 400, 500]
N_REPS = 10

CONFIGS = {
    "ga": {"seed": 79, "sde": {"gamma": 0.1, "S": 25.0, "sigma": 0.5, "T": 1.0}},
    "vp": {"seed": 79, "sde": {"s": 0.008, "T": 1.0}},
    "ve": {"seed": 8,  "sde": {"sigma_min": 0.01, "sigma_max": 50.0, "T": 1.0}},
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def build_sde(name, L_norm):
    cfg = CONFIGS[name]["sde"]
    if name == "ga":
        L_gamma = L_norm + cfg["gamma"] * np.eye(L_norm.shape[0])
        L_gamma_ten = torch.as_tensor(L_gamma, dtype=torch.float32, device=device)
        return GASDE(L_gamma=L_gamma_ten, S=cfg["S"], sigma=cfg["sigma"], T=cfg["T"], device=device)
    if name == "vp":
        return VPSDE(N=1000, s=cfg["s"], T=cfg["T"], device=device)
    if name == "ve":
        return VESDE(N=1000, sigma_min=cfg["sigma_min"], sigma_max=cfg["sigma_max"], T=cfg["T"], device=device)
    raise ValueError(f"Unknown SDE '{name}'")

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


def run_sbm_training(epochs=15_000):
    os.makedirs("checkpoints/sbm", exist_ok=True)
    for name in CONFIGS:
        seed_everything(CONFIGS[name]["seed"])

        X0, X0_test, _, L_norm = load_sbm()
        loader, loader_test = get_loaders(X0, X0_test)

        sde = build_sde(name, L_norm)
        gnn = build_gnn(L_norm)
        opt_gnn = torch.optim.AdamW(gnn.parameters(), lr=1e-3, weight_decay=1e-4)

        train(loader, loader_test, device, gnn, opt_gnn, sde, epochs=epochs, eval_every=20)
        torch.save(gnn.state_dict(), f"checkpoints/sbm/{name}.pth")

def run_sbm_experiments():
    os.makedirs("results", exist_ok=True)
    X0, X0_test, L, L_norm = load_sbm()
    X_real = X0_test.to(device)
    B, N = X0_test.shape

    for name in CONFIGS:
        sde = build_sde(name, L_norm)
        gnn = build_gnn(L_norm)
        gnn.load_state_dict(torch.load(f"checkpoints/sbm/{name}.pth", map_location=device))
        gnn.eval()

        res = np.zeros((len(STEPS), N_REPS, 3))  # MMD of TV, SC and DC
        for i, steps in enumerate(STEPS):
            for j in range(N_REPS):
                seed_everything(j)
                X_gen = sample(sde, gnn, shape=(B, N), steps=steps, device=device)
                res[i, j] = evaluate_metrics(X_gen, X_real, L)

        np.save(f"results/sbm_{name}.npy", res)
    np.save("results/sbm_steps.npy", np.array(STEPS))
