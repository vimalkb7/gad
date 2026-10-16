import os
import random
import numpy as np

import torch

from src.metrics import (
    total_variation,
    spectral_centroid,
    degree_correlation,
    mmd_distance,
)


def seed_everything(seed):
    random.seed(seed)                          # Python RNG
    np.random.seed(seed)                       # NumPy RNG
    torch.manual_seed(seed)                    # PyTorch RNG (CPU)
    torch.cuda.manual_seed_all(seed)           # PyTorch RNG (all CUDA devices)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.environ["PYTHONHASHSEED"] = str(seed)

def evaluate_metrics(X_gen, X_real, L):
    # MMD between the distributions of TV, SC and DC of real and generated signals
    tv_gen = total_variation(X_gen, L)
    tv_real = total_variation(X_real, L)

    sc_gen = spectral_centroid(X_gen, L)
    sc_real = spectral_centroid(X_real, L)

    d = np.diag(L)
    dc_gen = degree_correlation(X_gen, d, "mean")
    dc_real = degree_correlation(X_real, d, "mean")

    dist_tv = mmd_distance(tv_real, tv_gen, kernel="rbf", num_sigma=5, sigma_scale=1.0)
    dist_sc = mmd_distance(sc_real, sc_gen, kernel="rbf", num_sigma=5, sigma_scale=1.0)
    dist_dc = mmd_distance(dc_real, dc_gen, kernel="rbf", num_sigma=5, sigma_scale=1.0)

    return dist_tv.item(), dist_sc.item(), dist_dc.item()

def ammd(res):
    # res: (n_steps, n_reps, 3) with the MMD of TV, SC and DC
    mu = res.mean(axis=1)                                # (n_steps, 3)
    sd = res.std(axis=1)                                 # (n_steps, 3)
    mean = mu.mean(axis=1)
    std = np.sqrt((sd**2).sum(axis=1)) / 3.0
    return mean, std
