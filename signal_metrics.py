# signal_metrics.py
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Literal, Optional, Tuple, Union

import numpy as np
import torch

Tensor = torch.Tensor
ArrayLike = Union[Tensor, np.ndarray]

# =============================================================================
# Signal Metrics & Distribution Distances for Graph Signals
# =============================================================================
# 1) Convert graph signals X={x_1,...,x_N} to scalar metrics Φ(x):
#    - Total Variation (TV):             x^T L x
#    - Spectral Centroid (SC):           sum_i λ_i |u_i|^2 / sum_i |u_i|^2, u = U^T x
#    - Degree Correlation (DegCorr):     corr(d, x) averaged across feature dims
# 2) Compare the resulting scalar distributions F_X and F_Y using:
#    - Maximum Mean Discrepancy (MMD) with RBF or linear kernel
#    - 1D Wasserstein distance (quantile approximation)
#    - KL divergence between fitted univariate Gaussians
#    - Kolmogorov–Smirnov statistic (two-sample)
# Supports torch.Tensors (preferred) and NumPy arrays. X can be [B,N] or [B,N,D].
# -----------------------------------------------------------------------------


# ------------------------------ Utilities ------------------------------------
def _to_tensor(x: ArrayLike, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None) -> Tensor:
    if isinstance(x, np.ndarray):
        t = torch.from_numpy(x)
    elif isinstance(x, torch.Tensor):
        t = x
    else:
        t = torch.tensor(x)
    if dtype is not None:
        t = t.to(dtype)
    if device is not None:
        t = t.to(device)
    return t


def _ensure_2d_features(X: Tensor) -> Tensor:
    """Ensure X has shape [B, N, D]. If X is [B, N], add D=1."""
    if X.dim() == 2:
        return X.unsqueeze(-1)
    return X


def _degree_from_adjacency(A: Tensor) -> Tensor:
    """Compute degree vector d from adjacency A (assumed [N,N])."""
    return A.sum(dim=1)


def _eig_L(L: Tensor) -> Tuple[Tensor, Tensor]:
    """
    Symmetric eigendecomposition of Laplacian L.
    Returns (eigvals [N], eigvecs [N,N]) with ascending eigenvalues.
    """
    evals, evecs = torch.linalg.eigh(L)
    return evals, evecs


def _project_to_spectrum(X: Tensor, U: Tensor) -> Tensor:
    """
    Project batched signals X [B, N, D] to graph spectral domain using U [N,N].
    Returns coefficients U^T X --> [B, N, D]
    """
    return torch.einsum("ni,bid->bnd", U, X)


# ------------------------------ Metrics Φ ------------------------------------
def total_variation(X: ArrayLike, L: ArrayLike) -> Tensor:
    """
    TV(x) = x^T L x, averaged over feature dims if D>1.
    Args:
        X: [B,N] or [B,N,D]
        L: [N,N]
    Returns:
        tv: [B] tensor of scalars
    """
    X = _to_tensor(X)
    L = _to_tensor(L, device=X.device, dtype=X.dtype)
    X = _ensure_2d_features(X)  # [B,N,D]

    y = torch.einsum("ij,bjd->bid", L, X)   # [B,N,D]
    per_feat = (X * y).sum(dim=1)           # [B,D]
    tv = per_feat.mean(dim=1)               # [B]
    return tv


def spectral_centroid(
    X: ArrayLike,
    L: ArrayLike,
    eig: Optional[Tuple[Tensor, Tensor]] = None,
) -> Tensor:
    """
    SC(x) = sum_i λ_i |u_i|^2 / sum_i |u_i|^2, with u = U^T x.
    For D>1, average the per-feature centroid.
    Args:
        X:   [B,N] or [B,N,D]
        L:   [N,N]
        eig: optional (evals [N], evecs [N,N]) to reuse cached spectrum
    Returns:
        sc: [B] tensor
    """
    X = _to_tensor(X)
    L = _to_tensor(L, device=X.device, dtype=X.dtype)
    X = _ensure_2d_features(X)

    if eig is None:
        lam, U = _eig_L(L)
    else:
        lam, U = eig
        lam = lam.to(device=X.device, dtype=X.dtype)
        U = U.to(device=X.device, dtype=X.dtype)

    coeffs = _project_to_spectrum(X, U)     # [B,N,D], u = U^T x
    power = (coeffs**2)                      # [B,N,D] (real signals)

    num = (power * lam.view(1, -1, 1)).sum(dim=1)  # [B,D]
    den = power.sum(dim=1).clamp_min(1e-12)        # [B,D]
    sc_per_feat = num / den                         # [B,D]
    sc = sc_per_feat.mean(dim=1)                    # [B]
    return sc


def degree_correlation(
    X: ArrayLike,
    degrees: ArrayLike,
    reduce: Literal["mean", "median", "abs-mean", "abs-median"] = "mean",
) -> Tensor:
    """
    Pearson correlation between degree vector d and signal x (per feature),
    then reduce across features to one scalar.
    Args:
        X:        [B,N] or [B,N,D]
        degrees:  [N]
        reduce:   reduction across features
    Returns:
        corr: [B]
    """
    X = _to_tensor(X)
    d = _to_tensor(degrees, device=X.device, dtype=X.dtype).view(1, -1, 1)  # [1,N,1]
    X = _ensure_2d_features(X)  # [B,N,D]

    # Center both
    d_center = d - d.mean(dim=1, keepdim=True)     # [1,N,1]
    X_center = X - X.mean(dim=1, keepdim=True)     # [B,N,D]

    # Cov(d, x) per feature
    cov = (d_center * X_center).mean(dim=1)        # [B,D]
    std_d = (d_center.pow(2).mean(dim=1).sqrt()).clamp_min(1e-12)  # [1,1,1]
    std_x = (X_center.pow(2).mean(dim=1).sqrt()).clamp_min(1e-12)  # [B,1,D]

    corr = cov / (std_d + 0.0) / (std_x + 0.0)     # [B,D]

    if reduce == "mean":
        out = corr.mean(dim=1)
    elif reduce == "median":
        out = corr.median(dim=1).values
    elif reduce == "abs-mean":
        out = corr.abs().mean(dim=1)
    elif reduce == "abs-median":
        out = corr.abs().median(dim=1).values
    else:
        raise ValueError(f"Unknown reduce={reduce}")
    return out


# ------------------------ Feature Extraction API -----------------------------
MetricName = Literal["tv", "sc", "degcorr"]

@dataclass
class FeatureConfig:
    metrics: Tuple[MetricName, ...] = ("tv", "sc", "degcorr")
    degcorr_reduce: Literal["mean", "median", "abs-mean", "abs-median"] = "mean"

@dataclass
class DistanceConfig:
    mmd_kernel: Literal["rbf", "linear"] = "rbf"
    mmd_num_sigma: int = 5                 # number of RBF bandwidths (log-spaced)
    mmd_sigma_scale: float = 1.0           # scale factor for bandwidths
    wasserstein_quantiles: int = 200       # number of quantile points
    kl_min_var: float = 1e-6               # floor variance for Gaussian KL


def extract_feature_set(
    X: ArrayLike,
    L: ArrayLike,
    degrees: Optional[ArrayLike] = None,
    eig: Optional[Tuple[Tensor, Tensor]] = None,
    cfg: FeatureConfig = FeatureConfig(),
) -> Dict[str, Tensor]:
    """
    Compute selected metrics for each sample in X to produce scalar features.
    Args:
        X:       [B,N] or [B,N,D]
        L:       [N,N] Laplacian
        degrees: [N] degree vector. If None, fallback assumes unnormalized L = D - A.
        eig:     optional (evals, evecs) to avoid repeated eigendecompositions
        cfg:     which metrics to compute
    Returns:
        dict mapping metric -> [B] tensor of scalars
    """
    X = _to_tensor(X)
    L = _to_tensor(L, device=X.device, dtype=X.dtype)

    out: Dict[str, Tensor] = {}
    if "tv" in cfg.metrics:
        out["tv"] = total_variation(X, L)
    if "sc" in cfg.metrics:
        out["sc"] = spectral_centroid(X, L, eig=eig)
    if "degcorr" in cfg.metrics:
        if degrees is None:
            d = torch.diag(L)  # works only if L is unnormalized (L = D - A)
        else:
            d = _to_tensor(degrees, device=X.device, dtype=X.dtype)
        out["degcorr"] = degree_correlation(X, d, reduce=cfg.degcorr_reduce)
    return out


# ------------------------- Distance Computations -----------------------------
def _pairwise_sq_dists(x: Tensor, y: Tensor) -> Tensor:
    x2 = (x**2).unsqueeze(1)          # [n,1]
    y2 = (y**2).unsqueeze(0)          # [1,m]
    xy = x.unsqueeze(1) @ y.unsqueeze(0)  # [n,m]
    return x2 - 2 * xy + y2


def _median_heuristic_sigma(vals: Tensor) -> float:
    """Median pairwise distance heuristic on concatenated samples."""
    v = vals.view(-1, 1)
    pd = torch.cdist(v, v, p=2)
    med = torch.median(pd[pd > 0])
    if not torch.isfinite(med):
        med = torch.tensor(1.0, device=vals.device, dtype=vals.dtype)
    return float(med.item())


def mmd_distance(
    fx: ArrayLike,
    fy: ArrayLike,
    kernel: Literal["rbf", "linear"] = "rbf",
    num_sigma: int = 5,
    sigma_scale: float = 1.0,
) -> Tensor:
    """
    Maximum Mean Discrepancy between 1D samples fx, fy.
    RBF uses a mixture of Gaussian kernels with log-spaced bandwidths around median heuristic.
    """
    x = _to_tensor(fx).view(-1)
    y = _to_tensor(fy, device=x.device, dtype=x.dtype).view(-1)
    n, m = x.numel(), y.numel()

    if kernel == "linear":
        return (x.mean() - y.mean()).pow(2)

    with torch.no_grad():
        base_sigma = _median_heuristic_sigma(torch.cat([x, y], dim=0))
        base_sigma = max(float(base_sigma), 1e-6)
        sigmas = torch.logspace(-1, 1, steps=num_sigma, device=x.device, dtype=x.dtype) * (sigma_scale * base_sigma)

    XX = _pairwise_sq_dists(x, x)
    YY = _pairwise_sq_dists(y, y)
    XY = _pairwise_sq_dists(x, y)

    k_xx = 0.0
    k_yy = 0.0
    k_xy = 0.0
    for s in sigmas:
        gamma = 1.0 / (2.0 * s * s)
        k_xx = k_xx + torch.exp(-gamma * XX)
        k_yy = k_yy + torch.exp(-gamma * YY)
        k_xy = k_xy + torch.exp(-gamma * XY)

    mmd = (k_xx.sum() - k_xx.diag().sum()) / (n * (n - 1) + 1e-12) \
        + (k_yy.sum() - k_yy.diag().sum()) / (m * (m - 1) + 1e-12) \
        - 2.0 * k_xy.mean()
    return mmd


def wasserstein_1d(
    fx: ArrayLike,
    fy: ArrayLike,
    num_quantiles: int = 200,
) -> Tensor:
    """
    1-Wasserstein distance between 1D empirical distributions via quantile functions.
    Approximate W1 = integral_0^1 |Qx(p) - Qy(p)| dp using uniform grid of p.
    """
    x = torch.sort(_to_tensor(fx).view(-1))[0]
    y = torch.sort(_to_tensor(fy, device=x.device, dtype=x.dtype).view(-1))[0]

    px = torch.linspace(0.0, 1.0, steps=x.numel(), device=x.device)
    py = torch.linspace(0.0, 1.0, steps=y.numel(), device=y.device)
    p = torch.linspace(0.0, 1.0, steps=num_quantiles, device=x.device)

    def interp(qp, p_src, v_src):
        idx = torch.clamp((p_src.numel() - 1) * qp, 0, p_src.numel() - 1 - 1e-9)
        i0 = idx.floor().long()
        i1 = torch.clamp(i0 + 1, max=p_src.numel() - 1)
        t = (idx - i0.to(idx.dtype)).clamp(0, 1)
        return (1 - t) * v_src[i0] + t * v_src[i1]

    Qx = interp(p, px, x)
    Qy = interp(p, py, y)

    w1 = torch.mean(torch.abs(Qx - Qy))
    return w1


def gaussian_kl_1d(
    fx: ArrayLike,
    fy: ArrayLike,
    min_var: float = 1e-6,
) -> Tensor:
    """
    KL( N(mu_x, sig_x^2) || N(mu_y, sig_y^2) )
    """
    x = _to_tensor(fx).view(-1)
    y = _to_tensor(fy, device=x.device, dtype=x.dtype).view(-1)

    mu_x = x.mean()
    mu_y = y.mean()
    var_x = x.var(unbiased=False).clamp_min(min_var)
    var_y = y.var(unbiased=False).clamp_min(min_var)

    kl = 0.5 * ((var_x / var_y) + (mu_y - mu_x).pow(2) / var_y - 1.0 + torch.log(var_y / var_x))
    return kl


def ks_statistic(
    fx: ArrayLike,
    fy: ArrayLike,
    num_points: int = 1024,
) -> Tensor:
    """
    Two-sample Kolmogorov–Smirnov statistic in 1D (sup |F_x - F_y|).
    Approximated on a common grid spanning both supports.
    """
    x = _to_tensor(fx).view(-1)
    y = _to_tensor(fy, device=x.device, dtype=x.dtype).view(-1)

    xmin = torch.min(x.min(), y.min())
    xmax = torch.max(x.max(), y.max())
    grid = torch.linspace(xmin, xmax, steps=num_points, device=x.device)

    def ecdf(vals: Tensor, t: Tensor) -> Tensor:
        return (vals.view(1, -1) <= t.view(-1, 1)).float().mean(dim=1)

    Fx = ecdf(x, grid)
    Fy = ecdf(y, grid)
    return torch.max(torch.abs(Fx - Fy))


# ------------------------ High-level Evaluator --------------------------------
@dataclass
class SignalMetricsEvaluator:
    feature_cfg: FeatureConfig = field(default_factory=FeatureConfig)
    distance_cfg: DistanceConfig = field(default_factory=DistanceConfig)

    def features(
        self,
        X: ArrayLike,
        L: ArrayLike,
        degrees: Optional[ArrayLike] = None,
        eig: Optional[Tuple[Tensor, Tensor]] = None,
    ) -> Dict[str, Tensor]:
        return extract_feature_set(X, L, degrees=degrees, eig=eig, cfg=self.feature_cfg)

    def _dist_all(self, fx: Tensor, fy: Tensor) -> Dict[str, Tensor]:
        dc = self.distance_cfg
        return {
            "mmd": mmd_distance(fx, fy, kernel=dc.mmd_kernel, num_sigma=dc.mmd_num_sigma, sigma_scale=dc.mmd_sigma_scale),
            "wasserstein": wasserstein_1d(fx, fy, num_quantiles=dc.wasserstein_quantiles),
            "gaussian_kl": gaussian_kl_1d(fx, fy, min_var=dc.kl_min_var),
            "ks": ks_statistic(fx, fy),
        }

    @torch.no_grad()
    def compare(
        self,
        FX: Dict[str, Tensor],
        FY: Dict[str, Tensor],
    ) -> Dict[str, Dict[str, float]]:
        """
        Compare feature distributions for each metric in FX, FY.
        Returns dict: metric -> { "mmd":..., "wasserstein":..., "gaussian_kl":..., "ks":... }
        """
        out: Dict[str, Dict[str, float]] = {}
        for k in FX.keys():
            fx = FX[k].view(-1)
            fy = FY[k].view(-1).to(device=fx.device, dtype=fx.dtype)
            d = self._dist_all(fx, fy)
            out[k] = {name: float(val.item()) for name, val in d.items()}
        return out


# -------------------------- Convenience Hooks --------------------------------
@torch.no_grad()
def evaluate_signal_sets(
    X: ArrayLike,
    Y: ArrayLike,
    L: ArrayLike,
    degrees: Optional[ArrayLike] = None,
    eig: Optional[Tuple[Tensor, Tensor]] = None,
    feature_cfg: FeatureConfig = FeatureConfig(),
    distance_cfg: DistanceConfig = DistanceConfig(),
) -> Tuple[Dict[str, Tensor], Dict[str, Tensor], Dict[str, Dict[str, float]]]:
    """
    Compute features for X and Y and compare.
    Returns (FX, FY, distances)
    """
    evaluator = SignalMetricsEvaluator(feature_cfg=feature_cfg, distance_cfg=distance_cfg)
    FX = evaluator.features(X, L, degrees=degrees, eig=eig)
    FY = evaluator.features(Y, L, degrees=degrees, eig=eig)
    D = evaluator.compare(FX, FY)
    return FX, FY, D


def log_to_summary_writer(
    distances: Dict[str, Dict[str, float]],
    writer: Optional[object] = None,
    global_step: Optional[int] = None,
    prefix: str = "metrics/",
) -> None:
    """
    Optionally log distances to a torch.utils.tensorboard.SummaryWriter-like object.
    """
    if writer is None:
        return
    try:
        for metric, d in distances.items():
            for name, val in d.items():
                tag = f"{prefix}{metric}/{name}"
                writer.add_scalar(tag, val, global_step)
    except Exception:
        pass


__all__ = [
    "FeatureConfig",
    "DistanceConfig",
    "SignalMetricsEvaluator",
    "extract_feature_set",
    "evaluate_signal_sets",
    "total_variation",
    "spectral_centroid",
    "degree_correlation",
    "mmd_distance",
    "wasserstein_1d",
    "gaussian_kl_1d",
    "ks_statistic",
    "_eig_L",
    "_degree_from_adjacency",
    "log_to_summary_writer",
]
