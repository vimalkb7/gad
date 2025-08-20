# graph_dataset.py
from __future__ import annotations
import numpy as np
import networkx as nx
from dataclasses import dataclass
from typing import Tuple, Optional, Dict, Any, Union

import torch
from torch.utils.data import Dataset

try:
    from torch_geometric.data import Data
    TORCH_GEOMETRIC_AVAILABLE = True
except Exception:
    TORCH_GEOMETRIC_AVAILABLE = False


# ======================= Specs =======================
@dataclass
class CommunitySpec:
    """Single Gaussian for a community (scalar mean/std, isotropic in d_feat)."""
    mean: float
    std: float


@dataclass
class SBMDatasetConfig:
    sizes: Tuple[int, int] = (5, 5)         # two communities with 5 nodes each
    p_intra: float = 0.85                   # within-community edge prob
    p_inter: float = 0.05                   # cross-community edge prob
    # One Gaussian PER community (must be well separated)
    comm0: CommunitySpec = CommunitySpec(mean=-5.0, std=0.30)
    comm1: CommunitySpec = CommunitySpec(mean=+5.0, std=0.30)

    def validate_separation(self):
        left_max = self.comm0.mean + 3.0 * self.comm0.std
        right_min = self.comm1.mean - 3.0 * self.comm1.std
        if left_max >= right_min:
            raise ValueError(
                "Community Gaussians are not well-separated (3σ overlap).\n"
                f"C0 3σ upper: {left_max:.3f}  |  C1 3σ lower: {right_min:.3f}\n"
                "Choose means/stds so that (mean0 + 3*std0) < (mean1 - 3*std1)."
            )


# ======================= Builders =======================
def _build_sbm_topology(
    cfg: SBMDatasetConfig,
    seed_graph: int
) -> Dict[str, Any]:
    """Build ONE fixed SBM topology + communities (no features)."""
    n1, n2 = cfg.sizes
    p = np.array(
        [[cfg.p_intra, cfg.p_inter],
         [cfg.p_inter, cfg.p_intra]],
        dtype=float
    )
    G = nx.generators.community.stochastic_block_model(
        sizes=list(cfg.sizes), p=p, seed=int(seed_graph),
        directed=False, selfloops=False
    )
    A = nx.to_numpy_array(G, dtype=np.float32)              # [N,N]
    ei = np.column_stack(np.nonzero(A))                     # (E, 2), undirected -> both directions
    community = np.array([0]*n1 + [1]*n2, dtype=np.int64)   # [N]
    return {"adjacency_np": A, "edge_index_np": ei, "community_np": community}


def _sample_values_for_block(
    n_nodes: int,
    d_feat: int,
    spec: CommunitySpec,
    rng: np.random.Generator
) -> np.ndarray:
    """
    Sample d_feat-dimensional node-values for one community from N(mean, std^2 I).
    Returns float32 [n_nodes, d_feat].
    """
    vals = rng.normal(loc=spec.mean, scale=spec.std, size=(n_nodes, d_feat))
    return vals.astype(np.float32)


def _sample_features_all_nodes(
    cfg: SBMDatasetConfig, d_feat: int, seed_feat: int
) -> np.ndarray:
    """Sample features for all nodes on the fixed topology."""
    n1, n2 = cfg.sizes
    rng = np.random.default_rng(seed_feat)
    x0 = _sample_values_for_block(n1, d_feat, cfg.comm0, rng)  # [n1, d_feat]
    x1 = _sample_values_for_block(n2, d_feat, cfg.comm1, rng)  # [n2, d_feat]
    return np.vstack([x0, x1]).astype(np.float32)              # [N, d_feat]


# ======================= PyTorch Dataset =======================
class SBMSyntheticDataset(Dataset):
    """
    Dataset with a **fixed topology** (same A / edge_index / communities for all items)
    and **varying node features** per __getitem__.

    Parameters
    ----------
    M : int
        Number of graphs (samples) in the dataset.
    d_feat : int
        Feature dimensionality per node.
    sizes : (int, int)
        Community sizes (default (5,5)).
    p_intra : float
        Within-community edge probability.
    p_inter : float
        Cross-community edge probability.
    seed : int
        Base seed for reproducibility.
        - Topology uses `topology_seed` if provided, else `seed`.
        - Features use `seed + index` per sample.
    comm0, comm1 : CommunitySpec
        Gaussian specs per community (must be 3σ-separated).
    return_pyg : bool
        If True, return torch_geometric.data.Data objects.
    include_edge_index : bool
        If True (and not return_pyg), include `edge_index` in the dict output.
    dtype : torch.dtype
        Tensor dtype for outputs.
    topology_seed : Optional[int]
        If provided, overrides the topology RNG seed (fixes the graph).
    """
    def __init__(
        self,
        M: int,
        d_feat: int,
        sizes: Tuple[int, int] = (5, 5),
        p_intra: float = 0.85,
        p_inter: float = 0.05,
        seed: int = 42,                    # base seed
        comm0: Optional[CommunitySpec] = None,
        comm1: Optional[CommunitySpec] = None,
        return_pyg: bool = False,
        include_edge_index: bool = True,
        dtype: torch.dtype = torch.float32,
        topology_seed: Optional[int] = None
    ):
        super().__init__()
        self.M = int(M)
        self.d_feat = int(d_feat)
        self.seed = int(seed)
        self.return_pyg = bool(return_pyg)
        self.include_edge_index = bool(include_edge_index)
        self.dtype = dtype

        comm0 = comm0 or CommunitySpec(mean=-5.0, std=0.30)
        comm1 = comm1 or CommunitySpec(mean=+5.0, std=0.30)
        self.cfg = SBMDatasetConfig(
            sizes=sizes, p_intra=p_intra, p_inter=p_inter,
            comm0=comm0, comm1=comm1
        )
        self.cfg.validate_separation()

        if self.return_pyg and not TORCH_GEOMETRIC_AVAILABLE:
            raise RuntimeError("return_pyg=True but torch_geometric is not available.")

        # --- Build a SINGLE, fixed topology ---
        gseed = self.seed if topology_seed is None else int(topology_seed)
        topo = _build_sbm_topology(self.cfg, gseed)
        self.A = torch.tensor(topo["adjacency_np"], dtype=self.dtype)             # [N,N]
        self.edge_index = torch.tensor(topo["edge_index_np"].T, dtype=torch.long) # [2,E]
        self.community = torch.tensor(topo["community_np"], dtype=torch.long)     # [N]

        self.n1, self.n2 = self.cfg.sizes

    def __len__(self):
        return self.M

    def __getitem__(self, index: int) -> Union[Dict[str, torch.Tensor], "Data"]:
        # Features vary per index; topology stays fixed
        feat_seed = self.seed + int(index)
        X_np = _sample_features_all_nodes(self.cfg, self.d_feat, feat_seed)
        X = torch.tensor(X_np, dtype=self.dtype)  # [N, d_feat]

        if self.return_pyg:
            if not TORCH_GEOMETRIC_AVAILABLE:
                raise RuntimeError("torch_geometric not available.")
            data = Data(x=X, edge_index=self.edge_index)
            data.community = self.community
            # Optional: also keep adjacency if needed:
            # data.A = self.A
            return data

        sample = {
            "A": self.A,
            "X": X,
            "community": self.community
        }
        if self.include_edge_index:
            sample["edge_index"] = self.edge_index
        return sample


# ======================= One-off convenience builder =======================
def build_sbm_dataset(
    sizes: Tuple[int, int] = (5, 5),
    d_feat: int = 1,
    p_intra: float = 0.85,
    p_inter: float = 0.05,
    seed: Optional[int] = 42,
    comm0: Optional[CommunitySpec] = None,
    comm1: Optional[CommunitySpec] = None,
    return_pyg: bool = False,
    dtype: torch.dtype = torch.float32
):
    """
    Convenience one-off builder (returns a single sample) with fixed topology.
    Returns (Adjacency, X, community) tensors when return_pyg=False,
    or a PyG Data object when return_pyg=True.
    """
    comm0 = comm0 or CommunitySpec(mean=-5.0, std=0.30)
    comm1 = comm1 or CommunitySpec(mean=+5.0, std=0.30)
    cfg = SBMDatasetConfig(
        sizes=sizes, p_intra=p_intra, p_inter=p_inter,
        comm0=comm0, comm1=comm1
    )
    cfg.validate_separation()

    seed_int = int(seed if seed is not None else 0)
    topo = _build_sbm_topology(cfg, seed_int)
    X_np = _sample_features_all_nodes(cfg, d_feat, seed_int)

    A = torch.tensor(topo["adjacency_np"], dtype=dtype)                 # [N,N]
    X = torch.tensor(X_np, dtype=dtype)                                 # [N,d_feat]
    community = torch.tensor(topo["community_np"], dtype=torch.long)    # [N]

    if return_pyg:
        if not TORCH_GEOMETRIC_AVAILABLE:
            raise RuntimeError("torch_geometric is not available.")
        edge_index = torch.tensor(topo["edge_index_np"].T, dtype=torch.long)
        data = Data(x=X, edge_index=edge_index)
        data.community = community
        # data.A = A  # optionally attach adjacency
        return data

    return A, X, community
