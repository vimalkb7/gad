import networkx as nx
import torch
import numpy as np
from torch.utils.data import Dataset

def build_er_graph(n: int, p_edge: float, seed: int = 0):
    if seed is not None:
        np.random.seed(seed)
    G = nx.erdos_renyi_graph(n, p_edge, seed=seed)
    A = nx.to_numpy_array(G, dtype=float)
    return torch.tensor(A, dtype=torch.float64)


# ----------------------- Graph dataset ----------------------------
class GraphSubgraphDataset(Dataset):
    def __init__(self,M: int, n_nodes: int, d_feat: int,
                    weights=np.array([0.3, 0.4, 0.3]),
                    means=np.array([-5.0, 0.0, 4.0]),
                    stds =np.array([1.0, 0.8, 1.2]),
                    seed: int = 0):
        super().__init__()

        g = torch.Generator().manual_seed(seed)
        weights = torch.tensor(weights, dtype=torch.float64)
        means   = torch.tensor(means,   dtype=torch.float64)
        stds    = torch.tensor(stds,    dtype=torch.float64)

        K = weights.numel()
        probs = weights / weights.sum()
        comps = torch.multinomial(probs, num_samples=M*n_nodes*d_feat, replacement=True, generator=g)
        comps = comps.view(M, n_nodes, d_feat)  # (M,N,D)

        mu = means[comps]
        sd = stds[comps]
        eps = torch.randn(M, n_nodes, d_feat, dtype=torch.float64, generator=g)
        self.X0 = mu + sd * eps
        self.M = M

    def __len__(self):
        return self.M

    def __getitem__(self, index):
        return self.X0[index]