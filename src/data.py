import os
import pickle
import numpy as np
import pandas as pd
import networkx as nx
from scipy.io import loadmat

import torch
from torch.utils.data import DataLoader, TensorDataset


# ----------------------------------- SBM -------------------------------------
def create_sbm_graph(n_per_comm=10, p_in=0.8, p_out=0.05, seed=42):
    sizes = [n_per_comm, n_per_comm]
    probs = [[p_in, p_out],
             [p_out, p_in]]
    G = nx.stochastic_block_model(sizes, probs, seed=seed)
    G = nx.Graph(G)  # simple, undirected
    G.remove_edges_from(nx.selfloop_edges(G))
    return G

def graph_matrices(G):
    A = nx.to_numpy_array(G, dtype=float)
    deg = A.sum(axis=1)
    L = np.diag(deg) - A

    with np.errstate(divide='ignore'):
        d_sqrt_inv = np.where(deg > 0, 1.0 / np.sqrt(deg), 0.0)
    D_sqrt_inv = np.diag(d_sqrt_inv)
    L_sym = np.eye(A.shape[0]) - D_sqrt_inv @ A @ D_sqrt_inv

    return A, L, L_sym

def generate_signals(G, n_signals=100, var=1.0, seed=0):
    rng = np.random.default_rng(seed)
    N = G.number_of_nodes()

    means = np.concatenate([np.ones(N // 2), -np.ones(N - N // 2)])
    X_raw = rng.normal(loc=means, scale=np.sqrt(var), size=(n_signals, N))

    A = nx.to_numpy_array(G, dtype=float)
    deg = A.sum(axis=1)
    with np.errstate(divide='ignore'):
        d_inv = np.where(deg > 0, 1.0 / deg, 0.0)
    D_inv = np.diag(d_inv)

    S = D_inv @ A                # row i is 1/deg(i) over neighbors
    X_smooth = X_raw @ S.T       # shape (n_signals, N)

    return X_raw, X_smooth

def generate_sbm_data(path="data/sbm"):
    os.makedirs(path, exist_ok=True)

    G = create_sbm_graph(n_per_comm=10, p_in=0.8, p_out=0.05, seed=42)
    A, L, L_norm = graph_matrices(G)

    _, X_smooth = generate_signals(G, n_signals=500, var=1.0, seed=0)
    _, X_smooth_test = generate_signals(G, n_signals=500, var=1.0, seed=0)

    np.save(f'{path}/adj.npy', A)
    np.save(f'{path}/lap.npy', L)
    np.save(f'{path}/lap_norm.npy', L_norm)

    np.save(f'{path}/x_train.npy', X_smooth)
    np.save(f'{path}/x_test.npy', X_smooth_test)

def load_sbm(path="data/sbm"):
    L = np.load(f"{path}/lap.npy")
    L_norm = np.load(f"{path}/lap_norm.npy")
    x0 = np.load(f"{path}/x_train.npy")
    x0_test = np.load(f"{path}/x_test.npy")

    X0 = torch.as_tensor(x0, dtype=torch.float32)            # (B, N)
    X0_test = torch.as_tensor(x0_test, dtype=torch.float32)  # (B, N)
    return X0, X0_test, L, L_norm


# --------------------------------- METR-LA -----------------------------------
def load_metr_la(adj_pkl_path="data/mtrla/adj_METR-LA.pkl", h5_path="data/mtrla/METR-LA.h5"):
    with open(adj_pkl_path, "rb") as f:
        adj_data = pickle.load(f, encoding="latin1")
    A = np.array(adj_data[2], dtype=np.float32)
    A = ((A + A.T) > 0).astype(np.float32)
    np.fill_diagonal(A, 0)

    G = nx.from_numpy_array(A)
    nodelist = sorted(max(nx.connected_components(G), key=len))
    G = G.subgraph(nodelist).copy()
    A = nx.to_numpy_array(G, nodelist=nodelist, dtype=np.float32)

    df = pd.read_hdf(h5_path, key="df")
    X = df.values
    X = X[:, nodelist].astype(np.float32)

    d = A.sum(axis=1)
    L = np.diag(d) - A
    with np.errstate(divide="ignore"):
        d_inv_sqrt = 1.0 / np.sqrt(d)
        d_inv_sqrt[~np.isfinite(d_inv_sqrt)] = 0.0
    D_inv_sqrt = np.diag(d_inv_sqrt)
    L_norm = np.eye(A.shape[0], dtype=np.float32) - D_inv_sqrt @ A @ D_inv_sqrt
    return X, A, L.astype(np.float32), L_norm.astype(np.float32)

# --------------------------------- Molene ------------------------------------
def load_molene(mat_path="data/molene/grasp_molene_data.mat", k=5):
    mat = loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    G = mat["molene_graphs"][1]                      # temperature graph (37 stations)
    X = G.data[0].T.astype(np.float32)               # (744, 37) hourly temperatures
    D = G.distances                                  # (37, 37) distances between stations

    # Symmetric k-NN graph from the distances between stations
    N = D.shape[0]
    idx = np.argsort(D, axis=1)[:, 1:k+1]
    A = np.zeros((N, N), dtype=np.float32)
    A[np.arange(N)[:, None], idx] = 1.0
    A = np.maximum(A, A.T)

    d = A.sum(axis=1)
    L = np.diag(d) - A
    d_inv_sqrt = 1.0 / np.sqrt(d)
    D_inv_sqrt = np.diag(d_inv_sqrt)
    L_norm = np.eye(N, dtype=np.float32) - D_inv_sqrt @ A @ D_inv_sqrt
    return X, A, L.astype(np.float32), L_norm.astype(np.float32)


def split_train_test(X, n_train=500, n_test=200, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(X.shape[0])
    train_idx = idx[:n_train]
    test_idx = idx[n_train:n_train+n_test]
    X_train = torch.as_tensor(X[train_idx], dtype=torch.float32)
    X_test = torch.as_tensor(X[test_idx], dtype=torch.float32)
    return X_train, X_test


# --------------------------------- Loaders -----------------------------------
def get_loaders(X0, X0_test, batch_size=128):
    B = X0.shape[0]
    batch_size = max(1, min(batch_size, B))

    dataset, dataset_test = TensorDataset(X0), TensorDataset(X0_test)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    loader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=True, drop_last=False)
    return loader, loader_test
