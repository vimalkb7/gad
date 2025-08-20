import argparse
from graph_model import GraphEpsDenoiser
from trainer import Trainer
from graph_dataset import CommunitySpec, build_sbm_dataset



# --- Visualization (place after build_sbm_dataset(...)) ---
import numpy as np
import torch
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib import gridspec
from pathlib import Path



import torch
torch.set_default_dtype(torch.float64)




def save_graph_plot(
    Adj: torch.Tensor,
    community: torch.Tensor,
    out_dir,
    filename: str = "sbm_graph_better.png",
    layout: str = "multipartite",           # "multipartite" | "spring" | "kamada_kawai"
    seed: int = 42,
    show_adjacency: bool = True,            # show heatmap panel
    show_edge_counts: bool = True,          # add edge count annotations in subplots
    node_size: int = 700,
):
    """
    Save a clear, structured visualization of an SBM graph.

    Panels:
      (A) Full graph: intra edges thick and colored; inter edges thin gray dashed.
      (B) Zoom-in per community (two small axes): only intra edges inside each community.
      (C) Optional adjacency heatmap with community block boundaries.

    Args
    ----
    Adj          : [N, N] torch.Tensor (0/1) adjacency (undirected).
    community    : [N] torch.LongTensor with labels 0/1; nodes are assumed to be
                   ordered as [all C0, then all C1] (as in your dataset).
    out_dir      : directory to save the figure.
    filename     : output file name.
    layout       : layout for panel (A). "multipartite" separates communities horizontally.
    seed         : layout seed (for spring).
    show_adjacency : include adjacency heatmap panel.
    show_edge_counts : annotate |E_intra| in each community.
    node_size    : matplotlib size for nodes.
    """
    # ----------- prep -----------
    A = Adj.detach().cpu().numpy()
    comm = community.detach().cpu().numpy().astype(int)
    N = A.shape[0]
    assert N == len(comm)
    G = nx.from_numpy_array(A)  # undirected

    # Colors per community
    c0, c1 = "#1f77b4", "#ff7f0e"
    node_colors = np.where(comm == 0, c0, c1)

    # Edge partition (intra vs inter)
    intra_edges_c0, intra_edges_c1, inter_edges = [], [], []
    for u, v in G.edges():
        if comm[u] == comm[v] == 0:
            intra_edges_c0.append((u, v))
        elif comm[u] == comm[v] == 1:
            intra_edges_c1.append((u, v))
        else:
            inter_edges.append((u, v))

    # Node labels
    labels = {i: f"{i}\nC{comm[i]}" for i in range(N)}

    # Main layout (panel A)
    if layout == "multipartite":
        for i in range(N):
            G.nodes[i]["subset"] = int(comm[i])
        posA = nx.multipartite_layout(G, subset_key="subset", align="horizontal", scale=2.0)
    elif layout == "spring":
        posA = nx.spring_layout(G, seed=seed, k=0.8)
    else:  # kamada_kawai
        posA = nx.kamada_kawai_layout(G)

    # Subgraph layouts (panel B)
    idx0 = np.where(comm == 0)[0]
    idx1 = np.where(comm == 1)[0]
    G0 = G.subgraph(idx0)
    G1 = G.subgraph(idx1)
    pos0 = nx.circular_layout(G0)
    pos1 = nx.circular_layout(G1)

    # ----------- figure grid -----------
    if show_adjacency:
        fig = plt.figure(figsize=(12, 6))
        gs = gridspec.GridSpec(2, 3, width_ratios=[3.0, 1.3, 1.3], height_ratios=[1.0, 1.0])
        axA = fig.add_subplot(gs[:, 0])        # big full graph
        axB0 = fig.add_subplot(gs[0, 1])       # community 0
        axB1 = fig.add_subplot(gs[0, 2])       # community 1
        axC  = fig.add_subplot(gs[1, 1:])      # adjacency heatmap spanning two cols
    else:
        fig = plt.figure(figsize=(10, 5))
        gs = gridspec.GridSpec(1, 3, width_ratios=[2.2, 1.0, 1.0])
        axA = fig.add_subplot(gs[0, 0])
        axB0 = fig.add_subplot(gs[0, 1])
        axB1 = fig.add_subplot(gs[0, 2])

    # ----------- (A) full graph -----------
    axA.set_title("Full graph (intra colored & thick, inter dashed gray)")
    # Draw inter-community edges first (faded)
    nx.draw_networkx_edges(G, posA, ax=axA, edgelist=inter_edges,
                           edge_color="#6b6b6b", width=1.5, alpha=0.6, style="dashed")
    # Draw intra edges (thicker, colored)
    nx.draw_networkx_edges(G, posA, ax=axA, edgelist=intra_edges_c0,
                           edge_color=c0, width=3.0, alpha=0.9)
    nx.draw_networkx_edges(G, posA, ax=axA, edgelist=intra_edges_c1,
                           edge_color=c1, width=3.0, alpha=0.9)
    # Draw nodes & labels
    nx.draw_networkx_nodes(G, posA, ax=axA, node_color=node_colors,
                           node_size=node_size, edgecolors="k", linewidths=0.7)
    nx.draw_networkx_labels(G, posA, ax=axA, labels=labels, font_size=9)
    axA.axis("off")
    axA.legend(handles=[Patch(facecolor=c0, edgecolor="k", label="Community 0"),
                        Patch(facecolor=c1, edgecolor="k", label="Community 1")],
               loc="best", frameon=True)

    # ----------- (B) zoom per community -----------
    def draw_comm(ax, Gc, posc, color, name):
        nx.draw_networkx_edges(Gc, posc, ax=ax, edge_color=color, width=3.0, alpha=0.95)
        nx.draw_networkx_nodes(Gc, posc, ax=ax, node_color=color,
                               node_size=node_size*0.8, edgecolors="k", linewidths=0.7)
        lbls = {i: str(i) for i in Gc.nodes()}
        nx.draw_networkx_labels(Gc, posc, ax=ax, labels=lbls, font_size=9)
        m = f"{name} (|V|={Gc.number_of_nodes()}, |E|={Gc.number_of_edges()})" if show_edge_counts else name
        ax.set_title(m)
        ax.axis("off")

    draw_comm(axB0, G0, pos0, c0, "Community 0 — intra only")
    draw_comm(axB1, G1, pos1, c1, "Community 1 — intra only")

    # ----------- (C) adjacency heatmap -----------
    if show_adjacency:
        axC.set_title("Adjacency (nodes ordered by community)")
        im = axC.imshow(A, cmap="Greys", interpolation="nearest")
        # draw block boundaries (assuming blocks are contiguous)
        n0 = len(idx0)
        axC.axhline(n0 - 0.5, color="red", lw=1.2)
        axC.axvline(n0 - 0.5, color="red", lw=1.2)
        axC.set_xlabel("node index")
        axC.set_ylabel("node index")
        # small colorbar
        cbar = plt.colorbar(im, ax=axC, fraction=0.046, pad=0.04)
        cbar.set_label("A[i,j]")

    plt.tight_layout()

    # ----------- save -----------
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / filename
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved graph plot to: {out_path}")





def parse_args():
    parser = argparse.ArgumentParser(description="Train graph diffusion model.")

    # Graph Dataset Parameters & Distribution Parameters 
    parser.add_argument('--data_points', type=int, default=10000)
    parser.add_argument('--d_feat', type=int, default=10)
    parser.add_argument('--community_A_size', type=int, default=5)
    parser.add_argument('--community_B_size', type=int, default=5)
    parser.add_argument('--p_intra', type=float, default=0.85)
    parser.add_argument('--p_inter', type=float, default=0.20)
    parser.add_argument('--community_A_dsitribution', type=float, nargs=2, default=[-3.0, 0.25])
    parser.add_argument('--community_B_dsitribution', type=float, nargs=2, default=[3.5, 0.30])

    # Validation Parameters
    parser.add_argument('--val_batch_size', type=int, default=50)
    
    # Model hyperparameters
    parser.add_argument('--hid', type=int, default=128)
    parser.add_argument('--depth', type=int, default=2)
    parser.add_argument('--time_dim', type=int, default=128)
    parser.add_argument('--dropout', type=float, default=0.0)

    # Training hyperparameters
    parser.add_argument('--random_seed', type=int, default=42)
    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--timesteps', type=int, default=1000)
    parser.add_argument('--train_batch_size', type=int, default=256)
    parser.add_argument('--train_max_t', type=int, default=50)
    parser.add_argument('--train_lr', type=float, default=2e-5)
    parser.add_argument('--c', type=float, default=1.0)
    parser.add_argument('--gamma', type=float, default=5.01)
    parser.add_argument('--sigma', type=float, default=0.1)
    parser.add_argument('--dt', type=float or bool, default=None)
    parser.add_argument('--gradient_accumulate_every', type=int, default=1)
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--print_loss_every', type=int, default=10)
    parser.add_argument('--print_validation', type=int, default=10)

    parser.add_argument('--results_folder', type=str, default='/tudelft.net/staff-bulk/ewi/insy/MMC/vimal/Results/Graph_Aware_Graph_Signal_Diffusion/Output_SBM_Graph_till_t_max/t_max_50/Exp1')

    return parser.parse_args()




def main():
    args = parse_args()

    print("Arguments:")
    print("==============================================================================")
    print(args)
    print("==============================================================================")
    
    comm0 = CommunitySpec(mean=args.community_A_dsitribution[0], std=args.community_A_dsitribution[1])
    comm1 = CommunitySpec(mean=args.community_B_dsitribution[0], std=args.community_B_dsitribution[1])

    Adj, X0, community = build_sbm_dataset(
        sizes = (args.community_A_size, args.community_B_size),
        d_feat = args.d_feat,
        p_intra = args.p_intra,
        p_inter = args.p_inter,
        seed = args.random_seed,
        comm0 = comm0,
        comm1 = comm1
    )

    print("Adj:", Adj)
    print("X0:", X0)
    print("community:", community)

    fname = f"sbm_n{args.community_A_size + args.community_B_size}_d{args.d_feat}_seed{args.random_seed}.png"
    save_graph_plot(
        Adj, community,
        out_dir=args.results_folder,
        filename=fname,
        seed=args.random_seed
    )


    model = GraphEpsDenoiser(in_dim=args.d_feat, hidden=args.hid, depth=args.depth, time_dim=args.time_dim, dropout=args.dropout)

    trainer = Trainer(
        eps_network = model,
        adj0 = Adj,
        d_feat = args.d_feat,
        data_points = args.data_points,
        community_A_size = args.community_A_size,
        community_B_size = args.community_B_size,
        p_intra = args.p_intra,
        p_inter = args.p_inter,
        community_A_dsitribution = args.community_A_dsitribution,
        community_B_dsitribution = args.community_B_dsitribution,
        random_seed = args.random_seed,
        num_epochs = args.num_epochs,
        timesteps = args.timesteps,
        train_max_t = args.train_max_t,
        train_batch_size = args.train_batch_size,
        train_lr = args.train_lr,
        c = args.c,
        gamma = args.gamma,
        sigma = args.sigma,
        gradient_accumulate_every = args.gradient_accumulate_every,
        fp16 = args.fp16,
        print_loss_every = args.print_loss_every,
        print_validation = args.print_validation,
        results_folder = args.results_folder,
        val_batch_size = args.val_batch_size,
        use_steady_state_init = True,
        dt = args.dt
    )

    trainer.train()

if __name__ == "__main__":
    main()