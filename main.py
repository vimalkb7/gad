import argparse
from graph_model import GraphEpsDenoiser
from trainer import Trainer
from graph_dataset import build_er_graph


import torch
torch.set_default_dtype(torch.float64)



def parse_args():
    parser = argparse.ArgumentParser(description="Train graph diffusion model.")

    # Graph Dataset Parameters
    parser.add_argument('--num_nodes', type=int, default=5)
    parser.add_argument('--p_edge', type=float, default=0.5)
    parser.add_argument('--d_feat', type=int, default=10)
    parser.add_argument('--data_points', type=int, default=10000)

    # Initial Gaussian Mixture Distribution Parameters 

    parser.add_argument('--weights', type=int, nargs=3, default=[0.3, 0.4, 0.3], help='List of distribution weights')
    parser.add_argument('--means', type=int, nargs=3, default=[-5.0, 0.0, 4.0], help='List of distribution means')
    parser.add_argument('--stds', type=int, nargs=3, default=[1.0, 0.8, 1.2], help='List of distribution means')

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
    parser.add_argument('--train_lr', type=float, default=2e-5)
    parser.add_argument('--c', type=float, default=2)
    parser.add_argument('--gamma', type=float, default=0.9)
    parser.add_argument('--sigma', type=float, default=6)
    parser.add_argument('--dt', type=float or bool, default=None)
    parser.add_argument('--gradient_accumulate_every', type=int, default=1)
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--print_loss_every', type=int, default=10)
    parser.add_argument('--print_validation', type=int, default=10)

    parser.add_argument('--results_folder', type=str, default='/tudelft.net/staff-bulk/ewi/insy/MMC/vimal/Results/Graph_Aware_Graph_Signal_Diffusion/Output_Diagnostics/Exp3')

    return parser.parse_args()




def main():
    args = parse_args()

    print("Arguments:")
    print("==============================================================================")
    print(args)
    print("==============================================================================")
    
    Adj = build_er_graph(args.num_nodes, args.p_edge, args.random_seed)

    model = GraphEpsDenoiser(in_dim=args.d_feat, hidden=args.hid, depth=args.depth, time_dim=args.time_dim, dropout=args.dropout)

    trainer = Trainer(
        eps_network = model,
        adj0 = Adj,
        num_nodes = args.num_nodes,
        d_feat = args.d_feat,
        data_points = args.data_points,
        weights = args.weights,
        means = args.means,
        stds = args.stds,
        random_seed = args.random_seed,
        num_epochs = args.num_epochs,
        timesteps = args.timesteps,
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

    # Run the noise sweep diagnostic after training
    try:
        trainer.noise_sweep_experiment(epoch=-1)
    except Exception as e:
        print('Noise sweep experiment failed:', e)


if __name__ == "__main__":
    main()