import math
import copy
import torch
import random
import sys, os
from torch import nn
import torch.nn.functional as F
from functools import partial

from torch.utils import data
from pathlib import Path
from torch.optim import Adam
import numpy as np
import matplotlib.pyplot as plt

from graph_dataset import GraphSubgraphDataset

try:
    from apex import amp
    APEX_AVAILABLE = True
except:
    APEX_AVAILABLE = False

from gaussian_diffusion import GraphDDPMSchedule, GraphGaussianDDPM

torch.autograd.set_detect_anomaly(True)


# ------------------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------------------

def loss_backwards(fp16, loss, optimizer, **kwargs):
    if fp16:
        from apex import amp
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward(**kwargs)
    else:
        loss.backward(**kwargs)

# ============================================================================================
# ============================================================================================

class Trainer(object):
    def __init__(
        self,
        eps_network: nn.Module,
        adj0: torch.tensor,
        num_nodes: int = 5,
        d_feat: int = 10,
        data_points: int = 1000,
        weights: int = [0.3, 0.4, 0.3],
        means: int = [-5.0, 0.0, 4.0],
        stds: int = [1.0, 0.8, 1.2],
        random_seed: int = 42,
        num_epochs: int = 100,
        timesteps: int = 1000,
        train_batch_size: int = 32,
        train_lr: float = 2e-5,
        c: float = 1.0,
        gamma: float = 0.3,
        sigma: float = 1.0,
        gradient_accumulate_every: int = 1,
        fp16: bool = False,
        print_loss_every: int = 10,
        print_validation: int = 10,
        results_folder: str = './results',
        val_batch_size: int = 1,
        use_steady_state_init: bool = True,
        dt: float or bool = None,
    ):

        super().__init__()

        # Choose device from eps_network if placed already, otherwise default CUDA if available
        try:
            device = next(eps_network.parameters()).device
        except StopIteration:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        dtype  = adj0.dtype

        if not isinstance(adj0, np.ndarray):
            adj0 = np.asarray(adj0)
        N = adj0.shape[0]

        adj_t = torch.tensor(adj0, dtype=dtype, device=device)
        D = torch.diag(adj_t.sum(dim=1))
        L = (D - adj_t)

        self.base_adj = adj_t

        dt = 1.0/timesteps if dt is None else dt

        cfg = GraphDDPMSchedule(T=timesteps, dt=dt, c=c, sigma=sigma, gamma=gamma, use_steady_state_init=True)
        # Build diffusion (new API)
        self.model = GraphGaussianDDPM(L, eps_network, cfg)


        # misc training settings
        self.print_loss_every = print_loss_every
        self.print_validation = print_validation
        self.train_batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every
        self.num_epochs = num_epochs
        self.train_lr = train_lr
        self.seed = random_seed
        self.val_batch_size = val_batch_size
        self.d_feat = d_feat

        self.weights = torch.tensor(weights, dtype=torch.float64)
        self.means   = torch.tensor(means,   dtype=torch.float64)
        self.stds    = torch.tensor(stds,    dtype=torch.float64)

        # initialize loss tracking
        self.epoch_losses = []

        # dataset + loader
        self.ds = GraphSubgraphDataset(M = data_points, 
                                        n_nodes = num_nodes, 
                                        d_feat = d_feat,
                                        weights = weights,
                                        means = means,
                                        stds = stds,
                                        seed = random_seed)

        self.dl = data.DataLoader(self.ds, batch_size=self.train_batch_size, shuffle=True, pin_memory=True)

        self.opt = Adam(self.model.parameters(), lr=self.train_lr)
        self.step = 0

        assert not fp16 or fp16 and APEX_AVAILABLE, 'Apex must be installed in order for mixed precision training to be turned on'
        self.fp16 = fp16
        if fp16:
            from apex import amp
            (self.model, ), self.opt = amp.initialize([self.model], self.opt, opt_level='O1')

        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok = True)


    # -------------------- utilities --------------------
    @staticmethod
    def _as_features_tensor(features, device):
        """Robustly convert dataset 'features' batch to [B,N,D] tensor."""
        if isinstance(features, torch.Tensor):
            return features.to(device)
        if isinstance(features, list):
            # list of Tensors -> stack
            if isinstance(features[0], torch.Tensor):
                return torch.stack(features, dim=0).to(device)
            # list of numpy arrays
            return torch.tensor(np.stack(features, axis=0), dtype=torch.float32, device=device)
        if isinstance(features, np.ndarray):
            return torch.tensor(features, dtype=torch.float32, device=device)
        # fallback
        return torch.as_tensor(features, dtype=torch.float32, device=device)

    # -------------------- diagnostics --------------------
    @torch.no_grad()
    def noise_sweep_experiment(self, epoch: int = -1, ks = None, num_batches: int = 1):
        """
        Diagnostic loop:
        - take data x0
        - add noise at level k (forward)
        - reverse from k→0
        - collect MSE(x̂0, x0) vs k
        Saves plot to results folder and prints a compact table.
        """
        device = next(self.model.parameters()).device
        if ks is None:
            T = self.model.cfg.T
            ks = sorted({1, max(2, T//100), T//50, T//20, T//10, T//5, T//3, T//2, (3*T)//4, T})
            ks = [int(k) for k in ks if 1 <= k <= T]

        self.model.eval()
        mses = torch.zeros(len(ks), dtype=torch.float64)
        count = 0

        for b_idx, batch in enumerate(self.dl):
            if b_idx >= num_batches:
                break
            x0 = self._as_features_tensor(batch, device)   # [B,N,D]
            res = self.model.sweep_noisy_reconstruction(x0, self.base_adj, self.model.denoise_fn, ks)
            mses += res['mse'].to(mses)
            count += 1

            if b_idx == 0:
                import matplotlib.pyplot as plt
                plt.figure()
                plt.plot(res['k'], res['mse'].cpu().numpy(), marker='o')
                plt.xlabel('k (timestep)')
                plt.ylabel('Reconstruction MSE to x0')
                title_ep = f'epoch {epoch}' if epoch >= 0 else 'diagnostic'
                plt.title(f'Noisy→denoise sweep — {title_ep}')
                fig_path = self.results_folder / f'noise_sweep_mse_epoch{epoch if epoch>=0 else "X"}.png'
                plt.savefig(fig_path, bbox_inches='tight')
                plt.close()

        mses /= max(1, count)
        print('[NoiseSweep] k vs MSE:')
        print(' '.join([f'{k}:{mses[i].item():.5f}' for i, k in enumerate(ks)]))
        return ks, mses


    # -------------------- training --------------------
    def train(self):
        backwards = partial(loss_backwards, self.fp16)
        device = next(self.model.parameters()).device

        for epoch in range(1, self.num_epochs + 1):
            self.model.train()
            running_loss = 0.0
            epoch_loss_sum = 0.0
            epoch_batch_count = 0

            for i, batch in enumerate(self.dl, start=1):
                features = batch
                
                z = self._as_features_tensor(features, device).to(torch.float32)
                adj_batch = self.base_adj

                z = z.to(torch.float64)
                adj_batch = adj_batch.to(torch.float64)

                # Expect z: [B,N,D]; if single graph given, add batch dim
                if z.dim() == 2:
                    z = z.unsqueeze(0)

                # New API: compute ε-prediction loss directly
                loss = self.model.loss_epsilon_matching(eps_model=None, x0=z, adj=adj_batch)

                backwards(loss / self.gradient_accumulate_every, self.opt)
                running_loss += loss.item()
                epoch_loss_sum += loss.item()
                epoch_batch_count += 1

                if i % self.gradient_accumulate_every == 0:
                    self.opt.step()
                    self.opt.zero_grad()

                # if i % self.print_loss_every == 0:
                #     print(f'Epoch {epoch}, Batch {i}, Loss {running_loss/self.print_loss_every:.4f}')
                #     running_loss = 0.0

            # average loss for the epoch
            avg_loss = epoch_loss_sum / epoch_batch_count if epoch_batch_count > 0 else 0.0
            self.epoch_losses.append(avg_loss)

            if epoch % self.print_loss_every == 0:
                print(f'Epoch {epoch}, Loss {avg_loss:.4f}')
            

            # save loss plot
            self._save_loss_plot()

            # periodic sampling
            if epoch % self.print_validation == 0:
                self._sample_validation(epoch)
                print(f'Epoch {epoch}, Validation samples saved')

        print('Training completed.')

    def _save_loss_plot(self):
        """
        Plot and save the training loss curve to the results folder.
        """
        plt.figure()
        epochs = list(range(1, len(self.epoch_losses) + 1))
        plt.plot(epochs, self.epoch_losses)
        plt.xlabel('Epoch')
        plt.ylabel('Average Loss')
        plt.title('Training Loss per Epoch')
        plot_path = self.results_folder / 'train_loss.png'
        plt.savefig(plot_path)
        plt.close()


    @torch.no_grad()
    def _sample_validation(self, epoch: int):
        device = next(self.model.parameters()).device

        # save a checkpoint
        model_dir = self.results_folder / 'model_weights'
        model_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = model_dir / f'model_epoch-{epoch}.pt'
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.opt.state_dict()
        }, str(ckpt_path))

        # Determine sample shape
        batch_size  = getattr(self, 'val_batch_size', self.train_batch_size)
        feature_dim = getattr(self, 'd_feat', 4)


        # Reverse sampling: x_0 ~ p(x_0)
        samples = self.model.sample(
            B = batch_size, 
            D = feature_dim, 
            eps_model = None, 
            adj = self.base_adj,
        )  # [B, N, D]

        print("Samples generated")





        # ------------------------------------------------------------
        # Distribution check: do generated samples match the training GMM?
        # ------------------------------------------------------------
        with torch.no_grad():
            # Flatten all node/feature values across the validation batch
            x_gen = samples.detach().to(torch.float64).reshape(-1)
            M_ref = x_gen.numel()

            # Put params on same device
            probs  = (self.weights / self.weights.sum()).to(x_gen.device)
            means  = self.means.to(x_gen.device)
            stds   = self.stds.to(x_gen.device)

            # Draw a reference sample from the target GMM with the same size
            comps = torch.multinomial(probs, num_samples=M_ref, replacement=True)
            ref   = means[comps] + stds[comps] * torch.randn(M_ref, dtype=torch.float64, device=x_gen.device)

            # Theoretical moments of the GMM
            mu_th  = (probs * means).sum()
            var_th = (probs * (stds**2 + means**2)).sum() - mu_th**2

            # Empirical moments of generated sample
            mu_gen  = x_gen.mean()
            var_gen = x_gen.var(unbiased=True)

            print(f"[DistCheck@epoch {epoch}] mean(gen)={mu_gen.item():.4f} vs mean(target)={mu_th.item():.4f}")
            print(f"[DistCheck@epoch {epoch}] var(gen) ={var_gen.item():.4f} vs var(target) ={var_th.item():.4f}")

            # ---------- Two-sample KS statistic ----------
            x_sorted, _ = torch.sort(x_gen)
            y_sorted, _ = torch.sort(ref)

            # Build merged grid and empirical CDFs
            grid = torch.cat([x_sorted, y_sorted]).sort().values

            def ecdf(sample, grid_vals):
                # Fraction <= each grid point (right-inclusive)
                idx = torch.searchsorted(sample, grid_vals, right=True)
                return idx.to(torch.float64) / sample.numel()

            cdf_x = ecdf(x_sorted, grid)
            cdf_y = ecdf(y_sorted, grid)
            ks = torch.max(torch.abs(cdf_x - cdf_y)).item()

            # ---------- 1D Wasserstein (Earth Mover) distance ----------
            n = x_sorted.numel()
            w1 = torch.mean(torch.abs(x_sorted - y_sorted)).item()

            print(f"[DistCheck@epoch {epoch}] KS distance = {ks:.4f}")
            print(f"[DistCheck@epoch {epoch}] Wasserstein-1 = {w1:.4f}")

            # ---------- Plot: histogram of generated vs target PDF ----------
            import matplotlib.pyplot as plt
            import math

            plt.figure()
            plt.hist(x_gen.detach().cpu().numpy(), bins=100, density=True, alpha=0.5, label='Generated')

            # Target GMM PDF on a grid
            x_min = torch.min(torch.min(x_gen), torch.min(ref)).item()
            x_max = torch.max(torch.max(x_gen), torch.max(ref)).item()
            # add a small margin to avoid cutting tails
            margin = 0.1 * (x_max - x_min + 1e-6)
            g = torch.linspace(x_min - margin, x_max + margin, steps=600, device=x_gen.device, dtype=torch.float64)

            def normal_pdf(x, m, s):
                return torch.exp(-0.5 * ((x - m) / s) ** 2) / (s * math.sqrt(2.0 * math.pi))

            pdf = (probs[None, :] * normal_pdf(g[:, None], means[None, :], stds[None, :])).sum(dim=1)
            plt.plot(g.cpu().numpy(), pdf.cpu().numpy(), label='Target GMM PDF')

            plt.title(f'Distribution Check (epoch {epoch})\n'
                      f'mean(gen)={mu_gen.item():.3f} | var(gen)={var_gen.item():.3f} | KS={ks:.3f} | W1={w1:.3f}')
            plt.xlabel('value')
            plt.ylabel('density')
            plt.legend()
            out_path = self.results_folder / f'dist_check_epoch{epoch}.png'
            plt.tight_layout()
            plt.savefig(out_path)
            plt.close()

            print(f"[DistCheck@epoch {epoch}] Plot saved to {out_path}")




# ============================================================================================
# ============================================================================================
