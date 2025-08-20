import math
import copy
import torch
import random
import sys, os
from torch import nn
import torch.nn.functional as F
from functools import partial
import numpy as np
import imageio.v2 as imageio



from torch.utils import data
from pathlib import Path
from torch.optim import Adam
import matplotlib.pyplot as plt

from graph_dataset import CommunitySpec, SBMSyntheticDataset

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
        d_feat: int = 10,
        data_points: int = 1000,
        community_A_size: int = 5,
        community_B_size: int = 5,
        p_intra: int = 0.85,
        p_inter: int = 0.20,
        community_A_dsitribution = [-3.0, 0.25],
        community_B_dsitribution = [3.5, 0.30],
        random_seed: int = 42,
        num_epochs: int = 100,
        timesteps: int = 1000,
        train_max_t: int = 50,
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
        self.train_max_t = train_max_t

        dt = 1.0/timesteps if dt is None else dt

        cfg = GraphDDPMSchedule(T=timesteps, dt=dt, c=c, sigma=sigma, gamma=gamma, train_max_t=train_max_t, use_steady_state_init=True)
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

        # initialize loss tracking
        self.epoch_losses = []

        # dataset + loader
        comm0 = CommunitySpec(mean=community_A_dsitribution[0], std=community_A_dsitribution[1])
        comm1 = CommunitySpec(mean=community_B_dsitribution[0], std=community_B_dsitribution[1])
        self.ds = SBMSyntheticDataset(M = data_points,
                                        d_feat = d_feat,
                                        sizes = (community_A_size, community_B_size),
                                        p_intra = p_intra,
                                        p_inter = p_inter,
                                        seed = random_seed,
                                        comm0 = comm0,
                                        comm1 = comm1,
                                        return_pyg=False,
                                        include_edge_index=False,
                                        dtype=torch.float32,
                                        topology_seed=random_seed)

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

        # fixed validation indices
        B = min(self.val_batch_size, len(self.ds))
        g = torch.Generator().manual_seed(self.seed)   # reproducible
        self.val_indices = torch.randperm(len(self.ds), generator=g)[:B]


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
                features = batch["X"]
                
                z = self._as_features_tensor(features, device).to(torch.float32)
                adj_batch = self.base_adj

                z = z.float().to(torch.float64)
                adj_batch = adj_batch.float().to(torch.float64)

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
        """
        Forward clean data to k = self.train_max_t, then run the reverse process
        back to 0. At a few steps, compare the current (denoised) distribution
        against the CLEAN x0 using the same 2×2 panels as sample_validation_1.
        """
        device = next(self.model.parameters()).device
        self.model.eval()

        # ------------------- save a checkpoint -------------------
        model_dir = self.results_folder / 'model_weights'
        model_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = model_dir / f'model_epoch-{epoch}.pt'
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.opt.state_dict()
        }, str(ckpt_path))

        # ------------------- collect CLEAN x0 from dataset -------------------
        # Prefer fixed validation indices if present; else take first val_batch_size items.
        if hasattr(self, 'val_indices') and len(getattr(self, 'val_indices', [])) > 0:
            idx_list = self.val_indices.tolist()
        else:
            B_gen = getattr(self, 'val_batch_size', self.train_batch_size)
            idx_list = list(range(min(B_gen, len(self.ds))))

        clean_X_list = []
        for i in idx_list:
            item = self.ds[i]
            X = item["X"] if isinstance(item, dict) else item
            if isinstance(X, np.ndarray):
                X = torch.tensor(X)
            elif not isinstance(X, torch.Tensor):
                X = torch.as_tensor(X)
            if X.dim() == 2:  # [N,D]
                X = X.unsqueeze(0)  # -> [1,N,D]
            clean_X_list.append(X.squeeze(0))
        X0 = torch.stack(clean_X_list, dim=0).to(device=device, dtype=torch.float64)  # [B,N,D]
        print("Clean x0 batch collected from dataset.")

        # Sizes / community ordering (dataset guarantees [C0... , C1...])
        if hasattr(self, 'ds') and hasattr(self.ds, 'cfg'):
            n0, n1 = tuple(self.ds.cfg.sizes)
        else:
            N = X0.shape[1]
            n0 = getattr(self, 'community_A_size', N // 2)
            n1 = getattr(self, 'community_B_size', N - n0)
        N = n0 + n1
        mask0 = torch.zeros(N, dtype=torch.bool, device=X0.device); mask0[:n0] = True
        mask1 = ~mask0

        # ------------------- forward to k_start -------------------
        k_start = int(min(getattr(self, 'train_max_t', self.model.cfg.T), self.model.cfg.T))
        k_vec   = torch.full((X0.shape[0],), k_start, device=device, dtype=torch.long)
        Xk, _, _ = self.model.q_sample(X0, k=k_vec)      # [B,N,D] noisy at k_start
        X = Xk.clone()                                    # current state to be reversed

        # ------------------- helper metrics/plotting -------------------
        def ks_w1(a, b):
            a_sorted, _ = torch.sort(a); b_sorted, _ = torch.sort(b)
            grid = torch.cat([a_sorted, b_sorted]).sort().values
            def ecdf(sample, grid_vals):
                idx = torch.searchsorted(sample, grid_vals, right=True)
                return idx.to(torch.float64) / sample.numel()
            ks = torch.max(torch.abs(ecdf(a_sorted, grid) - ecdf(b_sorted, grid))).item()
            n = min(a_sorted.numel(), b_sorted.numel())
            w1 = torch.mean(torch.abs(a_sorted[:n] - b_sorted[:n])).item()
            return ks, w1

        import matplotlib.pyplot as plt

        def robust_limits(x_clean, lo=0.5, hi=99.5, pad=0.15):
            x_np = x_clean.detach().cpu().numpy()
            a, b = np.percentile(x_np, [lo, hi])
            m = b - a + 1e-9
            return a - pad*m, b + pad*m

        def zscore(x, ref):
            mu = ref.mean()
            sd = ref.std(unbiased=True) + 1e-12
            return (x - mu) / sd

        def make_panels_and_save(x_clean0, x_clean1, x_curr0, x_curr1, step, out_path):
            # metrics against CLEAN
            ks0, w10 = ks_w1(x_curr0, x_clean0)
            ks1, w11 = ks_w1(x_curr1, x_clean1)

            fig, axes = plt.subplots(2, 2, figsize=(11, 6), constrained_layout=True)

            # C0 — ZOOM
            ax = axes[0, 0]
            lo0, hi0 = robust_limits(x_clean0)
            ax.hist(x_clean0.cpu().numpy(), bins=120, range=(lo0, hi0), density=True, alpha=0.55, label='CLEAN C0')
            ax.hist(x_curr0 .cpu().numpy(), bins=120, range=(lo0, hi0), density=True, alpha=0.55, label='GEN C0')
            ax.set_title(f"C0 (ZOOM) | KS={ks0:.3f}, W1={w10:.3f}"); ax.set_xlabel("value"); ax.set_ylabel("density")
            ax.legend()

            # C0 — standardized by CLEAN
            ax = axes[0, 1]
            x0_ref_z = zscore(x_clean0, x_clean0); x0_gen_z = zscore(x_curr0, x_clean0)
            lo0z, hi0z = robust_limits(x0_ref_z)
            ax.hist(x0_ref_z.cpu().numpy(), bins=120, range=(lo0z, hi0z), density=True, alpha=0.55, label='CLEAN C0 (z)')
            ax.hist(x0_gen_z.cpu().numpy(), bins=120, range=(lo0z, hi0z), density=True, alpha=0.55, label='GEN C0 (z)')
            ax.set_title("C0 (standardized by CLEAN)"); ax.set_xlabel("z-score"); ax.set_ylabel("density"); ax.legend()

            # C1 — ZOOM
            ax = axes[1, 0]
            lo1, hi1 = robust_limits(x_clean1)
            ax.hist(x_clean1.cpu().numpy(), bins=120, range=(lo1, hi1), density=True, alpha=0.55, label='CLEAN C1')
            ax.hist(x_curr1 .cpu().numpy(), bins=120, range=(lo1, hi1), density=True, alpha=0.55, label='GEN C1')
            ax.set_title(f"C1 (ZOOM) | KS={ks1:.3f}, W1={w11:.3f}"); ax.set_xlabel("value"); ax.set_ylabel("density")
            ax.legend()

            # C1 — standardized by CLEAN
            ax = axes[1, 1]
            x1_ref_z = zscore(x_clean1, x_clean1); x1_gen_z = zscore(x_curr1, x_clean1)
            lo1z, hi1z = robust_limits(x1_ref_z)
            ax.hist(x1_ref_z.cpu().numpy(), bins=120, range=(lo1z, hi1z), density=True, alpha=0.55, label='CLEAN C1 (z)')
            ax.hist(x1_gen_z.cpu().numpy(), bins=120, range=(lo1z, hi1z), density=True, alpha=0.55, label='GEN C1 (z)')
            ax.set_title("C1 (standardized by CLEAN)"); ax.set_xlabel("z-score"); ax.set_ylabel("density"); ax.legend()

            fig.suptitle(f"Distribution check vs CLEAN x0 (epoch {epoch}, t={step})", y=1.05, fontsize=14)
            plt.savefig(out_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"[DistCheck@epoch {epoch}] Plot saved to {out_path}")

        # Helper to slice/flatten per community
        def split_flat(Xbatch):
            x0 = Xbatch[:, mask0, :].reshape(-1).to(torch.float64)
            x1 = Xbatch[:, mask1, :].reshape(-1).to(torch.float64)
            return x0, x1

        # First snapshot at t = k_start (noisy vs clean)
        x0_clean, x1_clean = split_flat(X0)
        x0_curr , x1_curr  = split_flat(X)
        make_panels_and_save(
            x_clean0=x0_clean, x_clean1=x1_clean,
            x_curr0=x0_curr , x_curr1=x1_curr,
            step=k_start,
            out_path=self.results_folder / f'dist_check_epoch{epoch}_t{k_start}.png'
        )

        # ------------------- reverse: k_start → 0, log every 5 steps -------------------
        selected_steps = set(range(k_start-5, -1, -5)) | {0}
        for step in range(k_start, 0, -1):
            k_vec_step = torch.full((X.shape[0],), step, device=device, dtype=torch.long)
            add_noise = not (step == 1)
            X = self.model.p_sample(X, k=k_vec_step, eps_model=None, adj=self.base_adj, add_noise=add_noise)

            if (step-1) in selected_steps:
                x0_curr, x1_curr = split_flat(X)
                make_panels_and_save(
                    x_clean0=x0_clean, x_clean1=x1_clean,
                    x_curr0=x0_curr , x_curr1=x1_curr,
                    step=step-1,
                    out_path=self.results_folder / f'dist_check_epoch{epoch}_t{step-1}.png'
                )

        
        
        # ------------------- make a GIF from saved panels -------------------
        try:
            from pathlib import Path

            # Find all panels for this epoch, e.g. dist_check_epoch10_t050.png ... t000.png
            png_paths = sorted(
                Path(self.results_folder).glob(f"dist_check_epoch{epoch}_t*.png"),
                key=lambda p: int(p.stem.split("_t")[-1])  # numeric sort by the t value
            )

            if png_paths:
                # Reverse order so animation runs from high t -> low t
                png_paths = list(reversed(png_paths))
                frames = [imageio.imread(str(p)) for p in png_paths]
                gif_path = Path(self.results_folder) / f"dist_check_epoch{epoch}.gif"

                # duration: seconds per frame; tweak to taste
                imageio.mimsave(gif_path, frames, duration=2.5, loop=0)
                print(f"[DistCheck@epoch {epoch}] GIF saved to {gif_path}")
            else:
                print(f"[DistCheck@epoch {epoch}] No PNGs found to build a GIF.")

        except Exception as e:
            print(f"[DistCheck@epoch {epoch}] Failed to make GIF: {e}")

        

        self.model.train()





# ============================================================================================
# ============================================================================================
