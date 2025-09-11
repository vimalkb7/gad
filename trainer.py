# trainer.py
# ==========
from __future__ import annotations

import copy
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils import data

from graph_dataset import CommunitySpec, SBMSyntheticDataset
from gaussian_diffusion_x0_pred import (
    GraphDDPMSchedule,
    GraphGaussianDDPM_X0,
)

import matplotlib.pyplot as plt
import random


# ------------------------------------------------------------------------------------------
# Laplacian
# ------------------------------------------------------------------------------------------
def laplacian_matrix(A: torch.Tensor, kind: str = "sym") -> torch.Tensor:
    """Build a graph Laplacian from adjacency A."""
    A = A.to(dtype=torch.get_default_dtype())
    N = A.shape[0]
    deg = A.sum(dim=1)
    if kind == "unnormalized":
        D = torch.diag(deg)
        return D - A
    elif kind == "sym":
        inv_sqrt_deg = torch.where(deg > 0, deg.rsqrt(), torch.zeros_like(deg))
        Dm12 = torch.diag(inv_sqrt_deg)
        I = torch.eye(N, device=A.device, dtype=A.dtype)
        return I - Dm12 @ A @ Dm12
    else:
        raise ValueError(f"Unknown laplacian kind: {kind!r}")


# ============================================================================================
# Trainer (simplified, x0-prediction with eigenbasis-stable DDPM)
# ============================================================================================
class Trainer:
    def __init__(
        self,
        x0_network: nn.Module,                # MLP: forward(x, t) -> x0_hat
        adj0: np.ndarray | torch.Tensor,      # fixed graph adjacency (N x N)
        d_feat: int = 10,
        data_points: int = 10000,
        community_A_size: int = 5,
        community_B_size: int = 5,
        p_intra: float = 0.85,
        p_inter: float = 0.20,
        community_A_dsitribution = (-3.0, 0.25),
        community_B_dsitribution = ( 3.5, 0.30),
        random_seed: int = 42,
        num_epochs: int = 500,
        timesteps: int = 1000,
        train_max_t: int = 50,
        train_batch_size: int = 256,
        train_lr: float = 2e-5,
        print_loss_every: int = 5,
        results_folder: str = "./results",
        val_batch_size: int = 8,
        grad_clip: float = 1.0,
        c: float = 1.0,
        gamma: float = 1.0,
        sigma: float = 0.01,
        warp: str = "loglinear",
        tau_T: float = 4.0,
        poly_p: float = 1.0,
        log_eps: float = 1e-2,
        laplacian: str = "sym",
        ema_decay: float = 0.999,
        use_ema_for_validation: bool = True,
    ):
        super().__init__()

        # device / dtype
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float64
        torch.set_default_dtype(self.dtype)

        # graph Laplacian
        if not isinstance(adj0, np.ndarray):
            adj0 = np.asarray(adj0)
        A = torch.tensor(adj0, dtype=self.dtype, device=self.device)
        L = laplacian_matrix(A, kind=laplacian)

        # diffusion schedule
        cfg = GraphDDPMSchedule(
            T=timesteps, c=c, sigma=sigma, gamma=gamma,
            train_max_t=train_max_t, warp=warp,
            tau_T=tau_T, poly_p=poly_p, log_eps=log_eps,
            dt=1.0 / timesteps if warp.lower() == "uniform" else None,
            jitter_factor=1e-6
        )

        # model
        x0_network = x0_network.to(self.device)
        self.model = GraphGaussianDDPM_X0(L, x0_network, cfg)

        # dataset
        comm0 = CommunitySpec(*community_A_dsitribution)
        comm1 = CommunitySpec(*community_B_dsitribution)
        self.ds = SBMSyntheticDataset(
            M=data_points,
            d_feat=d_feat,
            sizes=(community_A_size, community_B_size),
            p_intra=p_intra,
            p_inter=p_inter,
            seed=random_seed,
            comm0=comm0,
            comm1=comm1,
            return_pyg=False,
            include_edge_index=False,
            dtype=self.dtype,
            topology_seed=random_seed
        )
        self.dl = data.DataLoader(
            self.ds, batch_size=train_batch_size, shuffle=True, drop_last=False
        )

        # optimizer + LR scheduler
        self.opt = Adam(self.model.parameters(), lr=float(train_lr), weight_decay=1e-4)
        self.scheduler = CosineAnnealingLR(self.opt, T_max=num_epochs, eta_min=1e-6)

        # EMA
        self.ema = copy.deepcopy(self.model).eval()
        for p in self.ema.parameters(): p.requires_grad = False
        self.ema_decay = ema_decay
        self.use_ema_for_validation = use_ema_for_validation

        # misc
        self.num_epochs = num_epochs
        self.results_folder = Path(results_folder); self.results_folder.mkdir(parents=True, exist_ok=True)
        self.print_loss_every = print_loss_every
        self.val_batch_size = val_batch_size
        self.epoch_losses = []
        self.grad_clip = grad_clip

    # ----------------------------------------------------------------
    # EMA update
    # ----------------------------------------------------------------
    @torch.no_grad()
    def _update_ema(self):
        for p, q in zip(self.model.parameters(), self.ema.parameters()):
            q.data.mul_(self.ema_decay).add_(p.data, alpha=1.0 - self.ema_decay)

    # ----------------------------------------------------------------
    # Training loop
    # ----------------------------------------------------------------
    def train(self):
        for epoch in range(1, self.num_epochs + 1):
            self.model.train()
            running, batches = 0.0, 0

            for batch in self.dl:
                X = self._as_tensor(batch["X"], self.device).to(self.dtype)
                if X.dim() == 2: X = X.unsqueeze(0)

                loss = self.model.loss_x0_matching(x0_model=None, x0=X, k=None)
                loss_val = loss.detach()
                if not torch.isfinite(loss_val):
                    print("[train] Non-finite loss; skipping step.")
                    self.opt.zero_grad(set_to_none=True)
                    continue

                self.opt.zero_grad(set_to_none=True)
                loss.backward()

                # Optional: skip if any grad is non-finite
                bad_grad = False
                for p in self.model.parameters():
                    if p.grad is not None and not torch.isfinite(p.grad).all():
                        bad_grad = True
                        break
                if bad_grad:
                    print("[train] Non-finite gradient; skipping optimizer step.")
                    self.opt.zero_grad(set_to_none=True)
                    continue

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip)
                self.opt.step(); self.opt.zero_grad()
                self._update_ema()

                running += float(loss.item()); batches += 1

            avg = running / max(1, batches)
            self.epoch_losses.append(avg)
            self._save_loss_plot()

            if epoch % self.print_loss_every == 0:
                lr_now = self.opt.param_groups[0]["lr"]
                print(f"[epoch {epoch:04d}] loss={avg:.6f}  lr={lr_now:.2e}")

            self.scheduler.step()
            if epoch % (self.print_loss_every * 10) == 0:
                self._sample_and_compare(epoch)

        print("Training completed.")

    # ----------------------------------------------------------------
    # Validation: save checkpoint + histogram comparison
    # ----------------------------------------------------------------
    @torch.no_grad()
    def _sample_and_compare(self, epoch: int):
        self.model.eval(); self.ema.eval()
        sampler = self.ema if self.use_ema_for_validation else self.model

        # Save checkpoint
        ckpt_path = self.results_folder / f'epoch-{epoch:04d}.pt'
        torch.save(self.model.state_dict(), ckpt_path)
        print(f"[ckpt] Saved checkpoint: {ckpt_path}")

        # Collect clean data
        idx = torch.randperm(len(self.ds))[: self.val_batch_size].tolist()
        clean_list = []
        for i in idx:
            X = torch.as_tensor(self.ds[i]["X"], dtype=self.dtype, device=self.device)
            if X.dim() == 2: X = X.unsqueeze(0)
            clean_list.append(X.squeeze(0))
        X0_clean = torch.stack(clean_list, dim=0)  # [B,N,D]

        # Generate samples
        B, N, D = X0_clean.shape
        X_gen = sampler.sample(B=B, D=D, x0_model=None, deterministic_last=True,
                               start_k=self.model.cfg.train_max_t)

        # Histogram comparison
        def make_hist(a_clean, a_gen, title, out_path):
            plt.figure(figsize=(7.5, 4.0))
            plt.hist(a_clean.cpu().numpy().reshape(-1), bins=100, alpha=0.5, label="CLEAN")
            plt.hist(a_gen.cpu().numpy().reshape(-1), bins=100, alpha=0.5, label="GEN")
            plt.title(title); plt.xlabel("Value"); plt.ylabel("Density"); plt.legend()
            plt.tight_layout(); plt.savefig(out_path, dpi=150); plt.close()

        out_dir = self.results_folder / "histograms"; out_dir.mkdir(parents=True, exist_ok=True)

        # print("X0_clean:", X0_clean)
        # print("========================")
        # print("X_gen:", X_gen)

        make_hist(X0_clean, X_gen, f"Clean vs Gen (epoch {epoch})", out_dir / f"epoch{epoch}.png")

        self.model.train()

    # ----------------------------------------------------------------
    # Utils
    # ----------------------------------------------------------------
    @staticmethod
    def _as_tensor(features, device):
        if isinstance(features, torch.Tensor): return features.to(device)
        if isinstance(features, np.ndarray): return torch.tensor(features, dtype=torch.float32, device=device)
        return torch.as_tensor(features, dtype=torch.float32, device=device)

    def _save_loss_plot(self):
        if not self.epoch_losses: return
        plt.figure()
        plt.plot(range(1, len(self.epoch_losses) + 1), self.epoch_losses)
        plt.xlabel("Epoch"); plt.ylabel("Average Loss"); plt.title("Training Loss")
        out_png = self.results_folder / "train_loss.png"
        plt.savefig(out_png, dpi=150); plt.close()
