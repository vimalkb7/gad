import torch
import torch.nn as nn
import torch.nn.functional as F


class TimeEmbedding(nn.Module):
    def __init__(self, dim=64, max_period=10000.):
        super().__init__()
        self.dim = dim
        self.max_period = max_period

    def forward(self, t):  # t: (B,) in [0,1]
        half = self.dim // 2
        freqs = torch.exp(
            -torch.log(torch.tensor(self.max_period, device=t.device)) * torch.arange(0, half, device=t.device) / half
        )
        args = t[:, None] * freqs[None, :] * 2.0 * torch.pi
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.dim % 2 == 1:  # odd
            emb = F.pad(emb, (0,1))
        return emb  # (B, dim)

class DenoiserMLP(nn.Module):
    def __init__(self, n_nodes, t_dim=64, hidden=256):
        super().__init__()
        self.t_emb = TimeEmbedding(dim=t_dim)
        self.fc1 = nn.Linear(n_nodes + t_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, n_nodes)

    def forward(self, x_t, t):  # x_t: (B, N), t: (B,)
        temb = self.t_emb(t)
        h = torch.cat([x_t, temb], dim=-1)
        h = F.silu(self.fc1(h))
        h = F.silu(self.fc2(h))
        x0_hat = self.fc3(h)
        return x0_hat

class GraphFilterTap(nn.Module):
    def __init__(self, S: torch.Tensor, K: int, C_in: int, C_out: int,
                 activation="silu", use_layernorm=True):
        super().__init__()
        assert S.dim() == 2 and S.shape[0] == S.shape[1], "S must be square (N,N)"
        N = S.shape[0]
        self.N, self.K, self.C_in, self.C_out = N, K, C_in, C_out

        S_powers = [torch.eye(N, dtype=S.dtype, device=S.device)]
        for _ in range(1, K + 1):
            S_powers.append(S_powers[-1] @ S)
        for k, Sk in enumerate(S_powers):
            self.register_buffer(f"S{k}", Sk)

        self.theta = nn.Parameter(torch.zeros(C_in, K + 1))
        nn.init.normal_(self.theta, mean=0.0, std=0.05)

        self.mix = nn.Linear(C_in, C_out)
        nn.init.xavier_uniform_(self.mix.weight); nn.init.zeros_(self.mix.bias)

        act = (activation or "").lower()
        if act == "relu":       self.act = F.relu
        elif act == "tanh":     self.act = torch.tanh
        elif act == "gelu":     self.act = F.gelu
        elif act == "silu":     self.act = F.silu
        elif act in ("", "none"): self.act = (lambda x: x)
        else: raise ValueError(f"Unknown activation '{activation}'")

        self.ln = nn.LayerNorm(C_out) if use_layernorm else nn.Identity()

    def forward(self, x):  # x: (B, N, C_in)
        B, N, C_in = x.shape
        assert N == self.N and C_in == self.C_in

        x_bn = x.permute(0, 2, 1)  # (B, C_in, N)

        y_bn = 0.0
        for k in range(self.K + 1):
            Sk = getattr(self, f"S{k}")        # (N, N)
            xk_bn = x_bn @ Sk                  # apply S^k over nodes
            theta_k = self.theta[:, k].view(1, C_in, 1)  # (1, C_in, 1)
            y_bn = y_bn + theta_k * xk_bn      # scale per-channel

        y = y_bn.permute(0, 2, 1)              # (B, N, C_in)

        y = self.mix(y)                        # (B, N, C_out)
        y = self.act(y)
        y = self.ln(y)                         # (B, N, C_out)
        return y

class DenoiserGNN(nn.Module):
    def __init__(self, S: torch.Tensor, Ks=5, t_dim=64, C=16,
                 activation="silu", use_layernorm=True, use_residual=True):
        super().__init__()
        self.N = S.shape[0]
        self.C = C
        self.use_residual = use_residual
        self.t_emb = TimeEmbedding(dim=t_dim)

        if isinstance(Ks, int):
            Ks = [Ks, Ks]
        assert len(Ks) >= 1
        L = len(Ks)

        self.in_proj = nn.Linear(1 + t_dim, C)
        nn.init.xavier_uniform_(self.in_proj.weight); nn.init.zeros_(self.in_proj.bias)

        self.layers = nn.ModuleList([
            GraphFilterTap(S, K=k, C_in=C, C_out=C,
                                    activation=activation, use_layernorm=use_layernorm)
            for k in Ks
        ])

        self.out_proj = nn.Linear(C, 1)
        nn.init.xavier_uniform_(self.out_proj.weight); nn.init.zeros_(self.out_proj.bias)

    def forward(self, x_t, t):  # x_t: (B,N), t:(B,)
        B, N = x_t.shape
        assert N == self.N

        temb = self.t_emb(t)                                 # (B,t_dim)
        temb_nodes = temb.unsqueeze(1).expand(B, N, temb.shape[-1])  # (B,N,t_dim)

        feats = torch.cat([x_t.unsqueeze(-1), temb_nodes], dim=-1)   # (B,N,1+t_dim)
        h = self.in_proj(feats)                                      # (B,N,C)

        for layer in self.layers:
            h = layer(h)   # (B,N,C)

        out = self.out_proj(h).squeeze(-1)   # (B,N)
        return x_t + out if self.use_residual else out
