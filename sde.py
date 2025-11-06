import math
import torch


class VESDE:
    def __init__(self, N=1000, sigma_min=0.01, sigma_max=50.0, T=1.0, eps=1e-5, device=None):
        self.N = N
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)
        self.T = float(T)
        self.eps = eps
        self.device = device

        self._log_smin = math.log(self.sigma_min)
        self._log_ratio = math.log(self.sigma_max / self.sigma_min)
        self._c = math.sqrt(2.0 * self._log_ratio)

    def _expand_like(self, b, x):
        while b.dim() < x.dim():
            b = b.unsqueeze(-1)
        return b

    def _to_tensor_time(self, t, ref):
        if not torch.is_tensor(t):
            t = torch.tensor(t, device=ref.device if ref.is_cuda else None, dtype=ref.dtype)
        if t.dim() == 0:
            t = t.repeat(ref.shape[0])
        return t

    def _sigma(self, t):
        t = t / self.T
        t = torch.clamp(t, 0.0, 1.0)
        log_sigma = self._log_smin + t * self._log_ratio
        sigma = torch.exp(log_sigma)
        return torch.clamp(sigma, min=self.eps)

    def _g(self, t):
        return self._sigma(t) * self._c

    def sde(self, x, t):
        t = self._to_tensor_time(t, x)
        g_t = self._g(t).to(dtype=x.dtype, device=x.device)
        drift = torch.zeros_like(x)
        diffusion = g_t
        return drift, diffusion

    def reverse_sde(self, x, t, score_fn, probability_flow=False):
        drift, diffusion = self.sde(x, t)
        g2 = diffusion ** 2
        c = 0.5 if probability_flow else 1.0
        score = score_fn(x, t)
        drift_rev = drift - self._expand_like(g2, x) * score * c
        diffusion_rev = torch.zeros_like(diffusion) if probability_flow else diffusion
        return drift_rev, diffusion_rev

    def marginal_prob(self, x0, t):
        t = self._to_tensor_time(t, x0)
        sigma_t = self._sigma(t).to(dtype=x0.dtype, device=x0.device)
        mean = x0
        std = sigma_t
        return mean, std

    def prior_sampling(self, shape, device=None, dtype=torch.float32):
        dev = device or self.device or "cpu"
        return self.sigma_max * torch.randn(*shape, device=dev, dtype=dtype)

    @torch.no_grad()
    def euler_maruyama_step_forward(self, x, t, dt):
        drift, diffusion = self.sde(x, t)
        noise = torch.randn_like(x)
        x_next = x + drift * dt + self._expand_like(diffusion * math.sqrt(dt), x) * noise
        return x_next

    @torch.no_grad()
    def euler_maruyama_step_reverse(self, x, t, dt, score_fn, probability_flow=False):
        drift, diffusion = self.reverse_sde(x, t, score_fn, probability_flow=probability_flow)
        noise = torch.randn_like(x) if not probability_flow else 0.0
        x_prev = x + drift * dt
        if not probability_flow:
            x_prev = x_prev + self._expand_like(diffusion * math.sqrt(abs(dt)), x) * noise
        return x_prev


class VPSDE:
    def __init__(self, N=1000, s=0.008, T=1.0, eps=1e-5, device=None):
        self.N = N
        self.s = s 
        self.T = T
        self.eps = eps
        self.device = device

    def _alpha_bar(self, t):
        s = self.s
        t = t / self.T
        t = torch.clamp(t, 0.0, 1.0)
        theta = 0.5 * math.pi * (t + s) / (1.0 + s)
        a_bar = torch.cos(theta) ** 2
        return torch.clamp(a_bar, min=self.eps)

    def _beta(self, t):
        s = self.s
        t = t / self.T
        t = torch.clamp(t, 0.0, 1.0 - 1e-7)
        theta = 0.5 * math.pi * (t + s) / (1.0 + s)
        beta = (math.pi / (1.0 + s)) * torch.tan(theta)
        return torch.clamp(beta, min=self.eps)

    def _expand_like(self, b, x):
        while b.dim() < x.dim():
            b = b.unsqueeze(-1)
        return b

    def sde(self, x, t):
        if not torch.is_tensor(t):
            t = torch.tensor(t, device=x.device if x.is_cuda else None, dtype=x.dtype)
        if t.dim() == 0:
            t = t.repeat(x.shape[0])
        beta_t = self._beta(t).to(x.dtype).to(x.device)
        drift = -0.5 * self._expand_like(beta_t, x) * x
        diffusion = torch.sqrt(beta_t).to(x.dtype).to(x.device)
        return drift, diffusion

    def reverse_sde(self, x, t, score_fn, probability_flow=False):
        drift, diffusion = self.sde(x, t)
        beta_t = diffusion**2
        c = 0.5 if probability_flow else 1.0
        score = score_fn(x, t)
        drift_rev = drift - self._expand_like(beta_t, x) * score * c
        diffusion_rev = torch.zeros_like(diffusion) if probability_flow else diffusion
        return drift_rev, diffusion_rev

    def marginal_prob(self, x0, t):
        if not torch.is_tensor(t):
            t = torch.tensor(t, device=x0.device if x0.is_cuda else None, dtype=x0.dtype)
        if t.dim() == 0:
            t = t.repeat(x0.shape[0])
        a_bar = self._alpha_bar(t).to(x0.dtype).to(x0.device)
        mean = self._expand_like(torch.sqrt(a_bar), x0) * x0
        std = torch.sqrt(1.0 - a_bar)
        return mean, std

    def prior_sampling(self, shape, device=None, dtype=torch.float32):
        dev = device or self.device or "cpu"
        return torch.randn(*shape, device=dev, dtype=dtype)

    @torch.no_grad()
    def euler_maruyama_step_forward(self, x, t, dt):
        drift, diffusion = self.sde(x, t)
        B = x.shape[0]
        noise = torch.randn_like(x)
        x_next = x + drift * dt + self._expand_like(diffusion * math.sqrt(dt), x) * noise
        return x_next

    @torch.no_grad()
    def euler_maruyama_step_reverse(self, x, t, dt, score_fn, probability_flow=False):
        drift, diffusion = self.reverse_sde(x, t, score_fn, probability_flow=probability_flow)
        noise = torch.randn_like(x) if not probability_flow else 0.0
        x_prev = x + drift * dt
        if not probability_flow:
            x_prev = x_prev + self._expand_like(diffusion * math.sqrt(abs(dt)), x) * noise
        return x_prev


class GASDE:
    def __init__(self, L_gamma: torch.Tensor, S: float = 1.0,
                 sigma: float = 1.0, T: float = 1.0,
                 alpha: float = 4.0, c_min: float = 0.1, device=None):
        self.Lg = L_gamma.to(device)
        self.device = device or L_gamma.device
        self.dtype = L_gamma.dtype
        self.N = self.Lg.shape[0]
        self.sigma = float(sigma)
        self.T = float(T)
        self.S = float(S)
        self.alpha = float(alpha)      # NEW
        self.c_min = float(c_min)      # NEW

        lam, U = torch.linalg.eigh(self.Lg)
        self.U = U
        self.lam = lam

    def _c_of_t(self, t: torch.Tensor):
        u = (t / self.T).clamp(0.0, 1.0)
        k = (self.S - self.c_min * self.T) * (self.alpha + 1.0) / self.T
        return self.c_min + k * (u ** self.alpha)

    def _s_of_t(self, t: torch.Tensor):
        u = (t / self.T).clamp(0.0, 1.0)
        return self.c_min * t + (self.S - self.c_min * self.T) * (u ** (self.alpha + 1.0))


    def _expand_like(self, b, x):
        while b.dim() < x.dim():
            b = b.unsqueeze(-1)
        return b

    def sde(self, x, t):
        if t.dim() == 0:
            t = t.repeat(x.shape[0])

        c_t = self._c_of_t(t)     # (B,)
        drift = - (x @ self.Lg.T) * self._expand_like(c_t, x)        # (B,N)
        diffusion = torch.sqrt(c_t.clamp_min(0)) * (math.sqrt(2.0) * self.sigma)  # (B,)
        return drift, diffusion

    def reverse_sde(self, x, t, score_fn):
        drift_fwd, diffusion = self.sde(x, t)                                        
        score = score_fn(x, t)                                        # (B,N)
        drift_rev = drift_fwd - self._expand_like(diffusion**2 , x) * score
        return drift_rev, diffusion

    @torch.no_grad()
    def marginal_sampling(self, x0, t):
        if t.dim() == 0:
            t = t.repeat(x0.shape[0])
        s = self._s_of_t(t)      # (B,)

        z0 = x0 @ self.U                                                   # (B,N)
        decay = torch.exp(-s[:, None] * self.lam[None, :])                 # (B,N)
        mu_eig = (decay * z0)                                              # (B,N)
        var_eig = (self.sigma**2) * (1.0 - torch.exp(-2.0 * s[:, None] * self.lam[None, :])) / self.lam[None, :]  # (B,N)

        eps = torch.randn_like(var_eig)
        z_t = mu_eig + torch.sqrt(var_eig.clamp_min(0)) * eps
        return z_t @ self.U.T

    def prior_sampling(self, shape):
        B, N = shape
        z = torch.randn(B, N, device=self.device, dtype=torch.float32)
        x_eig = z * (self.sigma / torch.sqrt(self.lam)).view(1, -1)
        return x_eig @ self.U.T

    @torch.no_grad()
    def euler_maruyama_step_forward(self, x, t, dt):
        drift, diffusion = self.sde(x, t)
        noise = torch.randn_like(x)
        return x + drift * dt + self._expand_like(diffusion * math.sqrt(abs(dt)), x) * noise

    @torch.no_grad()
    def euler_maruyama_step_reverse(self, x, t, dt, score_fn):
        drift, diffusion = self.reverse_sde(x, t, score_fn)
        noise = torch.randn_like(x)
        return x + drift * dt + self._expand_like(diffusion * math.sqrt(abs(dt)), x) * noise
