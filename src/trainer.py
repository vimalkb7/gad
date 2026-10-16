import torch
import torch.nn.functional as F


def train_step(batch_x0, device, net, opt, sde):
    batch_x0 = batch_x0.to(device)
    B = batch_x0.shape[0]

    t = torch.rand(B, device=device) * sde.T   # sample in [0, T]
    x_t = sde.marginal_sampling(batch_x0, t)   # exact draw from p(x_t | x_0)
    x0_hat = net(x_t, t)
    loss = F.mse_loss(x0_hat, batch_x0)

    opt.zero_grad()
    loss.backward()
    opt.step()
    return loss.item()

@torch.no_grad()
def eval_step(batch_x0, device, net, sde):
    batch_x0 = batch_x0.to(device)
    B = batch_x0.shape[0]

    t = torch.rand(B, device=device) * sde.T
    x_t = sde.marginal_sampling(batch_x0, t)
    x0_hat = net(x_t, t)
    loss = F.mse_loss(x0_hat, batch_x0)
    return loss.item()

@torch.no_grad()
def evaluate(loader, device, net, sde, n_batches=10):
    losses = []
    for i, (batch_x0,) in enumerate(loader):
        losses.append(eval_step(batch_x0, device, net, sde))
        if i+1 >= n_batches:
            break
    return sum(losses)/len(losses)

def train(train_loader, val_loader, device, net, opt, sde, epochs=1000, eval_every=10):
    train_losses, eval_losses = [], []
    for epoch in range(epochs):
        # Train
        total, n = 0.0, 0
        for (batch_x0,) in train_loader:
            total += train_step(batch_x0, device, net, opt, sde)
            n += 1
        avg_train = total / n if n > 0 else float('nan')
        train_losses.append(avg_train)

        # Eval
        if (epoch+1) % eval_every == 0:
            avg_eval = evaluate(val_loader, device, net, sde, n_batches=10)
            eval_losses.append((epoch+1, avg_eval))
        else:
            eval_losses.append((epoch+1, None))

    return train_losses, eval_losses

@torch.no_grad()
def sample(sde, net, shape, steps=1000, device=None):
    def score_fn(x, t):
        if t.dim() == 0:
            t = t.repeat(x.shape[0])
        x0_hat = net(x, t)                                       # (B, N)
        return sde.score(x, t, x0_hat)                           # (B, N)

    x = sde.prior_sampling(shape)                                # (B, N)
    t_grid = torch.linspace(sde.T, 0.0, steps+1, device=device)  # (steps+1,)
    dt = -(sde.T / steps)                                        # negative step

    for k in range(steps):
        t = t_grid[k]
        x = sde.euler_maruyama_step_reverse(x, t, dt, score_fn)
    return x
