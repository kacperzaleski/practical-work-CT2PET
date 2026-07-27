"""
Rectified-flow path and sampler for PMRF (Posterior-Mean Rectified Flow).

Implements the optimal-transport (straight-line) coupling from the paper:

    z_t   = (1 - t) * z0 + t * y          (linear interpolation, Eq. 3)
    v*    = y - z0                          (constant target velocity, Eq. 4)
    loss  = || v_phi(z_t, t) - (y - z0) ||  (flow-matching, Eq. 6)

with the starting sample z0 built by the trainer:
    * PMRF      : z0 = posterior_mean(x) + sigma_s * eps
    * RF baseline: z0 = x + sigma_s * eps   (noised CT directly)
    * concat cond: z0 = eps (pure noise); conditioning CT is concatenated to z_t
      as an extra input channel, i.e. the field is v_phi(z_t, CT, t).

When a `cond` tensor is supplied it is concatenated to z_t along the channel
dim before the net call (net input channels must account for it); cond=None
keeps the single-channel (z_t, t) path used by PMRF / RF.

Inference integrates the learned ODE with K explicit Euler steps (Eq. 7):
    z_{k+1} = z_k + (1/K) * v_phi(z_k, k/K),   k = 0 .. K-1.
"""

import torch


class RectifiedFlow:
    def __init__(self, sigma_s=0.1):
        self.sigma_s = float(sigma_s)

    def perturb(self, base):
        """z0 = base + sigma_s * eps, eps ~ N(0, I). `base` is PM output or CT."""
        return base + self.sigma_s * torch.randn_like(base)

    @staticmethod
    def sample_t(batch_size, device):
        """t ~ U[0, 1], shape (B,)."""
        return torch.rand(batch_size, device=device)

    @staticmethod
    def interpolate(z0, y, t):
        """z_t = (1 - t) z0 + t y. t is (B,), broadcast over CHW."""
        t = t.view(-1, *([1] * (z0.dim() - 1)))
        return (1.0 - t) * z0 + t * y

    @staticmethod
    def target(z0, y):
        """Constant transport field v* = y - z0."""
        return y - z0

    @staticmethod
    def _net_input(z_t, cond):
        """z_t, or [z_t, cond] concatenated on the channel dim if cond is given."""
        return z_t if cond is None else torch.cat([z_t, cond], dim=1)

    def flow_matching_loss(self, net, z0, y, cond=None):
        """E_t || v_phi(z_t[,cond], t) - (y - z0) ||^2  (mean over batch & pixels)."""
        t = self.sample_t(z0.size(0), z0.device)
        z_t = self.interpolate(z0, y, t)
        v_pred = net(self._net_input(z_t, cond), t)
        return torch.mean((v_pred - self.target(z0, y)) ** 2)

    @torch.no_grad()
    def euler_sample(self, net, z0, num_steps, cond=None):
        """Integrate dz/dt = v_phi(z[,cond], t) from z0 (t=0) to t=1 with K Euler steps."""
        z = z0
        dt = 1.0 / num_steps
        for k in range(num_steps):
            t = torch.full((z.size(0),), k * dt, device=z.device)
            z = z + dt * net(self._net_input(z, cond), t)
        return z
