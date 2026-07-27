"""
Gaussian DDPM (epsilon-prediction) + DDIM sampler for the concatenation-
conditioned latent diffusion model (Diffusion 2 of the 2x2 comparison).

This is a *standard* conditional diffusion (the Palette recipe): the forward
process noises the PET latent x0, and a denoiser predicts the added noise given
the noisy latent together with the CT latent (the CT enters by channel
concatenation inside the OpenAI ``UNetModel``, see its ``condition_key``). It is
the plain-conditional-diffusion contrast to CPDM's Brownian-bridge + domain maps,
holding the VQGAN latent fixed.

Forward:   q(x_t | x0) = N(x_t; sqrt(acp_t) x0, (1 - acp_t) I)
Training:  L = || eps - eps_phi(x_t, t, CT) ||^2   (eps-prediction MSE)
Sampling:  deterministic DDIM (eta=0) over an evenly spaced timestep subsequence.

Buffers are registered so the schedule follows the owning LightningModule's
device. Operates in the VQGAN latent space (4x16x16); pixels are reached by
decoding with the frozen VQGAN afterwards.
"""

import torch
import torch.nn as nn


class GaussianDiffusion(nn.Module):
    def __init__(self, num_timesteps=1000, beta_start=1e-4, beta_end=2e-2):
        super().__init__()
        self.num_timesteps = int(num_timesteps)
        betas = torch.linspace(beta_start, beta_end, self.num_timesteps, dtype=torch.float64).float()
        alphas = 1.0 - betas
        acp = torch.cumprod(alphas, dim=0)
        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', acp)
        self.register_buffer('sqrt_acp', torch.sqrt(acp))
        self.register_buffer('sqrt_one_minus_acp', torch.sqrt(1.0 - acp))

    @staticmethod
    def _extract(a, t, shape):
        """Gather a[t] and reshape to broadcast over (B, C, H, W)."""
        out = a.gather(0, t)
        return out.view(t.shape[0], *([1] * (len(shape) - 1)))

    def sample_t(self, batch_size, device):
        """Uniform integer timesteps in [0, T)."""
        return torch.randint(0, self.num_timesteps, (batch_size,), device=device)

    def q_sample(self, x0, t, noise):
        """Forward diffusion: x_t = sqrt(acp_t) x0 + sqrt(1-acp_t) eps."""
        return (self._extract(self.sqrt_acp, t, x0.shape) * x0
                + self._extract(self.sqrt_one_minus_acp, t, x0.shape) * noise)

    def p_losses(self, model, x0, cond, t=None, noise=None):
        """eps-prediction MSE. ``model`` is a UNetModel concatenating ``cond``."""
        if t is None:
            t = self.sample_t(x0.size(0), x0.device)
        if noise is None:
            noise = torch.randn_like(x0)
        x_t = self.q_sample(x0, t, noise)
        eps_pred = model(x_t, timesteps=t, context=cond)
        return torch.mean((noise - eps_pred) ** 2)

    @torch.no_grad()
    def ddim_sample(self, model, shape, cond, num_steps, device, eta=0.0):
        """Deterministic (eta=0) DDIM from x_T ~ N(0, I), conditioned on ``cond``.
        Returns the denoised latent x0 estimate at t=0."""
        seq = torch.linspace(self.num_timesteps - 1, 0, num_steps, device=device).round().long()
        x = torch.randn(shape, device=device)
        for i in range(num_steps):
            t = torch.full((shape[0],), int(seq[i]), device=device, dtype=torch.long)
            eps = model(x, timesteps=t, context=cond)
            acp_t = self._extract(self.alphas_cumprod, t, x.shape)
            x0_pred = (x - torch.sqrt(1.0 - acp_t) * eps) / torch.sqrt(acp_t)
            if i < num_steps - 1:
                t_prev = torch.full((shape[0],), int(seq[i + 1]), device=device, dtype=torch.long)
                acp_prev = self._extract(self.alphas_cumprod, t_prev, x.shape)
            else:
                acp_prev = torch.ones_like(acp_t)
            sigma = eta * torch.sqrt(
                (1.0 - acp_prev) / (1.0 - acp_t) * (1.0 - acp_t / acp_prev)).clamp(min=0)
            dir_xt = torch.sqrt((1.0 - acp_prev - sigma ** 2).clamp(min=0)) * eps
            x = torch.sqrt(acp_prev) * x0_pred + dir_xt
            if eta > 0 and i < num_steps - 1:
                x = x + sigma * torch.randn_like(x)
        return x
