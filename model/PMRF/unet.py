"""
2D residual U-Net used by the PMRF pipeline.

A single architecture serves two roles, toggled by ``use_time``:
  * ``use_time=False`` — the Stage-1 posterior-mean predictor f_theta(x): maps a
    CT slice to the (MSE-optimal, smooth) PET estimate.
  * ``use_time=True``  — the rectified-flow vector field v_phi(z_t, t): a
    time-conditioned net that regresses the constant transport field y - z0.
    Time is injected into every residual block via a sinusoidal embedding + MLP
    (FiLM-style additive bias), the standard diffusion/flow conditioning trick.

Deliberately compact (GroupNorm + SiLU + 3x3 convs, no attention by default) so
it trains CPU-only on 64x64 slices. Channels follow ``base * ch_mult`` and the
spatial resolution halves at each stage, matching the VQGAN config style used
elsewhere in the repo.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def timestep_embedding(t, dim, max_period=10000):
    """Sinusoidal embedding of a (B,) tensor of times in [0, 1] -> (B, dim)."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(half, dtype=torch.float32, device=t.device) / half
    )
    args = t.float()[:, None] * freqs[None, :]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


def _norm(ch):
    # GroupNorm with a divisor-safe group count.
    groups = min(32, ch)
    while ch % groups:
        groups -= 1
    return nn.GroupNorm(groups, ch)


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, temb_dim=None, dropout=0.0):
        super().__init__()
        self.norm1 = _norm(in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.temb_proj = nn.Linear(temb_dim, out_ch) if temb_dim else None
        self.norm2 = _norm(out_ch)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, temb=None):
        h = self.conv1(F.silu(self.norm1(x)))
        if self.temb_proj is not None and temb is not None:
            h = h + self.temb_proj(F.silu(temb))[:, :, None, None]
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
        return h + self.skip(x)


class Downsample(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.op = nn.Conv2d(ch, ch, 3, stride=2, padding=1)

    def forward(self, x):
        return self.op(x)


class Upsample(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.op = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        return self.op(x)


class ResUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, base=64, ch_mult=(1, 2, 4),
                 num_res_blocks=2, dropout=0.0, use_time=False):
        super().__init__()
        self.use_time = use_time
        temb_dim = base * 4 if use_time else None
        self.temb_dim = temb_dim
        if use_time:
            self.time_mlp = nn.Sequential(
                nn.Linear(base, temb_dim), nn.SiLU(), nn.Linear(temb_dim, temb_dim)
            )
            self.time_base = base

        self.in_conv = nn.Conv2d(in_channels, base, 3, padding=1)

        # Encoder
        self.down_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        chans = [base]
        cur = base
        for level, mult in enumerate(ch_mult):
            out_ch = base * mult
            blocks = nn.ModuleList()
            for _ in range(num_res_blocks):
                blocks.append(ResBlock(cur, out_ch, temb_dim, dropout))
                cur = out_ch
                chans.append(cur)
            self.down_blocks.append(blocks)
            if level != len(ch_mult) - 1:
                self.downsamples.append(Downsample(cur))
                chans.append(cur)
            else:
                self.downsamples.append(None)

        # Bottleneck
        self.mid1 = ResBlock(cur, cur, temb_dim, dropout)
        self.mid2 = ResBlock(cur, cur, temb_dim, dropout)

        # Decoder (mirrors encoder, consuming skip connections)
        self.up_blocks = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        for level, mult in reversed(list(enumerate(ch_mult))):
            out_ch = base * mult
            blocks = nn.ModuleList()
            for _ in range(num_res_blocks + 1):
                blocks.append(ResBlock(cur + chans.pop(), out_ch, temb_dim, dropout))
                cur = out_ch
            self.up_blocks.append(blocks)
            if level != 0:
                self.upsamples.append(Upsample(cur))
            else:
                self.upsamples.append(None)

        self.out_norm = _norm(cur)
        self.out_conv = nn.Conv2d(cur, out_channels, 3, padding=1)

    def forward(self, x, t=None):
        temb = None
        if self.use_time:
            assert t is not None, 'time-conditioned ResUNet requires t'
            temb = self.time_mlp(timestep_embedding(t, self.time_base))

        h = self.in_conv(x)
        hs = [h]
        for level, blocks in enumerate(self.down_blocks):
            for block in blocks:
                h = block(h, temb)
                hs.append(h)
            if self.downsamples[level] is not None:
                h = self.downsamples[level](h)
                hs.append(h)

        h = self.mid1(h, temb)
        h = self.mid2(h, temb)

        for level, blocks in enumerate(self.up_blocks):
            for block in blocks:
                h = block(torch.cat([h, hs.pop()], dim=1), temb)
            if self.upsamples[level] is not None:
                h = self.upsamples[level](h)

        return self.out_conv(F.silu(self.out_norm(h)))
