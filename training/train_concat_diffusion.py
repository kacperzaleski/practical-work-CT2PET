"""
Concatenation-conditioned latent diffusion — Diffusion 2 of the 2x2 comparison.

A standard eps-prediction DDPM in the frozen VQGAN latent CPDM uses, conditioned
by concatenating the CT latent to the noisy PET latent (the OpenAI UNetModel does
the concat internally via condition_key='concat'). This is the plain-conditional-
diffusion contrast to CPDM (Brownian-bridge + attention/attenuation), latent fixed.

  python train_concat_diffusion.py --config config/ConcatDiff-autoPET-fb64.yaml \
      --data-root data/processed_fullbody --batch-size 8

Checkpoints land in checkpoints/ConcatDiff/. Sample + evaluate with
sample_concat_diffusion.py (dumps [0,1] .npy for eval_masked.py).
"""

import sys as _sys, pathlib as _pathlib  # repo-root bootstrap (script moved into a subfolder)
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent))
import argparse
import os
import time

import torch
import pytorch_lightning as pl

from model.VQGAN.vqgan import VQModel
from model.BrownianBridge.base.modules.diffusionmodules.openaimodel import UNetModel
from model.ConcatDiff.ddpm import GaussianDiffusion
from utils import dict2namespace
import pmrf_common as C


class ConcatDiffusionModule(pl.LightningModule):
    def __init__(self, vqgan_params, unet_params, diff_params, lr=5e-4,
                 max_epochs=100, print_every=20):
        super().__init__()
        self.save_hyperparameters()
        # Frozen VQGAN (loads its own ckpt via VQModel.__init__ -> init_from_ckpt).
        self.vqgan = VQModel(**vars(dict2namespace(vqgan_params))).eval()
        for p in self.vqgan.parameters():
            p.requires_grad_(False)
        self.unet = UNetModel(**unet_params)
        self.diffusion = GaussianDiffusion(
            num_timesteps=diff_params['num_timesteps'],
            beta_start=diff_params['beta_start'],
            beta_end=diff_params['beta_end'])
        self.sample_step = int(diff_params.get('sample_step', 200))
        self.lr = lr
        self.max_epochs = max_epochs
        self.print_every = print_every
        # LDM-style latent scaling: set on the first training batch, persisted as a
        # buffer so sampling reuses it. (On resume it reloads; recompute is skipped.)
        self.register_buffer('scale_factor', torch.tensor(1.0))
        self._scale_set = False
        self._t = time.time()

    # --- VQGAN latent I/O, identical to CT2PETDiffusionModel (no quantize on
    # encode, normalize_latent=False; quantize on decode). ---
    @torch.no_grad()
    def encode(self, x):
        h = self.vqgan.encoder(x)
        return self.vqgan.quant_conv(h)

    @torch.no_grad()
    def decode(self, latent):
        quant, _, _ = self.vqgan.quantize(latent)
        return self.vqgan.decode(quant)

    @staticmethod
    def _unpack(batch):
        (pet, _), (ct, _) = batch
        return ct, pet

    def _latents(self, ct, pet):
        with torch.no_grad():
            x0 = self.encode(pet)
            cond = self.encode(ct)
        if not self._scale_set:
            self.scale_factor.fill_(1.0 / x0.std().clamp(min=1e-6))
            self._scale_set = True
            print(f'[concatdiff] scale_factor = {self.scale_factor.item():.4f}', flush=True)
        return x0 * self.scale_factor, cond * self.scale_factor

    def training_step(self, batch, batch_idx):
        ct, pet = self._unpack(batch)
        x0, cond = self._latents(ct, pet)
        loss = self.diffusion.p_losses(self.unet, x0, cond)
        self.log('train_loss', loss, on_step=True, on_epoch=True, batch_size=ct.size(0))
        if batch_idx % self.print_every == 0:
            print(f'[cd train] e{self.current_epoch} step {batch_idx:5d} '
                  f'mse={loss.item():.5f} dt={time.time() - self._t:.1f}s', flush=True)
            self._t = time.time()
        return loss

    def validation_step(self, batch, batch_idx):
        ct, pet = self._unpack(batch)
        x0, cond = self._latents(ct, pet)
        loss = self.diffusion.p_losses(self.unet, x0, cond)
        self.log('val_loss', loss, on_epoch=True, sync_dist=True, batch_size=ct.size(0))
        return loss

    def on_validation_epoch_end(self):
        m = {k: float(v) for k, v in self.trainer.callback_metrics.items() if k.startswith('val')}
        print(f'[cd val ] e{self.current_epoch} {m}', flush=True)

    def on_load_checkpoint(self, checkpoint):
        # scale_factor came back via the buffer; don't overwrite it on resume.
        self._scale_set = True

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.unet.parameters(), lr=self.lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.max_epochs)
        return {'optimizer': opt, 'lr_scheduler': sched}


def main():
    parser = C.add_common_args(argparse.ArgumentParser())
    parser.set_defaults(config='config/ConcatDiff-autoPET-fb64.yaml')
    parser.add_argument('--ckpt-dir', default='checkpoints/ConcatDiff',
                        help='where to write checkpoints (use a fresh dir for the GAN-VQGAN rerun)')
    args = parser.parse_args()
    pl.seed_everything(args.seed)

    cfg = C.load_config(args.config)
    train_ds, val_ds = C.build_datasets(cfg, args.data_root, args.max_samples)
    print(f'train: {len(train_ds)}  val: {len(val_ds)}', flush=True)
    train_loader, val_loader = C.build_dataloaders(train_ds, val_ds, args.batch_size, args.num_workers)

    vqgan_params = cfg['model']['VQGAN']['params']
    unet_params = dict(cfg['model']['UNetParams'])
    diff_params = dict(cfg['model']['diffusion'])
    model = ConcatDiffusionModule(vqgan_params, unet_params, diff_params,
                                  lr=args.lr, max_epochs=args.max_epochs,
                                  print_every=args.print_every)
    n_unet = sum(p.numel() for p in model.unet.parameters())
    print(f'UNet denoiser params: {n_unet / 1e6:.2f}M (lighter than CPDM by design)', flush=True)

    ckpt_dir = args.ckpt_dir
    os.makedirs(ckpt_dir, exist_ok=True)
    logger = C.build_logger(args, cfg, 'CT2PET-ConcatDiff',
                            args.wandb_name or 'ConcatDiff-AutoPET-64-headtorso-cropped',
                            {'unet_params_M': round(n_unet / 1e6, 2), **diff_params,
                             'lr': args.lr, 'batch_size': args.batch_size,
                             'train': len(train_ds), 'val': len(val_ds)})
    trainer, ckpt_cb = C.build_trainer(args, logger, 'val_loss', ckpt_dir, 'cd', 'logs/ConcatDiff')

    print('Starting concat-conditioned diffusion training...', flush=True)
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    print(f'Best ckpt: {ckpt_cb.best_model_path}', flush=True)


if __name__ == '__main__':
    main()
