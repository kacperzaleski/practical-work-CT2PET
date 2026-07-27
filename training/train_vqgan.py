"""
Lightning trainer for the VQGAN encoder/decoder used by CPDM.

Trains model/VQGAN/vqgan.py:VQModel with a CPU-friendly objective:
L1 reconstruction + codebook loss only (no perceptual / no GAN discriminator).
The original VQModel.training_step assumed Lightning 1.x optimizer_idx; we
subclass it to add a Lightning 2.x compatible step. The saved checkpoint is a
standard Lightning .ckpt — VQModel.init_from_ckpt reads its `state_dict` key
directly. Keys are identical to VQModel, so CPDM can load it via
model.VQGAN.params.ckpt_path in config/CPDM-autoPET.yaml.
"""

import sys as _sys, pathlib as _pathlib  # repo-root bootstrap (script moved into a subfolder)
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent))
import argparse
import os
import sys
import time
import yaml
from pathlib import Path

import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
from datasets.CT2PETAlignedDataset import CT2PETAlignedDataset
from model.VQGAN.vqgan import VQModel


class VQGANLightning(VQModel):
    def __init__(self, ddconfig, n_embed, embed_dim, learning_rate=1e-4, print_every=10,
                 perceptual_weight=0.0):
        # VQModel.__init__ does Encoder(**vars(ddconfig)) and instantiate_from_config(vars(lossconfig)),
        # so both must support vars(). Use argparse.Namespace.
        if isinstance(ddconfig, dict):
            ddconfig = argparse.Namespace(**ddconfig)
        super().__init__(
            ddconfig=ddconfig,
            lossconfig=argparse.Namespace(target='torch.nn.Identity'),
            n_embed=n_embed,
            embed_dim=embed_dim,
        )
        self.learning_rate = learning_rate
        self.print_every = print_every
        self._last_print_t = time.time()

        # Optional VGG-LPIPS perceptual term (improves decode sharpness for CPDM).
        # Wrapped in a list so it is NOT registered as a submodule: its frozen VGG
        # weights then stay out of the saved VQGAN state_dict, keeping the ckpt
        # key-compatible with CT2PETDiffusionModel's VQModel.init_from_ckpt.
        self.perceptual_weight = float(perceptual_weight)
        if self.perceptual_weight > 0:
            import lpips
            net = lpips.LPIPS(net='vgg')
            net.eval()
            for p in net.parameters():
                p.requires_grad_(False)
            self._perceptual = [net]
        else:
            self._perceptual = []

    def _perceptual_loss(self, x_rec, x):
        # x, x_rec are single-channel in [-1, 1]; LPIPS wants 3-channel in [-1, 1].
        net = self._perceptual[0]
        if next(net.parameters()).device != x.device:
            net.to(x.device)
        x3 = x.repeat(1, 3, 1, 1)
        rec3 = x_rec.clamp(-1, 1).repeat(1, 3, 1, 1)
        return net(rec3, x3).mean()

    def _unpack(self, batch):
        (pet, _), (ct, _) = batch
        return torch.cat([ct, pet], dim=0)

    def training_step(self, batch, batch_idx):
        t0 = time.time()
        x = self._unpack(batch)
        x_rec, qloss = self(x)
        rec = F.l1_loss(x_rec, x)
        loss = rec + qloss
        logs = {'train_rec_loss': rec, 'train_codebook_loss': qloss}
        if self.perceptual_weight > 0:
            perc = self._perceptual_loss(x_rec, x)
            loss = loss + self.perceptual_weight * perc
            logs['train_perceptual_loss'] = perc
        logs['train_loss'] = loss
        self.log_dict(logs, on_step=True, on_epoch=True, batch_size=x.size(0))
        if batch_idx % self.print_every == 0:
            dt = time.time() - t0
            since = time.time() - self._last_print_t
            self._last_print_t = time.time()
            perc_str = f' perc={logs["train_perceptual_loss"].detach().item():.4f}' if self.perceptual_weight > 0 else ''
            print(f'[train] e{self.current_epoch} step {batch_idx:5d} '
                  f'rec={rec.detach().item():.4f} q={qloss.detach().item():.4f}{perc_str} '
                  f'loss={loss.detach().item():.4f} '
                  f'step_dt={dt:.2f}s window_dt={since:.1f}s',
                  flush=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x = self._unpack(batch)
        x_rec, qloss = self(x)
        rec = F.l1_loss(x_rec, x)
        loss = rec + qloss
        logs = {'val_rec_loss': rec, 'val_codebook_loss': qloss}
        if self.perceptual_weight > 0:
            perc = self._perceptual_loss(x_rec, x)
            loss = loss + self.perceptual_weight * perc
            logs['val_perceptual_loss'] = perc
        logs['val_loss'] = loss
        self.log_dict(logs, on_epoch=True, sync_dist=True, batch_size=x.size(0))
        return loss

    def on_validation_epoch_end(self):
        metrics = {k: float(v) for k, v in self.trainer.callback_metrics.items() if k.startswith('val_')}
        print(f'[val ] e{self.current_epoch} {metrics}', flush=True)

    def configure_optimizers(self):
        params = (
            list(self.encoder.parameters())
            + list(self.decoder.parameters())
            + list(self.quantize.parameters())
            + list(self.quant_conv.parameters())
            + list(self.post_quant_conv.parameters())
        )
        return torch.optim.Adam(params, lr=self.learning_rate, betas=(0.5, 0.9))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/VQGAN-autoPET.yaml')
    parser.add_argument('--data-root', type=str, default='data/processed')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--num-workers', type=int, default=2)
    parser.add_argument('--max-epochs', type=int, default=3)
    parser.add_argument('--limit-train-batches', type=float, default=1.0)
    parser.add_argument('--limit-val-batches', type=float, default=1.0)
    parser.add_argument('--gpu', type=int, default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--wandb-name', type=str, default=None)
    parser.add_argument('--ch', type=int, default=None,
                        help='Override base channel count from YAML (smaller = faster on CPU).')
    parser.add_argument('--print-every', type=int, default=10)
    parser.add_argument('--log-every-n-steps', type=int, default=20)
    parser.add_argument('--perceptual-weight', type=float, default=None,
                        help='Weight of the VGG-LPIPS perceptual term (0 = off). '
                             'Falls back to training.perceptual_weight in the YAML, else 0.')
    parser.add_argument('--resume-ckpt', type=str, default=None,
                        help='Resume training from a Lightning .ckpt (restores model + '
                             'optimizer + epoch). Use checkpoints/VQGAN_128/last.ckpt to continue.')
    args = parser.parse_args()

    pl.seed_everything(args.seed)

    with open(args.config, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    data_cfg = dict(cfg['data']['dataset_config'])
    data_cfg['data_root'] = args.data_root
    dataset_config = argparse.Namespace(**data_cfg)

    vq_params = cfg['model']['VQGAN']['params']
    paths_cfg = cfg.get('paths', {})
    wandb_cfg = cfg.get('wandb', {})

    checkpoint_dir = paths_cfg.get('checkpoint_dir', 'checkpoints/VQGAN')
    log_dir = paths_cfg.get('log_dir', 'logs/VQGAN')
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    print(f'Loading datasets from {args.data_root}')
    train_ds = CT2PETAlignedDataset(dataset_config, stage='train')
    val_ds = CT2PETAlignedDataset(dataset_config, stage='val')
    print(f'train: {len(train_ds)} pairs   val: {len(val_ds)} pairs')

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=False, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=False, drop_last=True)

    ddconfig = dict(vq_params['ddconfig'])
    if args.ch is not None:
        ddconfig['ch'] = int(args.ch)
        print(f'[init] overriding base channel count to {args.ch}', flush=True)

    if args.perceptual_weight is not None:
        perceptual_weight = args.perceptual_weight
    else:
        perceptual_weight = float(cfg.get('training', {}).get('perceptual_weight', 0.0))
    if perceptual_weight > 0:
        print(f'[init] perceptual (VGG-LPIPS) term enabled, weight={perceptual_weight}', flush=True)

    model = VQGANLightning(
        ddconfig=ddconfig,
        n_embed=int(vq_params['n_embed']),
        embed_dim=int(vq_params['embed_dim']),
        learning_rate=1e-4,
        print_every=args.print_every,
        perceptual_weight=perceptual_weight,
    )

    wandb_logger = WandbLogger(
        project=wandb_cfg.get('project', 'CT2PET-VQGAN'),
        name=args.wandb_name or wandb_cfg.get('name', 'VQGAN-AutoPET-64x64'),
        notes=wandb_cfg.get('notes', 'VQGAN training for CT-PET encoder/decoder'),
        log_model=False,
    )
    wandb_logger.experiment.config.update({
        'batch_size': args.batch_size,
        'max_epochs': args.max_epochs,
        'limit_train_batches': args.limit_train_batches,
        'limit_val_batches': args.limit_val_batches,
        'perceptual_weight': perceptual_weight,
        'dataset_sizes': {'train': len(train_ds), 'val': len(val_ds)},
    })

    checkpoint_cb = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename='vqgan-{epoch:02d}',
        monitor='val_loss',
        mode='min',
        save_top_k=2,
        save_last=True,
        # Overwrite last.ckpt / same-epoch files in place instead of spawning
        # last-v1.ckpt, last-v2.ckpt ... so resume always points at one file.
        enable_version_counter=False,
    )
    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    def _parse_limit(v):
        return int(v) if v > 1 else float(v)

    trainer = pl.Trainer(
        accelerator='gpu' if args.gpu is not None else 'cpu',
        devices=[args.gpu] if args.gpu is not None else 1,
        max_epochs=args.max_epochs,
        logger=wandb_logger,
        callbacks=[checkpoint_cb, lr_monitor],
        default_root_dir=log_dir,
        log_every_n_steps=args.log_every_n_steps,
        num_sanity_val_steps=2,
        enable_progress_bar=False,
        enable_model_summary=True,
        limit_train_batches=_parse_limit(args.limit_train_batches),
        limit_val_batches=_parse_limit(args.limit_val_batches),
    )

    if args.resume_ckpt:
        print(f'Resuming from {args.resume_ckpt}', flush=True)
    print('Starting VQGAN training...')
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader,
                ckpt_path=args.resume_ckpt)
    print(f'Best ckpt: {checkpoint_cb.best_model_path}')
    print(f'Last ckpt: {checkpoint_dir}/last.ckpt')


if __name__ == '__main__':
    main()
