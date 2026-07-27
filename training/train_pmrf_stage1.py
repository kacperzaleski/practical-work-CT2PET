"""
PMRF Stage 1 — Posterior-Mean Predictor (and standalone distortion baseline).

Trains a 2D residual U-Net f_theta(CT) -> PET with plain voxel-wise MSE
(Eq. 1). Because MSE regression converges to the conditional mean, the output
is high-fidelity but over-smoothed; it doubles as the paper's "Residual U-Net"
distortion baseline and as the starting point for the Stage-2 rectified flow.

  python train_pmrf_stage1.py --data-root data/processed_fullbody \
      --batch-size 32 --max-epochs 200

Checkpoints land in checkpoints/PMRF_stage1/. Pass the best one to train_flow.py
--mode pmrf --pm-ckpt <ckpt> to train the refiner.
"""

import sys as _sys, pathlib as _pathlib  # repo-root bootstrap (script moved into a subfolder)
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent))
import argparse
import os
import time

import torch
import torch.nn.functional as F
import pytorch_lightning as pl

from model.PMRF.unet import ResUNet
import pmrf_common as C


class PosteriorMeanModule(pl.LightningModule):
    def __init__(self, base_ch=64, ch_mult=(1, 2, 4), num_res_blocks=2,
                 lr=5e-4, max_epochs=200, print_every=20):
        super().__init__()
        self.save_hyperparameters()
        self.net = ResUNet(in_channels=1, out_channels=1, base=base_ch,
                           ch_mult=tuple(ch_mult), num_res_blocks=num_res_blocks,
                           use_time=False)
        self.lr = lr
        self.max_epochs = max_epochs
        self.print_every = print_every
        self._t = time.time()

    def forward(self, ct):
        return self.net(ct)

    @staticmethod
    def _unpack(batch):
        (pet, _), (ct, _) = batch
        return ct, pet

    def training_step(self, batch, batch_idx):
        ct, pet = self._unpack(batch)
        pred = self(ct)
        loss = F.mse_loss(pred, pet)
        self.log('train_loss', loss, on_step=True, on_epoch=True, batch_size=ct.size(0))
        if batch_idx % self.print_every == 0:
            print(f'[s1 train] e{self.current_epoch} step {batch_idx:5d} '
                  f'mse={loss.item():.5f} dt={time.time() - self._t:.1f}s', flush=True)
            self._t = time.time()
        return loss

    def validation_step(self, batch, batch_idx):
        ct, pet = self._unpack(batch)
        loss = F.mse_loss(self(ct), pet)
        self.log('val_loss', loss, on_epoch=True, sync_dist=True, batch_size=ct.size(0))
        return loss

    def on_validation_epoch_end(self):
        m = {k: float(v) for k, v in self.trainer.callback_metrics.items() if k.startswith('val')}
        print(f'[s1 val ] e{self.current_epoch} {m}', flush=True)

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.max_epochs)
        return {'optimizer': opt, 'lr_scheduler': sched}


def main():
    parser = C.add_common_args(argparse.ArgumentParser())
    args = parser.parse_args()
    pl.seed_everything(args.seed)

    cfg = C.load_config(args.config)
    train_ds, val_ds = C.build_datasets(cfg, args.data_root, args.max_samples)
    print(f'train: {len(train_ds)}  val: {len(val_ds)}', flush=True)
    train_loader, val_loader = C.build_dataloaders(train_ds, val_ds, args.batch_size, args.num_workers)

    unet_cfg = C.get_unet_cfg(cfg, args.base_ch)
    model = PosteriorMeanModule(lr=args.lr, max_epochs=args.max_epochs,
                                print_every=args.print_every, **unet_cfg)

    ckpt_dir = 'checkpoints/PMRF_stage1'
    os.makedirs(ckpt_dir, exist_ok=True)
    logger = C.build_logger(args, cfg, 'CT2PET-PMRF', args.wandb_name or 'PMRF-stage1-PM',
                            {'stage': 'posterior_mean', **unet_cfg,
                             'lr': args.lr, 'batch_size': args.batch_size,
                             'train': len(train_ds), 'val': len(val_ds)})
    trainer, ckpt_cb = C.build_trainer(args, logger, 'val_loss', ckpt_dir, 'pm', 'logs/PMRF')

    print('Starting Stage-1 (posterior-mean) training...', flush=True)
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    print(f'Best ckpt: {ckpt_cb.best_model_path}', flush=True)


if __name__ == '__main__':
    main()
