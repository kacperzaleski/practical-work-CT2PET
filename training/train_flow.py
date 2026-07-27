"""
Rectified-flow trainer — covers both flow models in the comparison.

  --mode pmrf : PMRF Stage 2. Starts from the (frozen) Stage-1 posterior mean,
                z0 = f_theta(CT) + sigma_s * eps, and learns to transport it to
                the true PET. Refines realism on top of the smooth PM estimate.
                Requires --pm-ckpt (a train_pmrf_stage1.py checkpoint).

  --mode rf   : RF baseline. Conditions directly on the noised CT,
                z0 = CT + sigma_s * eps, same architecture / loss. Probes the
                perception extreme of the perception-distortion trade-off.

  --mode cond : Concatenation-conditioned flow (Flow 1 of the 2x2). The velocity
                field v_phi(z_t, CT, t) takes the CT as an extra input channel and
                the flow runs from pure noise z0 = eps to PET. Mirrors the
                concat-conditioned diffusion on the flow side.

Both optimise the flow-matching loss || v_phi(.) - (y - z0) ||^2 with a
time-conditioned 2D residual U-Net. Examples:

  python train_flow.py --mode rf   --data-root data/processed_fullbody
  python train_flow.py --mode pmrf --pm-ckpt checkpoints/PMRF_stage1/last.ckpt
  python train_flow.py --mode cond --data-root data/processed_fullbody
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
from model.PMRF.flow import RectifiedFlow
from training.train_pmrf_stage1 import PosteriorMeanModule
import pmrf_common as C


class FlowModule(pl.LightningModule):
    def __init__(self, mode='rf', pm_ckpt=None, sigma_s=0.1, base_ch=64,
                 ch_mult=(1, 2, 4), num_res_blocks=2, lr=5e-4, max_epochs=200,
                 print_every=20):
        super().__init__()
        self.save_hyperparameters()
        assert mode in ('rf', 'pmrf', 'cond')
        self.mode = mode
        # 'cond' concatenates the CT to z_t, so the velocity net takes 2 channels.
        in_channels = 2 if mode == 'cond' else 1
        self.flow = RectifiedFlow(sigma_s=sigma_s)
        self.net = ResUNet(in_channels=in_channels, out_channels=1, base=base_ch,
                           ch_mult=tuple(ch_mult), num_res_blocks=num_res_blocks,
                           use_time=True)
        self.lr = lr
        self.max_epochs = max_epochs
        self.print_every = print_every
        self._t = time.time()

        self.pm_net = None
        if mode == 'pmrf':
            assert pm_ckpt is not None, '--mode pmrf requires --pm-ckpt'
            pm = PosteriorMeanModule.load_from_checkpoint(pm_ckpt, map_location='cpu')
            self.pm_net = pm.net.eval()
            for p in self.pm_net.parameters():
                p.requires_grad_(False)

    @staticmethod
    def _unpack(batch):
        (pet, _), (ct, _) = batch
        return ct, pet

    def _base(self, ct):
        """Starting point before noise perturbation: PM(ct) for pmrf, ct for rf."""
        if self.mode == 'pmrf':
            with torch.no_grad():
                return self.pm_net(ct)
        return ct

    def _step(self, batch):
        ct, pet = self._unpack(batch)
        if self.mode == 'cond':
            # flow from pure noise; conditioning enters via the CT channel
            z0 = torch.randn_like(pet)
            return self.flow.flow_matching_loss(self.net, z0, pet, cond=ct)
        z0 = self.flow.perturb(self._base(ct))
        return self.flow.flow_matching_loss(self.net, z0, pet)

    def training_step(self, batch, batch_idx):
        loss = self._step(batch)
        self.log('train_loss', loss, on_step=True, on_epoch=True, batch_size=batch[0][0].size(0))
        if batch_idx % self.print_every == 0:
            print(f'[{self.mode} train] e{self.current_epoch} step {batch_idx:5d} '
                  f'fm={loss.item():.5f} dt={time.time() - self._t:.1f}s', flush=True)
            self._t = time.time()
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._step(batch)
        self.log('val_loss', loss, on_epoch=True, sync_dist=True, batch_size=batch[0][0].size(0))
        return loss

    def on_validation_epoch_end(self):
        m = {k: float(v) for k, v in self.trainer.callback_metrics.items() if k.startswith('val')}
        print(f'[{self.mode} val ] e{self.current_epoch} {m}', flush=True)

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.net.parameters(), lr=self.lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.max_epochs)
        return {'optimizer': opt, 'lr_scheduler': sched}


def main():
    parser = C.add_common_args(argparse.ArgumentParser())
    parser.add_argument('--mode', type=str, default='rf', choices=['rf', 'pmrf', 'cond'])
    parser.add_argument('--pm-ckpt', type=str, default=None,
                        help='Stage-1 posterior-mean checkpoint (required for --mode pmrf)')
    parser.add_argument('--sigma-s', type=float, default=0.1)
    args = parser.parse_args()
    pl.seed_everything(args.seed)

    cfg = C.load_config(args.config)
    train_ds, val_ds = C.build_datasets(cfg, args.data_root, args.max_samples)
    print(f'mode={args.mode}  train: {len(train_ds)}  val: {len(val_ds)}', flush=True)
    train_loader, val_loader = C.build_dataloaders(train_ds, val_ds, args.batch_size, args.num_workers)

    unet_cfg = C.get_unet_cfg(cfg, args.base_ch)
    model = FlowModule(mode=args.mode, pm_ckpt=args.pm_ckpt, sigma_s=args.sigma_s,
                       lr=args.lr, max_epochs=args.max_epochs,
                       print_every=args.print_every, **unet_cfg)

    ckpt_dir = f'checkpoints/PMRF_{args.mode}'
    os.makedirs(ckpt_dir, exist_ok=True)
    name = f'PMRF-{args.mode}'
    logger = C.build_logger(args, cfg, 'CT2PET-PMRF', args.wandb_name or name,
                            {'stage': args.mode, 'sigma_s': args.sigma_s, **unet_cfg,
                             'lr': args.lr, 'batch_size': args.batch_size,
                             'train': len(train_ds), 'val': len(val_ds),
                             'pm_ckpt': args.pm_ckpt})
    trainer, ckpt_cb = C.build_trainer(args, logger, 'val_loss', ckpt_dir, args.mode, 'logs/PMRF')

    print(f'Starting {args.mode} flow training...', flush=True)
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    print(f'Best ckpt: {ckpt_cb.best_model_path}', flush=True)


if __name__ == '__main__':
    main()
