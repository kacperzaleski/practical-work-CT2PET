"""
Shared utilities for the PMRF (flow-based) trainers.

Keeps train_pmrf_stage1.py and train_flow.py thin: config loading, dataloaders
over the full-body CT/PET .npy set, and a Lightning Trainer wired with cosine
annealing + early stopping + checkpointing, mirroring the AdamW / cosine /
patience-20 recipe from the paper's Appendix A. Logging defaults to W&B but
falls back to TensorBoard with --no-wandb (handy for CPU smoke tests offline).
"""

import argparse

import yaml
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from torch.utils.data import DataLoader

from datasets.CT2PETAlignedDataset import CT2PETAlignedDataset


def load_config(path):
    with open(path, 'r') as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def build_datasets(cfg, data_root=None, max_samples=None):
    data_cfg = dict(cfg['data']['dataset_config'])
    if data_root is not None:
        data_cfg['data_root'] = data_root
    if max_samples is not None:
        # CLI override: single int applied to all splits.
        data_cfg['max_samples'] = max_samples
    dataset_config = argparse.Namespace(**data_cfg)
    train_ds = CT2PETAlignedDataset(dataset_config, stage='train')
    val_ds = CT2PETAlignedDataset(dataset_config, stage='val')
    return train_ds, val_ds


def build_dataloaders(train_ds, val_ds, batch_size, num_workers):
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=False, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=False, drop_last=True)
    return train_loader, val_loader


def add_common_args(parser):
    parser.add_argument('--config', type=str, default='config/PMRF-autoPET.yaml')
    parser.add_argument('--data-root', type=str, default='data/processed_fullbody')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--max-epochs', type=int, default=100)
    parser.add_argument('--max-samples', type=int, default=None,
                        help='Cap samples per split (CPU smoke tests).')
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--base-ch', type=int, default=None,
                        help='Override base channel count from the YAML model.unet block.')
    parser.add_argument('--patience', type=int, default=20,
                        help='Early-stopping patience (epochs) on val loss.')
    parser.add_argument('--limit-train-batches', type=float, default=1.0)
    parser.add_argument('--limit-val-batches', type=float, default=1.0)
    parser.add_argument('--gpu', type=int, default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--no-wandb', action='store_true')
    parser.add_argument('--wandb-name', type=str, default=None)
    parser.add_argument('--log-every-n-steps', type=int, default=20)
    parser.add_argument('--print-every', type=int, default=20)
    return parser


def get_unet_cfg(cfg, base_ch_override=None):
    """Pull the residual U-Net architecture from the YAML model.unet block,
    with an optional CLI base_ch override. Returns a kwargs dict."""
    unet = dict(cfg.get('model', {}).get('unet', {}))
    base_ch = base_ch_override if base_ch_override is not None else int(unet.get('base_ch', 64))
    ch_mult = tuple(unet.get('ch_mult', (1, 2, 4)))
    num_res_blocks = int(unet.get('num_res_blocks', 2))
    return {'base_ch': base_ch, 'ch_mult': ch_mult, 'num_res_blocks': num_res_blocks}


def parse_limit(v):
    return int(v) if v > 1 else float(v)


def build_logger(args, cfg, default_project, default_name, run_config):
    if args.no_wandb:
        return TensorBoardLogger(save_dir='logs/PMRF', name=default_name)
    wandb_cfg = cfg.get('wandb', {})
    logger = WandbLogger(
        project=wandb_cfg.get('project', default_project),
        name=args.wandb_name or default_name,
        log_model=False,
    )
    logger.experiment.config.update(run_config)
    return logger


def build_trainer(args, logger, monitor, ckpt_dir, ckpt_prefix, log_dir):
    checkpoint_cb = ModelCheckpoint(
        dirpath=ckpt_dir, filename=ckpt_prefix + '-{epoch:02d}',
        monitor=monitor, mode='min', save_top_k=2, save_last=True,
    )
    early_stop_cb = EarlyStopping(monitor=monitor, mode='min', patience=args.patience)
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    trainer = pl.Trainer(
        accelerator='gpu' if args.gpu is not None else 'cpu',
        devices=[args.gpu] if args.gpu is not None else 1,
        max_epochs=args.max_epochs,
        logger=logger,
        callbacks=[checkpoint_cb, early_stop_cb, lr_monitor],
        default_root_dir=log_dir,
        log_every_n_steps=args.log_every_n_steps,
        num_sanity_val_steps=1,
        enable_progress_bar=False,
        enable_model_summary=True,
        limit_train_batches=parse_limit(args.limit_train_batches),
        limit_val_batches=parse_limit(args.limit_val_batches),
        gradient_clip_val=1.0,
    )
    return trainer, checkpoint_cb
