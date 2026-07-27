"""
Train UNet for attention map generation from CT images.
Uses PyTorch Lightning with W&B logging.
"""

import sys as _sys, pathlib as _pathlib  # repo-root bootstrap (script moved into a subfolder)
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent))
import os
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
import argparse
import yaml

from datasets.AttentionMapDataset import AttentionMapDataset


class UNetAttentionMapModel(pl.LightningModule):
    """
    PyTorch Lightning module for UNet-based attention map generation.
    Learns to generate binary attention maps from CT images.
    """
    
    def __init__(
        self,
        encoder_name='resnet34',
        in_channels=1,
        out_classes=1,
        learning_rate=0.001,
        warmup_epochs=5,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Create UNet model
        self.model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights='imagenet',
            in_channels=in_channels,
            classes=out_classes,
            decoder_attention_type='scse'
        )
        
        # Loss function: Binary Cross Entropy with Dice Loss (for better balance)
        self.bce_loss = torch.nn.BCEWithLogitsLoss()
        self.dice_loss = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)
        
        # Metrics
        self.iou_metric = smp.metrics.iou_score
        self.f1_metric = smp.metrics.f1_score
        
        self.training_step_outputs = []
        self.validation_step_outputs = []
    
    def forward(self, x):
        """Forward pass."""
        return self.model(x)
    
    def _compute_loss(self, logits, mask):
        """Compute combined loss: 0.7*Dice + 0.3*BCE"""
        dice = self.dice_loss(logits, mask)
        bce = self.bce_loss(logits, mask)
        return 0.7 * dice + 0.3 * bce
    
    def _shared_step(self, batch, stage='train'):
        """Shared logic for training/validation/test step."""
        ct = batch['ct']  # (B, 1, H, W)
        mask = batch['attention_map']  # (B, 1, H, W)
        
        # Forward pass
        logits = self.forward(ct)
        
        # Compute loss
        loss = self._compute_loss(logits, mask)
        
        # Compute metrics
        with torch.no_grad():
            prob_mask = torch.sigmoid(logits)
            pred_mask = (prob_mask > 0.5).float()
            
            tp, fp, fn, tn = smp.metrics.get_stats(
                pred_mask.long(), mask.long(), mode='binary'
            )
        
        return {
            'loss': loss,
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'tn': tn,
            'logits': logits.detach(),
            'mask': mask
        }
    
    def training_step(self, batch, batch_idx):
        """Training step."""
        result = self._shared_step(batch, 'train')
        self.training_step_outputs.append(result)
        
        self.log('train_loss', result['loss'], on_step=True, on_epoch=True, prog_bar=True)
        return result['loss']
    
    def on_train_epoch_end(self):
        """Log metrics at end of training epoch."""
        outputs = self.training_step_outputs
        
        tp = torch.cat([x['tp'] for x in outputs])
        fp = torch.cat([x['fp'] for x in outputs])
        fn = torch.cat([x['fn'] for x in outputs])
        tn = torch.cat([x['tn'] for x in outputs])
        
        per_image_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction='micro-imagewise')
        dataset_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction='micro')
        f1 = smp.metrics.f1_score(tp, fp, fn, tn, reduction='micro')
        
        metrics = {
            'train_per_image_iou': per_image_iou,
            'train_dataset_iou': dataset_iou,
            'train_f1_score': f1,
        }
        
        self.log_dict(metrics, on_epoch=True, prog_bar=True)
        self.training_step_outputs.clear()
    
    def validation_step(self, batch, batch_idx):
        """Validation step."""
        result = self._shared_step(batch, 'val')
        self.validation_step_outputs.append(result)
        
        self.log('val_loss', result['loss'], on_step=False, on_epoch=True, prog_bar=True)
        return result
    
    def on_validation_epoch_end(self):
        """Log metrics at end of validation epoch."""
        outputs = self.validation_step_outputs
        
        tp = torch.cat([x['tp'] for x in outputs])
        fp = torch.cat([x['fp'] for x in outputs])
        fn = torch.cat([x['fn'] for x in outputs])
        tn = torch.cat([x['tn'] for x in outputs])
        
        per_image_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction='micro-imagewise')
        dataset_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction='micro')
        f1 = smp.metrics.f1_score(tp, fp, fn, tn, reduction='micro')
        
        metrics = {
            'val_per_image_iou': per_image_iou,
            'val_dataset_iou': dataset_iou,
            'val_f1_score': f1,
        }
        
        self.log_dict(metrics, on_epoch=True, prog_bar=True)
        self.validation_step_outputs.clear()
    
    def configure_optimizers(self):
        """Configure optimizer and scheduler."""
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.trainer.max_epochs
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',
                'frequency': 1
            }
        }


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train attention map UNet')
    parser.add_argument('--config', type=str, default='config/AttentionMap-UNet-autoPET.yaml',
                       help='Path to config file')
    parser.add_argument('--data-root', type=str, default='data/processed',
                       help='Path to processed data')
    parser.add_argument('--image-size', type=int, default=64,
                       help='Square slice size (use 128 for the head+torso CPDM run)')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--num-workers', type=int, default=4,
                       help='Number of workers')
    parser.add_argument('--max-epochs', type=int, default=50,
                       help='Maximum epochs')
    parser.add_argument('--gpu', type=int, default=None,
                       help='GPU device ID (None = CPU)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--limit-train-batches', type=float, default=1.0,
                       help='Cap on training batches per epoch (int or fraction). Useful for CPU runs.')
    parser.add_argument('--limit-val-batches', type=float, default=1.0,
                       help='Cap on validation batches per epoch (int or fraction).')
    parser.add_argument('--wandb-name', type=str, default=None,
                       help='Override W&B run name.')
    
    args = parser.parse_args()
    
    # Set seed
    pl.seed_everything(args.seed)
    
    # Load config if provided
    if os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        data_config = config.get('data', {})
        model_config = config.get('model', {})
        training_config = config.get('training', {})
        wandb_config = config.get('wandb', {})
        paths_config = config.get('paths', {})
    else:
        data_config = training_config = wandb_config = paths_config = {}
    
    # Override with command line args
    batch_size = args.batch_size or data_config.get('train', {}).get('batch_size', 32)
    num_workers = args.num_workers or data_config.get('train', {}).get('num_workers', 4)
    max_epochs = args.max_epochs or training_config.get('n_epochs', 50)
    
    # Create data directories
    results_dir = paths_config.get('results_dir', 'results/AttentionMap')
    checkpoint_dir = paths_config.get('checkpoint_dir', 'checkpoints/AttentionMap')
    log_dir = paths_config.get('log_dir', 'logs/AttentionMap')
    
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    print(f"Creating datasets from: {args.data_root}")
    
    # Create datasets
    train_dataset = AttentionMapDataset(args.data_root, split='train', image_size=args.image_size, flip=True)
    val_dataset = AttentionMapDataset(args.data_root, split='val', image_size=args.image_size, flip=False)
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Val dataset size: {len(val_dataset)}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    # Create model
    model = UNetAttentionMapModel(
        encoder_name='resnet34',
        in_channels=1,
        out_classes=1,
        learning_rate=training_config.get('lr', 0.001),
        warmup_epochs=training_config.get('warmup_epochs', 5)
    )
    
    # Initialize W&B logger
    wandb_logger = WandbLogger(
        project=wandb_config.get('project', 'CT2PET-AttentionMap'),
        name=args.wandb_name or wandb_config.get('name', 'UNet-AttentionMap-64x64'),
        notes=wandb_config.get('notes', 'UNet training for attention map generation'),
        log_model=True  # Log model checkpoints to W&B
    )
    
    # Log config to W&B
    wandb_logger.experiment.config.update({
        'batch_size': batch_size,
        'num_workers': num_workers,
        'max_epochs': max_epochs,
        'model_params': dict(model.hparams),
        'dataset_sizes': {
            'train': len(train_dataset),
            'val': len(val_dataset)
        },
        'data_root': args.data_root
    })
    
    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename='attention_map_unet-{epoch:02d}-{val_dataset_iou:.4f}',
        monitor='val_dataset_iou',
        mode='max',
        save_top_k=3,
        save_last=True
    )
    
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    
    early_stop = EarlyStopping(
        monitor='val_dataset_iou',
        patience=10,
        mode='max',
        verbose=True
    )
    
    # Lightning accepts int (exact batch count) or float in (0, 1] (fraction).
    def _parse_limit(v):
        return int(v) if v > 1 else float(v)

    # Create trainer
    trainer = pl.Trainer(
        accelerator='gpu' if args.gpu is not None else 'cpu',
        devices=[args.gpu] if args.gpu is not None else 1,
        max_epochs=max_epochs,
        logger=wandb_logger,
        callbacks=[checkpoint_callback, lr_monitor, early_stop],
        default_root_dir=log_dir,
        log_every_n_steps=10,
        num_sanity_val_steps=2,
        enable_progress_bar=True,
        enable_model_summary=True,
        limit_train_batches=_parse_limit(args.limit_train_batches),
        limit_val_batches=_parse_limit(args.limit_val_batches),
    )
    
    # Log model architecture
    print(f"\nModel Architecture:")
    print(f"Encoder: ResNet34")
    print(f"Input channels: 1")
    print(f"Output channels: 1")
    print(f"Image size: {args.image_size}x{args.image_size}")
    
    # Train
    print(f"\nStarting training...")
    print(f"Batch size: {batch_size}")
    print(f"Max epochs: {max_epochs}")
    print(f"GPU: {args.gpu if args.gpu is not None else 'CPU'}")
    
    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader
    )
    
    print(f"\nTraining complete!")
    print(f"Best checkpoint: {checkpoint_callback.best_model_path}")
    print(f"Results saved to: {results_dir}")
    
    # Save final model
    torch.save(model.state_dict(), os.path.join(results_dir, 'final_model.pt'))
    print(f"Final model saved to: {os.path.join(results_dir, 'final_model.pt')}")


if __name__ == '__main__':
    main()
