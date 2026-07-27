"""Adversarial VQGAN trainer (vanilla PyTorch, no Lightning).

Trains model/VQGAN/vqgan.py:VQModel with the *full* taming-transformers objective, the
one the original CPDM uses: L1 reconstruction + VGG-LPIPS perceptual + codebook loss +
a PatchGAN discriminator (adaptive-weighted, warmed up after `--disc-start` steps). The
Lightning path in train_vqgan.py could not run the two-optimizer GAN step under
Lightning 2.x, so this is a plain PyTorch loop instead.

Why: the discriminator-free VQGAN (train_vqgan.py) under-implements the cited model and
gives a soft decode that caps the latent-diffusion models (CPDM, concat-diffusion). A
faithful GAN-VQGAN raises that reconstruction ceiling. This trainer is kept under a NEW
name and writes to NEW checkpoint/wandb targets, so nothing in the current results moves
until the retrained diffusion models are evaluated.

Checkpoint format stays CPDM-compatible: the saved dict has a `state_dict` key holding
exactly VQModel's parameters (encoder/decoder/quantize/quant_conv/post_quant_conv), which
CT2PETDiffusionModel loads via VQModel.init_from_ckpt. Discriminator + optimizer states are
stored under separate keys (ignored by CPDM's strict=False load) for resuming.

Run (CPU):
  python training/train_vqgan_gan.py --config config/VQGAN-autoPET-fb64.yaml \
    --data-root data/processed_fullbody --batch-size 16 \
    --max-samples 12000 --val-samples 1500 --max-epochs 60 --disc-start 3000 \
    --wandb-name VQGAN-GAN-fb64
"""
import sys as _sys, pathlib as _pathlib  # repo-root bootstrap
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent))
import argparse
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from datasets.CT2PETAlignedDataset import CT2PETAlignedDataset
from model.VQGAN.vqgan import VQModel
from model.VQGAN.discriminator import (NLayerDiscriminator, weights_init,
                                       hinge_d_loss, adopt_weight, calculate_adaptive_weight)


def unpack(batch):
    """CT2PETAlignedDataset yields ((pet, name), (ct, name)); the VQGAN trains on both
    modalities (as CPDM encodes both), concatenated along the batch dim."""
    (pet, _), (ct, _) = batch
    return torch.cat([ct, pet], dim=0)


def build_dataset_config(cfg, data_root, max_train, max_val):
    dc = dict(cfg['data']['dataset_config'])
    dc['data_root'] = data_root
    caps = {}
    if max_train is not None:
        caps['train'] = int(max_train)
    if max_val is not None:
        caps['val'] = int(max_val)
    if caps:
        dc['max_samples'] = caps
    return argparse.Namespace(**dc)


@torch.no_grad()
def validate(vqmodel, perceptual, perc_w, loader, device, max_batches=None):
    vqmodel.eval()
    rec_sum, perc_sum, n = 0.0, 0.0, 0
    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        x = unpack(batch).to(device)
        xrec, qloss = vqmodel(x)
        rec_sum += F.l1_loss(xrec, x).item() * x.size(0)
        if perc_w > 0:
            perc_sum += perceptual(xrec.clamp(-1, 1).repeat(1, 3, 1, 1),
                                   x.repeat(1, 3, 1, 1)).mean().item() * x.size(0)
        n += x.size(0)
    vqmodel.train()
    return rec_sum / max(n, 1), perc_sum / max(n, 1)


@torch.no_grad()
def recon_panel(vqmodel, batch, device, k=6):
    """Return a (2, k, H, W) numpy array of [input; reconstruction] in [0,1] for logging."""
    x = unpack(batch).to(device)[:k]
    xrec, _ = vqmodel(x)
    x = ((x.clamp(-1, 1) + 1) / 2).cpu().numpy()[:, 0]
    xrec = ((xrec.clamp(-1, 1) + 1) / 2).cpu().numpy()[:, 0]
    return np.stack([x, xrec], axis=0)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--config', default='config/VQGAN-autoPET-fb64.yaml')
    p.add_argument('--data-root', default='data/processed_fullbody')
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--num-workers', type=int, default=4)
    p.add_argument('--max-epochs', type=int, default=60)
    p.add_argument('--max-samples', type=int, default=12000,
                   help='Cap train slices per epoch (CPU budget). None/-1 = full set.')
    p.add_argument('--val-samples', type=int, default=1500)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--disc-start', type=int, default=3000,
                   help='Global step at which the discriminator (GAN term) switches on.')
    p.add_argument('--disc-weight', type=float, default=None, help='override config disc_weight')
    p.add_argument('--codebook-weight', type=float, default=None)
    p.add_argument('--perceptual-weight', type=float, default=None)
    p.add_argument('--disc-layers', type=int, default=None)
    p.add_argument('--grad-clip', type=float, default=1.0)
    p.add_argument('--patience', type=int, default=15, help='early-stop patience on val rec loss')
    p.add_argument('--val-interval', type=int, default=1, help='validate every N epochs')
    p.add_argument('--ckpt-every-steps', type=int, default=300,
                   help='also write a resumable last.ckpt every N steps (guards long epochs)')
    p.add_argument('--log-every', type=int, default=50, help='log train scalars every N steps')
    p.add_argument('--img-every', type=int, default=2, help='log recon images every N epochs')
    p.add_argument('--ckpt-dir', default='checkpoints/VQGAN_gan_fb64')
    p.add_argument('--wandb-project', default='CT2PET-VQGAN')
    p.add_argument('--wandb-name', default='VQGAN-GAN-fb64')
    p.add_argument('--no-wandb', action='store_true')
    p.add_argument('--resume', default=None)
    p.add_argument('--gpu', type=int, default=None, help='CUDA index; default CPU')
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(f'cuda:{args.gpu}' if args.gpu is not None else 'cpu')
    os.makedirs(args.ckpt_dir, exist_ok=True)

    with open(args.config) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    vq = cfg['model']['VQGAN']['params']
    disc_weight = args.disc_weight if args.disc_weight is not None else float(vq.get('disc_weight', 0.75))
    codebook_weight = args.codebook_weight if args.codebook_weight is not None else float(vq.get('codebook_weight', 1.0))
    perc_w = args.perceptual_weight if args.perceptual_weight is not None else float(vq.get('perceptual_weight', 1.0))
    disc_layers = args.disc_layers if args.disc_layers is not None else \
        int(cfg['model'].get('discriminator', {}).get('params', {}).get('n_layers', 3))
    max_train = None if args.max_samples in (None, -1) else args.max_samples
    max_val = None if args.val_samples in (None, -1) else args.val_samples

    # ---- data ----
    ds_cfg = build_dataset_config(cfg, args.data_root, max_train, max_val)
    train_ds = CT2PETAlignedDataset(ds_cfg, stage='train')
    val_ds = CT2PETAlignedDataset(ds_cfg, stage='val')
    print(f'train {len(train_ds)}  val {len(val_ds)}  (each yields CT+PET => x2 images/batch)', flush=True)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, drop_last=True, pin_memory=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, drop_last=True, pin_memory=False)
    fixed_val_batch = next(iter(val_loader))

    # ---- models ----
    ddconfig = argparse.Namespace(**dict(vq['ddconfig']))
    in_ch = int(getattr(ddconfig, 'in_channels', 1))
    vqmodel = VQModel(ddconfig=ddconfig,
                      lossconfig=argparse.Namespace(target='torch.nn.Identity'),
                      n_embed=int(vq['n_embed']), embed_dim=int(vq['embed_dim'])).to(device)
    discriminator = NLayerDiscriminator(input_nc=in_ch, ndf=64, n_layers=disc_layers).to(device)
    discriminator.apply(weights_init)

    perceptual = None
    if perc_w > 0:
        import lpips
        perceptual = lpips.LPIPS(net='vgg').to(device).eval()
        for prm in perceptual.parameters():
            prm.requires_grad_(False)

    gen_params = (list(vqmodel.encoder.parameters()) + list(vqmodel.decoder.parameters())
                  + list(vqmodel.quantize.parameters()) + list(vqmodel.quant_conv.parameters())
                  + list(vqmodel.post_quant_conv.parameters()))
    opt_g = torch.optim.Adam(gen_params, lr=args.lr, betas=(0.5, 0.9))
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(0.5, 0.9))
    last_layer = vqmodel.get_last_layer()

    n_disc_params = sum(pp.numel() for pp in discriminator.parameters())
    n_gen_params = sum(pp.numel() for pp in gen_params)
    print(f'generator {n_gen_params/1e6:.1f}M  discriminator {n_disc_params/1e6:.1f}M  '
          f'disc_weight={disc_weight} codebook_weight={codebook_weight} '
          f'perceptual_weight={perc_w} disc_start={args.disc_start} disc_layers={disc_layers}',
          flush=True)

    start_epoch, global_step, best_val = 0, 0, float('inf')
    if args.resume:
        ck = torch.load(args.resume, map_location=device)
        vqmodel.load_state_dict(ck['state_dict'], strict=False)
        if 'discriminator' in ck:
            discriminator.load_state_dict(ck['discriminator'])
        if 'opt_g' in ck:
            opt_g.load_state_dict(ck['opt_g']); opt_d.load_state_dict(ck['opt_d'])
        start_epoch = ck.get('epoch', 0) + 1
        global_step = ck.get('global_step', 0)
        best_val = ck.get('best_val', float('inf'))
        print(f'resumed from {args.resume} @ epoch {start_epoch} step {global_step}', flush=True)

    # ---- wandb ----
    use_wandb = not args.no_wandb
    if use_wandb:
        try:
            import wandb
            wandb.init(project=args.wandb_project, name=args.wandb_name,
                       config={**vars(args), 'disc_weight': disc_weight,
                               'codebook_weight': codebook_weight, 'perceptual_weight': perc_w,
                               'disc_layers': disc_layers, 'ddconfig': dict(vq['ddconfig']),
                               'n_embed': vq['n_embed'], 'embed_dim': vq['embed_dim']})
        except Exception as e:  # degrade cleanly to stdout-only
            print(f'wandb init failed ({e}); continuing without it', flush=True)
            use_wandb = False

    def log(d, step=None):
        if use_wandb:
            import wandb
            wandb.log(d, step=step)

    def save(path, epoch, slim=False):
        # `state_dict` is exactly VQModel's params, so CT2PETDiffusionModel loads it via
        # VQModel.init_from_ckpt. slim=True drops optimizer/discriminator state (the CPDM-ready
        # copy, ~220MB); the full copy (for resume) also carries them.
        ckpt = {'state_dict': vqmodel.state_dict(),
                'epoch': epoch, 'global_step': global_step, 'best_val': best_val,
                'ddconfig': dict(vq['ddconfig']), 'n_embed': vq['n_embed'],
                'embed_dim': vq['embed_dim']}
        if not slim:
            ckpt.update({'discriminator': discriminator.state_dict(),
                         'opt_g': opt_g.state_dict(), 'opt_d': opt_d.state_dict()})
        torch.save(ckpt, path)

    # ---- train ----
    vqmodel.train()
    epochs_no_improve = 0
    for epoch in range(start_epoch, args.max_epochs):
        t_epoch = time.time()
        for batch in train_loader:
            x = unpack(batch).to(device)
            global_step += 1
            disc_factor = adopt_weight(1.0, global_step, threshold=args.disc_start)

            # ---- generator / autoencoder update ----
            opt_g.zero_grad(set_to_none=True)
            xrec, qloss = vqmodel(x)
            rec_l1 = F.l1_loss(xrec, x)
            if perc_w > 0:
                p_loss = perceptual(xrec.clamp(-1, 1).repeat(1, 3, 1, 1),
                                    x.repeat(1, 3, 1, 1)).mean()
            else:
                p_loss = torch.zeros((), device=device)
            nll_loss = rec_l1 + perc_w * p_loss

            if disc_factor > 0:
                g_loss = -discriminator(xrec).mean()
                d_weight = calculate_adaptive_weight(nll_loss, g_loss, last_layer, disc_weight)
            else:
                g_loss = torch.zeros((), device=device)
                d_weight = torch.zeros((), device=device)

            gen_loss = nll_loss + codebook_weight * qloss + d_weight * disc_factor * g_loss
            gen_loss.backward()
            if args.grad_clip:
                torch.nn.utils.clip_grad_norm_(gen_params, args.grad_clip)
            opt_g.step()

            # ---- discriminator update ----
            if disc_factor > 0:
                opt_d.zero_grad(set_to_none=True)
                logits_real = discriminator(x.detach())
                logits_fake = discriminator(xrec.detach())
                d_loss = disc_factor * hinge_d_loss(logits_real, logits_fake)
                d_loss.backward()
                opt_d.step()
            else:
                d_loss = torch.zeros(())

            if global_step % args.log_every == 0:
                m_rec, m_perc, m_q = rec_l1.detach().item(), p_loss.detach().item(), qloss.detach().item()
                m_g, m_d, m_dw = g_loss.detach().item(), d_loss.detach().item(), d_weight.detach().item()
                log({'train/rec_l1': m_rec, 'train/perceptual': m_perc,
                     'train/codebook': m_q, 'train/g_loss': m_g,
                     'train/d_loss': m_d, 'train/d_weight': m_dw,
                     'train/disc_factor': disc_factor, 'train/gen_total': gen_loss.detach().item(),
                     'epoch': epoch}, step=global_step)
                print(f'e{epoch} step {global_step} rec={m_rec:.4f} perc={m_perc:.4f} '
                      f'q={m_q:.4f} g={m_g:.4f} d={m_d:.4f} dw={m_dw:.3f} df={disc_factor:.0f}',
                      flush=True)

            if args.ckpt_every_steps and global_step % args.ckpt_every_steps == 0:
                save(os.path.join(args.ckpt_dir, 'last.ckpt'), epoch)  # mid-epoch resume point
                print(f'  [ckpt] last.ckpt @ step {global_step}', flush=True)

        # ---- validation + checkpoints ----
        if (epoch + 1) % args.val_interval == 0 or epoch == args.max_epochs - 1:
            val_rec, val_perc = validate(vqmodel, perceptual, perc_w, val_loader, device)
            log({'val/rec_l1': val_rec, 'val/perceptual': val_perc, 'epoch': epoch}, step=global_step)
            print(f'[val] e{epoch} rec={val_rec:.4f} perc={val_perc:.4f} '
                  f'({time.time()-t_epoch:.0f}s)', flush=True)
            save(os.path.join(args.ckpt_dir, 'last.ckpt'), epoch)
            if val_rec < best_val:
                best_val = val_rec
                epochs_no_improve = 0
                save(os.path.join(args.ckpt_dir, 'best.ckpt'), epoch, slim=True)
                print(f'  new best val rec {best_val:.4f} -> best.ckpt', flush=True)
            else:
                epochs_no_improve += 1

        if use_wandb and (epoch + 1) % args.img_every == 0:
            import wandb
            panel = recon_panel(vqmodel, fixed_val_batch, device)  # (2,k,H,W)
            imgs = [wandb.Image(np.concatenate([panel[0, j], panel[1, j]], axis=0),
                                caption=f'top=input bottom=recon #{j}') for j in range(panel.shape[1])]
            wandb.log({'val/reconstructions': imgs}, step=global_step)

        if epochs_no_improve >= args.patience:
            print(f'early stop: no val improvement for {args.patience} epochs', flush=True)
            break

    print(f'done. best val rec {best_val:.4f}. ckpts in {args.ckpt_dir}/', flush=True)
    if use_wandb:
        import wandb
        wandb.finish()


if __name__ == '__main__':
    main()
