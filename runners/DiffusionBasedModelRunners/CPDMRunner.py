import math
import os

import numpy as np
import torch
import torch.optim.lr_scheduler
from torch.utils.data import DataLoader

from PIL import Image
from Register import Registers
from model.BrownianBridge.BrownianBridgeModel import BrownianBridgeModel
from model.BrownianBridge.LatentBrownianBridgeModel import LatentBrownianBridgeModel
from model.BrownianBridge.CT2PETDiffusionModel import CT2PETDiffusionModel
from runners.DiffusionBasedModelRunners.DiffusionBaseRunner import DiffusionBaseRunner
from runners.utils import weights_init, get_optimizer, get_dataset, make_dir, get_image_grid, save_single_image
from tqdm.autonotebook import tqdm


class WandbTBWriter:
    """
    Adapter exposing the subset of torch.utils.tensorboard.SummaryWriter that CPDMRunner /
    BaseRunner / DiffusionBaseRunner actually call (add_scalar, add_image), mirroring each
    log to both TensorBoard and Weights & Biases. Any other attribute access falls through
    to the wrapped TB writer so the rest of the runner code keeps working unmodified.
    """
    def __init__(self, tb_writer, wandb_run):
        self._tb = tb_writer
        self._wb = wandb_run  # already-initialized wandb run

    def add_scalar(self, tag, value, step):
        self._tb.add_scalar(tag, value, step)
        try:
            v = value.detach().item() if isinstance(value, torch.Tensor) else float(value)
        except Exception:
            return
        self._wb.log({tag: v}, step=int(step))

    def add_image(self, tag, image, step, dataformats='HWC'):
        self._tb.add_image(tag, image, step, dataformats=dataformats)
        import wandb as _wandb
        # image arrives as a uint8 numpy array (HWC by default in this codebase).
        self._wb.log({tag: _wandb.Image(image)}, step=int(step))

    def __getattr__(self, name):
        return getattr(self._tb, name)

@Registers.runners.register_with_name('CPDMRunner')
class CPDMRunner(DiffusionBaseRunner):
    def __init__(self, config):
        super().__init__(config)
        self._maybe_attach_wandb(config)
        self._init_eval_metrics(config)
        # Early-stopping state tracked across validation_epoch calls.
        self._best_val_loss = float('inf')
        self._patience_counter = 0
        self._stop_signaled = False
        es = getattr(config.training, 'early_stopping', None)
        self._es_patience = int(getattr(es, 'patience', 0)) if es is not None else 0
        self._es_min_delta = float(getattr(es, 'min_delta', 0.0)) if es is not None else 0.0
        self._es_enabled = self._es_patience > 0
        # Metric-keyed checkpoint selection. The paper metrics are already computed every
        # validation epoch, so selecting on them costs one extra torch.save per improvement
        # (no extra forward passes). Kept separately from top_model_*, which tracks the latent
        # val loss; the latent loss can keep falling for many epochs after SSIM/LPIPS have
        # turned, so best-by-latent-loss is not the best-by-image-quality checkpoint.
        self._best_metric_ssim = float('-inf')   # higher is better
        self._best_metric_lpips = float('inf')   # lower is better

    def _maybe_attach_wandb(self, config):
        """
        If config has a `wandb` section with a project, init a wandb run and wrap
        self.writer so add_scalar / add_image push to both TensorBoard and W&B.
        Safe to omit (or set wandb.project: null) to keep TensorBoard-only logging.

        Only rank 0 logs under DDP. Errors initializing wandb are swallowed so
        training is never blocked by network or auth issues.
        """
        if not hasattr(config, 'wandb'):
            return
        wb_cfg = config.wandb
        project = getattr(wb_cfg, 'project', None)
        if not project:
            return
        if getattr(config.training, 'use_DDP', False) and getattr(config.training, 'local_rank', 0) != 0:
            return
        try:
            import wandb
            from utils import namespace2dict  # lazy: utils.py imports CPDMRunner at top
            run = wandb.init(
                project=project,
                entity=getattr(wb_cfg, 'entity', None) or None,
                name=getattr(wb_cfg, 'name', None),
                notes=getattr(wb_cfg, 'notes', None),
                config=namespace2dict(config),
                dir=self.config.result.log_path,
                resume='allow',
            )
            self.writer = WandbTBWriter(self.writer, run)
            print(f'[wandb] logging to project={project} run={run.name}')
        except Exception as e:
            print(f'[wandb] init failed ({e!r}); falling back to TensorBoard-only logging')

    # --- paper-aligned eval metrics: LPIPS, MAE, SSIM, PSNR ---------------------------------

    def _init_eval_metrics(self, config):
        """Set how often paper metrics are recomputed (per epoch). LPIPS is lazy-loaded."""
        self._lpips_model = None
        self._metrics_eval_max_batches = int(
            getattr(getattr(config, 'testing', object()), 'metrics_eval_max_batches', 4)
        )

    def _get_lpips(self, device):
        if self._lpips_model is None:
            import lpips as _lpips
            # VGG net used by the paper.
            m = _lpips.LPIPS(net='vgg').to(device).eval()
            for p in m.parameters():
                p.requires_grad = False
            self._lpips_model = m
        return self._lpips_model

    @staticmethod
    def _to3ch(x):
        # x in [-1, 1], (B, 1, H, W) -> (B, 3, H, W) tiled.
        return x.expand(-1, 3, -1, -1)

    @torch.no_grad()
    def _compute_paper_metrics(self, pred, target):
        """
        Pixel-space LPIPS/MAE/SSIM/PSNR matching the paper's evaluation.
        Inputs are float tensors in [-1, 1], shape (B, 1, H, W).
        """
        from skimage.metrics import structural_similarity as _ssim
        from skimage.metrics import peak_signal_noise_ratio as _psnr

        pred = pred.detach().clamp_(-1.0, 1.0)
        target = target.detach().clamp_(-1.0, 1.0)
        device = pred.device

        # Map to [0, 1] for SSIM/PSNR (interpretable as image intensity).
        pred01 = pred.mul(0.5).add(0.5)
        tgt01 = target.mul(0.5).add(0.5)

        mae = (pred - target).abs().mean().item()

        # LPIPS expects 3ch images in [-1, 1].
        lpips_model = self._get_lpips(device)
        lpips_val = lpips_model(self._to3ch(pred), self._to3ch(target)).mean().item()

        # SSIM / PSNR per-image then averaged.
        pred_np = pred01.squeeze(1).cpu().numpy()
        tgt_np = tgt01.squeeze(1).cpu().numpy()
        ssim_vals, psnr_vals = [], []
        for p, t in zip(pred_np, tgt_np):
            ssim_vals.append(_ssim(p, t, data_range=1.0))
            try:
                psnr_vals.append(_psnr(t, p, data_range=1.0))
            except Exception:
                psnr_vals.append(float('nan'))
        ssim = float(np.nanmean(ssim_vals)) if ssim_vals else float('nan')
        psnr = float(np.nanmean(psnr_vals)) if psnr_vals else float('nan')
        return {'lpips': lpips_val, 'mae': mae, 'ssim': ssim, 'psnr': psnr}

    @torch.no_grad()
    def _evaluate_paper_metrics(self, val_loader, epoch):
        """
        Runs the full BB reverse process on up to _metrics_eval_max_batches val batches
        and reports LPIPS/MAE/SSIM/PSNR averaged across them. Logs scalars to TB+W&B.
        """
        net = self.net.module if self.config.training.use_DDP else self.net
        net.eval()
        device = self.config.training.device[0]
        agg = {'lpips': [], 'mae': [], 'ssim': [], 'psnr': []}
        n = 0
        for batch in val_loader:
            if n >= self._metrics_eval_max_batches:
                break
            (x, x_name), (x_cond, x_cond_name) = batch
            x = x.to(device)
            x_cond = x_cond.to(device)
            try:
                sample, _ = net.sample(x_cond, x_name, 'val', clip_denoised=False)
            except Exception as e:
                print(f'[metrics] sampling failed at epoch {epoch}: {e!r}')
                return None
            m = self._compute_paper_metrics(sample, x)
            for k, v in m.items():
                if not (isinstance(v, float) and math.isnan(v)):
                    agg[k].append(v)
            n += 1
        if n == 0 or not agg['mae']:
            return None
        means = {k: float(np.mean(v)) if v else float('nan') for k, v in agg.items()}
        for k, v in means.items():
            self.writer.add_scalar(f'metrics/{k}', v, epoch)
        print(f'[metrics] epoch={epoch} ' + ' '.join(f'{k}={v:.4f}' for k, v in means.items()))
        return means

    # --- early-stopping integration -------------------------------------------------------

    @torch.no_grad()
    def validation_epoch(self, val_loader, epoch):
        average_loss = super().validation_epoch(val_loader, epoch)
        try:
            avg_l = float(average_loss.detach()) if torch.is_tensor(average_loss) else float(average_loss)
        except Exception:
            avg_l = None

        # Paper metrics on a small fixed val subset, every epoch.
        means = self._evaluate_paper_metrics(val_loader, epoch)
        self._save_metric_keyed_checkpoints(means, epoch)

        if self._es_enabled and avg_l is not None and math.isfinite(avg_l):
            if avg_l < self._best_val_loss - self._es_min_delta:
                self._best_val_loss = avg_l
                self._patience_counter = 0
            else:
                self._patience_counter += 1
                print(f'[early-stop] epoch={epoch} no improvement '
                      f'({avg_l:.5f} vs best {self._best_val_loss:.5f}); '
                      f'patience {self._patience_counter}/{self._es_patience}')
            if self._patience_counter >= self._es_patience:
                # Force-break the outer epoch loop in BaseRunner.train by tightening n_steps.
                self.config.training.n_steps = self.global_step - 1
                self._stop_signaled = True
                print(f'[early-stop] triggered at epoch {epoch} '
                      f'(best val loss {self._best_val_loss:.5f}); '
                      f'capping n_steps={self.config.training.n_steps}')
        return average_loss

    @torch.no_grad()
    def _save_metric_keyed_checkpoints(self, means, epoch):
        """
        Save best-by-SSIM and best-by-LPIPS model checkpoints, using the metrics already
        computed this validation epoch (no extra sampling). Only the model weights are written
        (optimizer/scheduler state is not needed for evaluation), and each file is overwritten
        in place so at most two extra checkpoints exist. Rank-0 only, mirroring the other saves.
        """
        if means is None:
            return
        if self.config.training.use_DDP and self.config.training.local_rank != 0:
            return
        ckpt_dir = self.config.result.ckpt_path
        ssim = means.get('ssim')
        lpips = means.get('lpips')

        def _save(tag):
            model_states, _ = self.get_checkpoint_states(stage='epoch_end')
            torch.save(model_states, os.path.join(ckpt_dir, f'best_{tag}_model.pth'))

        if ssim is not None and math.isfinite(ssim) and ssim > self._best_metric_ssim:
            self._best_metric_ssim = ssim
            _save('ssim')
            print(f'[metric-ckpt] epoch={epoch} new best SSIM={ssim:.4f} -> best_ssim_model.pth', flush=True)
        if lpips is not None and math.isfinite(lpips) and lpips < self._best_metric_lpips:
            self._best_metric_lpips = lpips
            _save('lpips')
            print(f'[metric-ckpt] epoch={epoch} new best LPIPS={lpips:.4f} -> best_lpips_model.pth', flush=True)

    def initialize_model(self, config):
        if config.model.model_type == "BBDM":
            bbdmnet = BrownianBridgeModel(config.model).to(config.training.device[0])
        elif config.model.model_type == "LBBDM":
            bbdmnet = LatentBrownianBridgeModel(config.model).to(config.training.device[0])
        elif config.model.model_type == "CPDM":
            bbdmnet = CT2PETDiffusionModel(config.model).to(config.training.device[0])
        else:
            raise NotImplementedError
        bbdmnet.apply(weights_init)
        return bbdmnet

    def load_model_from_checkpoint(self):
        states = None
        if self.config.model.only_load_latent_mean_std:
            if self.config.model.__contains__('model_load_path') and self.config.model.model_load_path is not None:
                states = torch.load(self.config.model.model_load_path, map_location='cpu')
        else:
            states = super().load_model_from_checkpoint()

        if self.config.model.normalize_latent:
            if states is not None:
                self.net.ori_latent_mean = states['ori_latent_mean'].to(self.config.training.device[0])
                self.net.ori_latent_std = states['ori_latent_std'].to(self.config.training.device[0])
                self.net.cond_latent_mean = states['cond_latent_mean'].to(self.config.training.device[0])
                self.net.cond_latent_std = states['cond_latent_std'].to(self.config.training.device[0])
            else:
                if self.config.args.train:
                    self.get_latent_mean_std()

    def print_model_summary(self, net):
        def get_parameter_number(model):
            total_num = sum(p.numel() for p in model.parameters())
            trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
            return total_num, trainable_num

        total_num, trainable_num = get_parameter_number(net)
        print("Total Number of parameter: %.2fM" % (total_num / 1e6))
        print("Trainable Number of parameter: %.2fM" % (trainable_num / 1e6))

    def initialize_optimizer_scheduler(self, net, config):
        optimizer = get_optimizer(config.model.BB.optimizer, net.get_parameters())
        # `verbose=` was removed from ReduceLROnPlateau in PyTorch 2.x.
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                               mode='min',
                                                               threshold_mode='rel',
                                                               **vars(config.model.BB.lr_scheduler))
        return [optimizer], [scheduler]

    @torch.no_grad()
    def get_checkpoint_states(self, stage='epoch_end'):
        model_states, optimizer_scheduler_states = super().get_checkpoint_states()
        if self.config.model.normalize_latent:
            if self.config.training.use_DDP:
                model_states['ori_latent_mean'] = self.net.module.ori_latent_mean
                model_states['ori_latent_std'] = self.net.module.ori_latent_std
                model_states['cond_latent_mean'] = self.net.module.cond_latent_mean
                model_states['cond_latent_std'] = self.net.module.cond_latent_std
            else:
                model_states['ori_latent_mean'] = self.net.ori_latent_mean
                model_states['ori_latent_std'] = self.net.ori_latent_std
                model_states['cond_latent_mean'] = self.net.cond_latent_mean
                model_states['cond_latent_std'] = self.net.cond_latent_std
        return model_states, optimizer_scheduler_states

    def get_latent_mean_std(self):
        train_dataset, val_dataset, test_dataset = get_dataset(self.config.data)
        train_loader = DataLoader(train_dataset,
                                  batch_size=self.config.data.train.batch_size,
                                  shuffle=True,
                                  num_workers=24,
                                  drop_last=True)

        total_ori_mean = None
        total_ori_var = None
        total_cond_mean = None
        total_cond_var = None
        max_batch_num = 30000 // self.config.data.train.batch_size

        def calc_mean(batch, total_ori_mean=None, total_cond_mean=None):
            (x, x_name), (x_cond, x_cond_name) = batch
            x = x.to(self.config.training.device[0])
            x_cond = x_cond.to(self.config.training.device[0])

            x_latent = self.net.encode(x, cond=False, normalize=False)
            x_cond_latent = self.net.encode(x_cond, cond=True, normalize=False)
            x_mean = x_latent.mean(axis=[0, 2, 3], keepdim=True)
            total_ori_mean = x_mean if total_ori_mean is None else x_mean + total_ori_mean

            x_cond_mean = x_cond_latent.mean(axis=[0, 2, 3], keepdim=True)
            total_cond_mean = x_cond_mean if total_cond_mean is None else x_cond_mean + total_cond_mean
            return total_ori_mean, total_cond_mean

        def calc_var(batch, ori_latent_mean=None, cond_latent_mean=None, total_ori_var=None, total_cond_var=None):
            (x, x_name), (x_cond, x_cond_name) = batch
            x = x.to(self.config.training.device[0])
            x_cond = x_cond.to(self.config.training.device[0])

            x_latent = self.net.encode(x, cond=False, normalize=False)
            x_cond_latent = self.net.encode(x_cond, cond=True, normalize=False)
            x_var = ((x_latent - ori_latent_mean) ** 2).mean(axis=[0, 2, 3], keepdim=True)
            total_ori_var = x_var if total_ori_var is None else x_var + total_ori_var

            x_cond_var = ((x_cond_latent - cond_latent_mean) ** 2).mean(axis=[0, 2, 3], keepdim=True)
            total_cond_var = x_cond_var if total_cond_var is None else x_cond_var + total_cond_var
            return total_ori_var, total_cond_var

        print(f"start calculating latent mean")
        batch_count = 0
        for train_batch in tqdm(train_loader, total=len(train_loader), smoothing=0.01):
            # if batch_count >= max_batch_num:
            #     break
            batch_count += 1
            total_ori_mean, total_cond_mean = calc_mean(train_batch, total_ori_mean, total_cond_mean)

        ori_latent_mean = total_ori_mean / batch_count
        self.net.ori_latent_mean = ori_latent_mean

        cond_latent_mean = total_cond_mean / batch_count
        self.net.cond_latent_mean = cond_latent_mean

        print(f"start calculating latent std")
        batch_count = 0
        for train_batch in tqdm(train_loader, total=len(train_loader), smoothing=0.01):
            # if batch_count >= max_batch_num:
            #     break
            batch_count += 1
            total_ori_var, total_cond_var = calc_var(train_batch,
                                                     ori_latent_mean=ori_latent_mean,
                                                     cond_latent_mean=cond_latent_mean,
                                                     total_ori_var=total_ori_var,
                                                     total_cond_var=total_cond_var)
            # break

        ori_latent_var = total_ori_var / batch_count
        cond_latent_var = total_cond_var / batch_count

        self.net.ori_latent_std = torch.sqrt(ori_latent_var)
        self.net.cond_latent_std = torch.sqrt(cond_latent_var)
        print(self.net.ori_latent_mean)
        print(self.net.ori_latent_std)
        print(self.net.cond_latent_mean)
        print(self.net.cond_latent_std)

    def loss_fn(self, net, batch, epoch, step, opt_idx=0, stage='train', write=True):
        (x, x_name), (x_cond, x_cond_name) = batch  # PET, CT 
        x = x.to(self.config.training.device[0])
        x_cond = x_cond.to(self.config.training.device[0])

        loss, additional_info = net(x, x_name, x_cond, stage)
        if write:
            self.writer.add_scalar(f'loss/{stage}', loss, step)
            if additional_info.__contains__('recloss_noise'):
                self.writer.add_scalar(f'recloss_noise/{stage}', additional_info['recloss_noise'], step)
            if additional_info.__contains__('recloss_xy'):
                self.writer.add_scalar(f'recloss_xy/{stage}', additional_info['recloss_xy'], step)
        return loss

    @torch.no_grad()
    def sample(self, net, batch, sample_path, stage='train'):
        sample_path = make_dir(os.path.join(sample_path, f'{stage}_sample'))
        reverse_sample_path = make_dir(os.path.join(sample_path, 'reverse_sample'))
        reverse_one_step_path = make_dir(os.path.join(sample_path, 'reverse_one_step_samples'))

        (x, x_name), (x_cond, x_cond_name) = batch

        batch_size = x.shape[0] if x.shape[0] < 4 else 4

        x = x[0:batch_size].to(self.config.training.device[0])
        x_cond = x_cond[0:batch_size].to(self.config.training.device[0])

        grid_size = 4

        sample, add_cond = net.sample(x_cond, x_name, stage, clip_denoised=self.config.testing.clip_denoised)
        sample.to('cpu')
        add_cond.to('cpu')
        image_grid = get_image_grid(sample, grid_size, to_normal=self.config.data.dataset_config.to_normal)
        im = Image.fromarray(image_grid)
        im.save(os.path.join(sample_path, 'skip_sample.png'))
        if stage != 'test':
            self.writer.add_image(f'{stage}_skip_sample', image_grid, self.global_step, dataformats='HWC')

        image_grid = get_image_grid(x_cond.to('cpu'), grid_size, to_normal=self.config.data.dataset_config.to_normal)
        im = Image.fromarray(image_grid)
        im.save(os.path.join(sample_path, 'condition.png'))
        if stage != 'test':
            self.writer.add_image(f'{stage}_condition', image_grid, self.global_step, dataformats='HWC')

        image_grid = get_image_grid(x.to('cpu'), grid_size, to_normal=self.config.data.dataset_config.to_normal)
        im = Image.fromarray(image_grid)
        im.save(os.path.join(sample_path, 'ground_truth.png'))
        if stage != 'test':
            self.writer.add_image(f'{stage}_ground_truth', image_grid, self.global_step, dataformats='HWC')
        
        segment_map = add_cond[:, 0].unsqueeze(1)
        image_grid = get_image_grid(segment_map, grid_size, to_normal=False)
        im = Image.fromarray(image_grid)
        im.save(os.path.join(sample_path, 'segmentation_map.png'))
        
        att_map = add_cond[:, 1].unsqueeze(1)
        att_map = ((att_map - att_map.min()) / (att_map.max() - att_map.min())).clamp_(0, 1.)
        image_grid = get_image_grid(att_map, grid_size, to_normal=False)
        im = Image.fromarray(image_grid)
        im.save(os.path.join(sample_path, 'attenuation_map.png'))

    @torch.no_grad()
    def sample_to_eval(self, net, test_loader, sample_path):
        add_condition_path = make_dir(os.path.join(sample_path, f'additional_condition'))
        condition_path = make_dir(os.path.join(sample_path, f'condition'))
        gt_path = make_dir(os.path.join(sample_path, 'ground_truth'))
        result_path = make_dir(os.path.join(sample_path, str(self.config.model.BB.params.sample_step)))

        pbar = tqdm(test_loader, total=len(test_loader), smoothing=0.01)
        batch_size = self.config.data.test.batch_size
        to_normal = self.config.data.dataset_config.to_normal
        sample_num = self.config.testing.sample_num
        for test_batch in pbar:
            (x, x_name), (x_cond, x_cond_name) = test_batch
            x = x.to(self.config.training.device[0])
            x_cond = x_cond.to(self.config.training.device[0])

            for j in range(sample_num):
                sample, add_cond = net.sample(x_cond, x_name, 'val_step', clip_denoised=False)
                for i in range(batch_size):
                    condition = x_cond[i].detach().clone()
                    gt = x[i]
                    result = sample[i]
                    if j == 0:
                        save_single_image(add_cond[i], add_condition_path, f'{x_name[i]}.npy', max_pixel=1, to_normal=False)
                        save_single_image(condition, condition_path, f'{x_cond_name[i]}.npy', max_pixel=self.config.data.dataset_config.max_pixel_cond, to_normal=False)
                        save_single_image(gt, gt_path, f'{x_name[i]}.npy', max_pixel=self.config.data.dataset_config.max_pixel_ori, to_normal=True)
                    if sample_num > 1:
                        result_path_i = make_dir(os.path.join(result_path, x_name[i]))
                        save_single_image(result, result_path_i, f'output_{j}.npy', max_pixel=self.config.data.dataset_config.max_pixel_ori, to_normal=True)
                    else:
                        save_single_image(result, result_path, f'{x_name[i]}.npy', max_pixel=self.config.data.dataset_config.max_pixel_ori, to_normal=True)
