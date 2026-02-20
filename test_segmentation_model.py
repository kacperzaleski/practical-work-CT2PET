from datasets.BrainSliceDataset import BrainSliceDataset
import segmentation_models_pytorch as smp
import argparse
import omegaconf 
import os
import torch
import pytorch_lightning as pl
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

CHECKPOINT_PATH = 'results/checkpoints'
DATA_PATH = 'data/processed/brain_slices'
IMAGE_SIZE = 256
CT_MAX = 2047
PET_MAX = 5.0
BATCH_SIZE = 8

class SegmentationModel(pl.LightningModule):
    def __init__(self, arch, encoder_name, in_channels, out_classes, **kwargs):
        super().__init__()
        self.model = smp.create_model(
            arch, encoder_name=encoder_name, in_channels=in_channels, classes=out_classes, **kwargs
        )

        # preprocessing parameteres for image
        params = smp.encoders.get_preprocessing_params(encoder_name)
        self.register_buffer("std", torch.tensor(params["std"]).view(1, 3, 1, 1))
        self.register_buffer("mean", torch.tensor(params["mean"]).view(1, 3, 1, 1))

        # for image segmentation dice loss could be the best first choice
        self.loss_fn = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)
        
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []

    def forward(self, image):
        # normalize image here
        # image = (image - self.mean) / self.std
        mask = self.model(image)
        return mask

    def shared_step(self, batch, stage):
        image = batch[0]

        # Shape of the image should be (batch_size, num_channels, height, width)
        # if you work with grayscale images, expand channels dim to have [batch_size, 1, height, width]
        assert image.ndim == 4

        # Check that image dimensions are divisible by 32, 
        # encoder and decoder connected by `skip connections` and usually encoder have 5 stages of 
        # downsampling by factor 2 (2 ^ 5 = 32); e.g. if we have image with shape 65x65 we will have 
        # following shapes of features in encoder and decoder: 84, 42, 21, 10, 5 -> 5, 10, 20, 40, 80
        # and we will get an error trying to concat these features
        h, w = image.shape[2:]
        assert h % 32 == 0 and w % 32 == 0

        mask = batch[2]

        # Shape of the mask should be [batch_size, num_classes, height, width]
        # for binary segmentation num_classes = 1
        assert mask.ndim == 4

        # Check that mask values in between 0 and 1, NOT 0 and 255 for binary segmentation
        assert mask.max() <= 1.0 and mask.min() >= 0

        logits_mask = self.forward(image)
        
        # Predicted mask contains logits, and loss_fn param `from_logits` is set to True
        loss = self.loss_fn(logits_mask, mask)

        # Lets compute metrics for some threshold
        # first convert mask values to probabilities, then 
        # apply thresholding
        prob_mask = logits_mask.sigmoid()
        pred_mask = (prob_mask > 0.5).float()

        # We will compute IoU metric by two ways
        #   1. dataset-wise
        #   2. image-wise
        # but for now we just compute true positive, false positive, false negative and
        # true negative 'pixels' for each image and class
        # these values will be aggregated in the end of an epoch
        tp, fp, fn, tn = smp.metrics.get_stats(pred_mask.long(), mask.long(), mode="binary")

        return {
            "loss": loss,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        }

    def shared_epoch_end(self, outputs, stage):
        # aggregate step metics
        tp = torch.cat([x["tp"] for x in outputs])
        fp = torch.cat([x["fp"] for x in outputs])
        fn = torch.cat([x["fn"] for x in outputs])
        tn = torch.cat([x["tn"] for x in outputs])

        # per image IoU means that we first calculate IoU score for each image 
        # and then compute mean over these scores
        per_image_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro-imagewise")
        
        # dataset IoU means that we aggregate intersection and union over whole dataset
        # and then compute IoU score. The difference between dataset_iou and per_image_iou scores
        # in this particular case will not be much, however for dataset 
        # with "empty" images (images without target class) a large gap could be observed. 
        # Empty images influence a lot on per_image_iou and much less on dataset_iou.
        dataset_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")

        metrics = {
            f"{stage}_per_image_iou": per_image_iou,
            f"{stage}_dataset_iou": dataset_iou,
        }
        
        self.log_dict(metrics, prog_bar=True)

    def training_step(self, batch, batch_idx):
        res = self.shared_step(batch, "train")
        self.training_step_outputs.append(res)
        return res

    def on_train_epoch_end(self):
        self.shared_epoch_end(self.training_step_outputs, "train")
        self.training_step_outputs.clear()

    def validation_step(self, batch, batch_idx):
        res = self.shared_step(batch, "valid")
        self.validation_step_outputs.append(res)
        return res

    def on_validation_epoch_end(self):
        self.shared_epoch_end(self.validation_step_outputs, "valid")
        self.validation_step_outputs.clear()

    def test_step(self, batch, batch_idx):
        res = self.shared_step(batch, "test")
        self.test_step_outputs.append(res)
        return res

    def on_test_epoch_end(self):
        self.shared_epoch_end(self.test_step_outputs, "test")
        self.test_step_outputs.clear()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.0001)

def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict) or isinstance(value, omegaconf.dictconfig.DictConfig):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace

def get_image_paths_from_dir(fdir):
    flist = os.listdir(fdir)
    flist.sort()
    image_paths = []
    for i in range(0, len(flist)):
        fpath = os.path.join(fdir, flist[i])
        if os.path.isdir(fpath):
            image_paths.extend(get_image_paths_from_dir(fpath))
        else:
            image_paths.append(fpath)
    return image_paths

def get_dataset_by_stage(data_path, stage, image_size, ct_max_pixel, pet_max_pixel, flip):
    ct_paths = get_image_paths_from_dir(os.path.join(data_path, f'{stage}/A'))
    pet_paths = get_image_paths_from_dir(os.path.join(data_path, f'{stage}/B'))

    return SegmentedPETCTDataset(ct_paths, pet_paths, image_size, ct_max_pixel, pet_max_pixel, flip)
    
def main():
    train_dataset = BrainSliceDataset(DATA_PATH, split='train', image_size=(IMAGE_SIZE, IMAGE_SIZE), ct_max=CT_MAX, pet_max=PET_MAX, flip=False)
    val_dataset = BrainSliceDataset(DATA_PATH, split='val', image_size=(IMAGE_SIZE, IMAGE_SIZE), ct_max=CT_MAX, pet_max=PET_MAX, flip=False)
    test_dataset = BrainSliceDataset(DATA_PATH, split='test', image_size=(IMAGE_SIZE, IMAGE_SIZE), ct_max=CT_MAX, pet_max=PET_MAX, flip=False)

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    valid_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    CHECKPOINT_FILE_PATH = os.path.join(CHECKPOINT_PATH, "Unet/best.ckpt")
    model_name = "Unet"
    encoder_name = "resnet34"
    # Note: ensure weights are loaded correctly according to PL 2.x if needed
    model = SegmentationModel.load_from_checkpoint(CHECKPOINT_FILE_PATH, arch=model_name, encoder_name=encoder_name, in_channels=1, out_classes=1)
    model.eval()
    
    SAVE_PATH = os.path.join(CHECKPOINT_PATH, "Unet/samples")
    
    with torch.no_grad():
        # for batch in tqdm(train_dataloader):
        # for batch in tqdm(valid_dataloader):
        for batch in tqdm(test_dataloader):
            logits = model(batch[0])
            
            pr_masks = logits.sigmoid()
            
            n = pr_masks.shape[0]

            for i in range(n):
                directory = os.path.dirname(SAVE_PATH + f"/{batch[3][i]}.npy")
                
                if not os.path.exists(directory):
                    os.makedirs(directory)
                    
                np.save(SAVE_PATH + f"/{batch[3][i]}.npy", pr_masks[i].cpu().numpy().squeeze())
    
if __name__ == '__main__':
    main()