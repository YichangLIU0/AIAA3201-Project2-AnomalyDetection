"""Compact normal-only AE; no reference bank and no pretrained parameters."""
import numpy as np
import cv2
import torch
from torch import nn
from torch.nn import functional as F

from .template import preprocess, image_score, read_rgb

CONFIG = {
    'method': 'compact_normal_ae', 'experiment_mode': 'normal_training',
    'training_pool': 'entire manifest normal_training_pool; excludes all 20 calibration images',
    'seeds': [0, 1, 2], 'epochs': 12, 'batch_size': 8, 'learning_rate': .001,
    'optimizer': 'Adam', 'loss': 'mean RGB squared reconstruction error',
    'architecture': 'four stride2 Conv3x3 encoder blocks 16/32/48/64; nearest-upsample Conv3x3 decoder 48/32/16/3; ReLU, final sigmoid',
    'augmentation': 'independent horizontal and vertical flips p=.5 per training image; no synthetic defects',
    'input_geometry': 'same short-edge448 bicubic and bottom/right crop to14 as Template; reflection pad bottom/right to16 for AE, crop reconstruction back',
    'map': 'mean absolute RGB reconstruction error, Gaussian sigma2 on input grid; no morphology',
    'image_score': 'mean largest floor(1% of unpadded input-map pixels), at least1',
    'native_map': 'OpenCV linear resize to original size; original GT unchanged',
    'threshold': 'linear 95th percentile of exactly20 held-out normal calibration scores',
    'selection': 'fixed architecture/12 final epochs a priori; no test-label selection, no calibration fitting or early stopping',
    'precision': 'float32; no AMP; native maps stored losslessly as float32 compressed NPZ',
    'determinism': 'fixed seeds; cudnn deterministic true; benchmark false; torch threads4',
    'comparison_scope': 'normal-training extension, not a K-shot experiment; training counts reported separately',
}


class CompactAE(nn.Module):
    def __init__(self):
        super().__init__()
        enc, incoming = [], 3
        for channels in [16, 32, 48, 64]:
            enc += [nn.Conv2d(incoming, channels, 3, 2, 1), nn.ReLU()]
            incoming = channels
        self.encoder = nn.Sequential(*enc)
        self.decoder = nn.ModuleList([nn.Conv2d(a, b, 3, 1, 1)
                                     for a, b in [(64,48), (48,32), (32,16), (16,3)]])

    def forward(self, x):
        height, width = x.shape[-2:]
        x = F.pad(x, (0, -width % 16, 0, -height % 16), mode='reflect')
        x = self.encoder(x)
        for i, layer in enumerate(self.decoder):
            x = layer(F.interpolate(x, scale_factor=2, mode='nearest'))
            x = torch.sigmoid(x) if i == 3 else F.relu(x)
        return x[..., :height, :width]


def tensor_image(rgb, device):
    return torch.from_numpy(np.ascontiguousarray(rgb.transpose(2,0,1))).to(device, dtype=torch.float32)[None] / 255.


@torch.inference_mode()
def predict(model, path, device='cuda'):
    original = read_rgb(path)
    image = preprocess(original)
    x = tensor_image(image, device)
    reconstruction = model(x)
    error = (x - reconstruction).abs().mean(1)[0].cpu().numpy()
    error = cv2.GaussianBlur(error, (0,0), 2., borderType=cv2.BORDER_REFLECT_101)
    native = cv2.resize(error, (original.shape[1], original.shape[0]), interpolation=cv2.INTER_LINEAR)
    return image_score(error), native.astype(np.float32), list(image.shape[:2])
