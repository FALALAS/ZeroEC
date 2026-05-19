import torch
import torch.nn as nn


class CameraModel01(nn.Module):
    """Exposure and camera-response simulation used by CoTF GAN training."""

    def __init__(self):
        super().__init__()
        self.exposure_factor = 1.41421356
        self.epsilon = 1e-7

    def forward(self, hdr_images, exposure, gamma, beta):
        exposed_images = torch.pow(self.exposure_factor, exposure) * hdr_images
        clipped_images = torch.where(
            exposed_images > 1.0,
            torch.ones_like(exposed_images),
            exposed_images,
        )
        crf_input = clipped_images + self.epsilon
        crf_pow = torch.pow(crf_input, gamma)
        return (1 + beta) * crf_pow / (crf_pow + beta)
