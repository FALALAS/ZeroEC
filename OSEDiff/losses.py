from torch import nn
from pytorch_msssim import ssim


class SSIMLoss(nn.Module):
    def __init__(self, loss_weight=1.0, data_range=1.0, size_average=True):
        super().__init__()
        self.loss_weight = loss_weight
        self.data_range = data_range
        self.size_average = size_average

    def forward(self, x, y):
        return self.loss_weight * (
            1 - ssim(x, y, data_range=self.data_range, size_average=self.size_average)
        )
