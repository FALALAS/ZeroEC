import torch
import torch.nn as nn
import numpy as np
from glob import glob

class CameraModel01(nn.Module):
    """
    模拟相机模型，将 HDR 图像转换为 LDR 图像。
    严格按照用户原始代码版本。
    """

    def __init__(self):
        super().__init__()
        # 曝光计算因子 (约等于 2 的 1/2 次方)
        self.exposure_factor = 1.41421356  # 使用原始值
        self.epsilon = 1e-7  # 避免除以零的小值

    def forward(self, hdr_images, e, gamma, beta):
        """
        Args:
            hdr_images (torch.Tensor): HDR 图像批次，形状 (B, C, H, W)，值范围 [0, 1]。
            e (torch.Tensor): 曝光调整值 (EV stops)，形状 (B, 1, 1, 1)。
            gamma (torch.Tensor): CRF gamma 参数，形状与 e 相同。
            beta (torch.Tensor):  CRF beta 参数，形状与 e 相同。

        Returns:
            torch.Tensor: LDR 图像批次，形状与 hdr_images 相同，值范围 [-1, 1]。
        """
        #hdr_images = 1-hdr_images
        # 1. 曝光调整

        exposed_images = torch.pow(self.exposure_factor, e) * hdr_images

        # 2. 裁剪 (使用原始逻辑：仅裁剪 > 1.0 的部分)
        # 注意：这可能导致负值未被裁剪，如果 hdr_images * factor^e 可能小于0的话 (虽然通常不会)
        clipped_images = torch.where(exposed_images > 1.0, torch.ones_like(exposed_images), exposed_images)
        # 如果需要严格[0,1]裁剪，应使用 torch.clamp(exposed_images, 0.0, 1.0)

        # 3. 相机响应函数 (CRF) 模拟
        crf_input = clipped_images + self.epsilon  # 加上 epsilon 避免 log(0) 或 pow(0, negative)
        crf_pow = torch.pow(crf_input, gamma)
        ldr_images_0_1 = (1 + beta) * crf_pow / (crf_pow + beta)  # 此处范围接近 [0, 1]
        #ldr_images_0_1 = 1 - ldr_images_0_1

        # 4. 缩放到 [-1, 1] 范围 (按照原始代码逻辑)
        #ldr_images_neg1_1 = 2.0 * ldr_images_0_1 - 1.0
        ldr_images_neg1_1 = ldr_images_0_1

        # 注意：这里不再 clamp 到 [0, 1]，因为原始模型最后一步是缩放到 [-1, 1]
        return ldr_images_neg1_1