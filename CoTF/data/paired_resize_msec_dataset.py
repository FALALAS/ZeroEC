import cv2
import os
import random
import torch
from torch.utils import data as data
from torchvision.transforms.functional import normalize

from basicsr.utils import FileClient, imfrombytes, img2tensor, scandir
from basicsr.utils.registry import DATASET_REGISTRY

from .cam import CameraModel01


@DATASET_REGISTRY.register()
class PairedOneToManyBilateralDataset(data.Dataset):
    """
    Train:
      - read GT
      - resize to gt_size (no augment)
      - sample params (e_val, gamma, beta)
      - generate lq and the opposite-exposure target on the fly
      - return degradation parameters for training-side re-degradation

    Val/Test:
      - read real lq/gt pairs
    """

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.file_client = None

        self.io_backend_opt = opt["io_backend"]
        self.mean = opt.get("mean", None)
        self.std = opt.get("std", None)
        self.phase = opt["phase"]

        self.gt_folder = opt["dataroot_gt"]
        self.lq_folder = opt.get("dataroot_lq", None)
        self.noise_std = float(opt.get("noise_std", 0.0))

        self.paths = []
        if self.phase == "train":
            if "gt_size" not in opt:
                raise KeyError("'gt_size' is required for train phase.")
            self.gt_size = opt["gt_size"]

            self.paths = list(scandir(self.gt_folder, full_path=True))
            self.camera_model = CameraModel01()
            self.camera_model.eval()
            print("[TRAIN] Generate LQ and real_exposed on the fly.")

        else:
            lq_paths = list(scandir(self.lq_folder))
            gt_paths = list(scandir(self.gt_folder))
            for lq_name in lq_paths:
                id5 = lq_name[:5]
                gt_name = [g for g in gt_paths if g[:5] == id5][0]
                self.paths.append({
                    "lq_path": os.path.join(self.lq_folder, lq_name),
                    "gt_path": os.path.join(self.gt_folder, gt_name),
                })

    def _sample_params(self):
        e_val = random.uniform(-2.0, 2.0)
        gamma = random.gauss(0.9, 0.1)
        beta = random.gauss(0.6, 0.1)
        return e_val, gamma, beta

    @torch.no_grad()
    def _forward_degrade_and_reverse(self, x_0_1, e_val, gamma, beta):
        """
        if e<0:
          x_src = 1 - cam(1-x, e_abs)
          real  = cam(x, e_abs)
        else:
          x_src = cam(x, e_abs)
          real  = 1 - cam(1-x, e_abs)
        """
        device = x_0_1.device
        dtype = x_0_1.dtype

        e_abs = abs(e_val)
        e_t = torch.tensor([[[[e_abs]]]], dtype=dtype, device=device)  # (1,1,1,1)
        gamma_t = torch.tensor(gamma, dtype=dtype, device=device).view(1, 1, 1, 1)
        beta_t = torch.tensor(beta, dtype=dtype, device=device).view(1, 1, 1, 1)

        if e_val < 0:
            ldr_tmp = self.camera_model(1.0 - x_0_1, e_t, gamma_t, beta_t)
            x_src = 1.0 - ldr_tmp
            real_exposed = self.camera_model(x_0_1, e_t, gamma_t, beta_t)
        else:
            x_src = self.camera_model(x_0_1, e_t, gamma_t, beta_t)
            tmp = self.camera_model(1.0 - x_0_1, e_t, gamma_t, beta_t)
            real_exposed = 1.0 - tmp

        return x_src.clamp(0, 1), real_exposed.clamp(0, 1)

    def __getitem__(self, index):
        if self.file_client is None:
            io_opt = dict(self.io_backend_opt)
            self.file_client = FileClient(io_opt.pop("type"), **io_opt)

        if self.phase == "train":
            gt_path = self.paths[index]
            img_bytes = self.file_client.get(gt_path, "gt")
            img_gt = imfrombytes(img_bytes, float32=True)

            img_gt = cv2.resize(img_gt, (self.gt_size, self.gt_size), interpolation=cv2.INTER_AREA)

            gt = img2tensor(img_gt, bgr2rgb=True, float32=True)
            x_tgt = gt.unsqueeze(0)

            e_val, gamma, beta = self._sample_params()
            x_src, real_exposed = self._forward_degrade_and_reverse(x_tgt, e_val, gamma, beta)

            lq = x_src.squeeze(0)
            real_exposed = real_exposed.squeeze(0)

            if self.noise_std > 0:
                noise = torch.randn_like(lq) * self.noise_std
                lq = (lq + noise).clamp(0, 1)

            if self.mean is not None or self.std is not None:
                normalize(lq, self.mean, self.std, inplace=True)
                normalize(gt, self.mean, self.std, inplace=True)
                normalize(real_exposed, self.mean, self.std, inplace=True)

            deg_params = {
                "e_val": torch.tensor(e_val, dtype=torch.float32),
                "gamma": torch.tensor(gamma, dtype=torch.float32),
                "beta": torch.tensor(beta, dtype=torch.float32),
            }

            return {
                "lq": lq,
                "gt": gt,
                "real_exposed": real_exposed,
                "deg_params": deg_params,
                "lq_path": gt_path,
                "gt_path": gt_path,
            }

        else:
            lq_path = self.paths[index]["lq_path"]
            gt_path = self.paths[index]["gt_path"]

            img_bytes = self.file_client.get(lq_path, "lq")
            img_lq = imfrombytes(img_bytes, float32=True)
            img_bytes = self.file_client.get(gt_path, "gt")
            img_gt = imfrombytes(img_bytes, float32=True)

            lq, gt = img2tensor([img_lq, img_gt], bgr2rgb=True, float32=True)

            if self.mean is not None or self.std is not None:
                normalize(lq, self.mean, self.std, inplace=True)
                normalize(gt, self.mean, self.std, inplace=True)

            return {
                "lq": lq,
                "gt": gt,
                "lq_path": lq_path,
                "gt_path": gt_path,
            }

    def __len__(self):
        return len(self.paths)
