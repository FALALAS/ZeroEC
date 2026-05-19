from collections import OrderedDict
import torch

from basicsr.archs import build_network
from basicsr.losses import build_loss
from basicsr.models.sr_model import SRModel
from basicsr.utils import get_root_logger
from basicsr.utils.registry import MODEL_REGISTRY

import vision_aided_loss
from .cam import CameraModel01


@MODEL_REGISTRY.register()
class FTModel2(SRModel):

    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt["train"]

        self.ema_decay = train_opt.get("ema_decay", 0)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(f"Use Exponential Moving Average with decay: {self.ema_decay}")
            self.net_g_ema = build_network(self.opt["network_g"]).to(self.device)
            load_path = self.opt["path"].get("pretrain_network_g", None)
            if load_path is not None:
                self.load_network(self.net_g_ema, load_path, self.opt["path"].get("strict_load_g", True), "params_ema")
            else:
                self.model_ema(0)
            self.net_g_ema.eval()

        self.cri_pix = build_loss(train_opt["pixel_opt"]).to(self.device) if train_opt.get("pixel_opt") else None
        self.cri_ssim = build_loss(train_opt["ssim_opt"]).to(self.device) if train_opt.get("ssim_opt") else None
        self.cri_perceptual = build_loss(train_opt["perceptual_opt"]).to(self.device) if train_opt.get("perceptual_opt") else None

        self.camera_model = CameraModel01().to(self.device)
        self.camera_model.eval()

        self.gan_w = float(train_opt.get("gan_loss_weight", 0.0))
        if self.gan_w > 0:
            self.net_disc_ldr = vision_aided_loss.Discriminator(
                cv_type="clip",
                loss_type="multilevel_sigmoid",
                device=self.device,
            )
            self.net_disc_ldr.cv_ensemble.requires_grad_(False)
            self.net_disc_ldr.to(self.device)
            self.net_disc_ldr.train()
        else:
            self.net_disc_ldr = None

        self.setup_optimizers()
        self.setup_schedulers()

    def feed_data(self, data):
        self.lq_path = data["lq_path"]
        self.lq = data["lq"].to(self.device)
        if "gt" in data:
            self.gt = data["gt"].to(self.device)

        self.real_exposed = data.get("real_exposed", None)
        if self.real_exposed is not None:
            self.real_exposed = self.real_exposed.to(self.device)

        self.deg_params = data.get("deg_params", None)

    def setup_optimizers(self):
        train_opt = self.opt["train"]

        optim_params, sampler_params = [], []
        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
                if "sampler" in k:
                    sampler_params.append(v)
                else:
                    optim_params.append(v)
            else:
                get_root_logger().warning(f"Params {k} will not be optimized.")

        optim_type_g = train_opt["optim_g"].pop("type")
        my_groups = [{"params": optim_params}, {"params": sampler_params, "lr": 0.00004}]
        self.optimizer_g = self.get_optimizer(optim_type_g, my_groups, **train_opt["optim_g"])
        self.optimizers.append(self.optimizer_g)

        # optimizer D
        if self.net_disc_ldr is not None:
            if "optim_d" not in train_opt:
                raise KeyError("gan_loss_weight>0 but train.optim_d is missing in yaml.")
            optim_type_d = train_opt["optim_d"].pop("type")
            self.optimizer_d = self.get_optimizer(optim_type_d, self.net_disc_ldr.parameters(), **train_opt["optim_d"])
            self.optimizers.append(self.optimizer_d)
        else:
            self.optimizer_d = None

    def _to_b111(self, x, B, dtype, device):
        if not torch.is_tensor(x):
            x = torch.tensor(x)
        x = x.to(device=device, dtype=dtype)
        if x.dim() == 0:
            x = x.view(1, 1, 1, 1).repeat(B, 1, 1, 1)
        elif x.dim() == 1:
            x = x.view(B, 1, 1, 1)
        else:
            x = x.view(B, 1, 1, 1)
        return x

    def _re_degrade_to_reverse(self, pred_0_1, deg_params):
        """
        if e<0: pseudo = cam(pred)
        else:   pseudo = 1 - cam(1-pred)
        """
        B = pred_0_1.shape[0]
        device = pred_0_1.device
        dtype = pred_0_1.dtype

        e_val = deg_params["e_val"].to(device=device, dtype=torch.float32)
        gamma = deg_params["gamma"].to(device=device, dtype=torch.float32)
        beta = deg_params["beta"].to(device=device, dtype=torch.float32)

        e_abs = self._to_b111(e_val.abs(), B, dtype, device)
        gamma_t = self._to_b111(gamma, B, dtype, device)
        beta_t = self._to_b111(beta, B, dtype, device)

        e_neg = (e_val < 0).view(B, 1, 1, 1).to(device=device)

        out_neg = self.camera_model(pred_0_1, e_abs, gamma_t, beta_t)
        out_pos = 1.0 - self.camera_model(1.0 - pred_0_1, e_abs, gamma_t, beta_t)

        pseudo = torch.where(e_neg, out_neg, out_pos)
        return pseudo.clamp(0, 1)

    def _to_neg1_1(self, x):
        return x * 2.0 - 1.0

    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()
        self.output = self.net_g(self.lq)

        loss_dict = OrderedDict()
        l_total = 0

        if self.cri_pix:
            l_pix = self.cri_pix(self.output, self.gt)
            l_total += l_pix
            loss_dict["l_pix"] = l_pix
        if self.cri_perceptual:
            l_percep, l_style = self.cri_perceptual(self.output, self.gt)
            if l_percep is not None:
                l_total += l_percep
                loss_dict["l_percep"] = l_percep
            if l_style is not None:
                l_total += l_style
                loss_dict["l_style"] = l_style
        if self.cri_ssim:
            l_ssim = self.cri_ssim(self.output, self.gt)
            l_total += l_ssim
            loss_dict["l_ssim"] = l_ssim

        if (self.real_exposed is not None) and (self.deg_params is not None):
            pred_0_1 = self.output.clamp(0, 1)
            pseudo_exposed = self._re_degrade_to_reverse(pred_0_1, self.deg_params)
            if self.net_disc_ldr is not None and self.gan_w > 0:
                self.net_disc_ldr.requires_grad_(False)

                pseudo_n11 = self._to_neg1_1(pseudo_exposed)
                l_g_gan = self.net_disc_ldr(pseudo_n11, for_G=True).mean() * self.gan_w

                l_total += l_g_gan
                loss_dict["l_g_gan"] = l_g_gan

        l_total.backward()
        self.optimizer_g.step()

        if self.net_disc_ldr is not None and self.optimizer_d is not None and self.gan_w > 0:
            if (self.real_exposed is not None) and (self.deg_params is not None):
                self.optimizer_d.zero_grad()
                self.net_disc_ldr.requires_grad_(True)

                with torch.no_grad():
                    pred_0_1_detach = self.output.detach().clamp(0, 1)
                pseudo_exposed_detach = self._re_degrade_to_reverse(pred_0_1_detach, self.deg_params).detach()

                real_n11 = self._to_neg1_1(self.real_exposed)
                fake_n11 = self._to_neg1_1(pseudo_exposed_detach)

                l_d_real = self.net_disc_ldr(real_n11, for_real=True).mean()
                l_d_fake = self.net_disc_ldr(fake_n11, for_real=False).mean()
                l_d = 0.5 * (l_d_real + l_d_fake)

                l_d.backward()
                self.optimizer_d.step()

                loss_dict["l_d_real"] = l_d_real
                loss_dict["l_d_fake"] = l_d_fake
                loss_dict["l_d"] = l_d

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

    def test(self):
        if hasattr(self, "net_g_ema"):
            self.net_g_ema.eval()
            with torch.no_grad():
                self.output = self.net_g_ema(self.lq)
        else:
            self.net_g.eval()
            with torch.no_grad():
                self.output = self.net_g(self.lq)
            self.net_g.train()
