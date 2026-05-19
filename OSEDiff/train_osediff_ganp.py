import argparse
import os
from pathlib import Path

import diffusers
import lpips
import torch
import torch.nn.functional as F
import transformers
import vision_aided_loss
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers.optimization import get_scheduler
from diffusers.utils.import_utils import is_xformers_available
from tqdm.auto import tqdm

from camera import CameraModel01 as CameraModel
from dataloaders.paired_dataset import PairedCaptionDataset
from losses import SSIMLoss
from osediff import OSEDiff_gen


DEFAULT_NEG_PROMPT = (
    "painting, oil painting, illustration, drawing, art, sketch, cartoon, "
    "CG Style, 3D render, unreal engine, blurring, dirty, messy, worst quality, "
    "low quality, frames, watermark, signature, jpeg artifacts, deformed, "
    "lowres, over-smooth"
)


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(
        description="Train OSEDiff-GANP from paired GT images and exposure prompts."
    )

    parser.add_argument(
        "--root_folders",
        type=str,
        required=True,
        help=(
            "Comma-separated dataset roots. Each root must contain gt/ images and "
            "tag/ prompt files named <image_stem>_+2.txt and <image_stem>_-2.txt."
        ),
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        required=True,
        help="Path or Hugging Face id for Stable Diffusion 2.1-base compatible weights.",
    )
    parser.add_argument("--output_dir", default="outputs/osediff_ganp")
    parser.add_argument("--logging_dir", type=str, default="logs")
    parser.add_argument("--tracker_project_name", type=str, default="train_osediff_ganp")

    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--num_training_epochs", type=int, default=10000)
    parser.add_argument("--max_train_steps", type=int, default=100000)
    parser.add_argument("--checkpointing_steps", type=int, default=1000)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--dataloader_num_workers", type=int, default=0)

    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        choices=[
            "linear",
            "cosine",
            "cosine_with_restarts",
            "polynomial",
            "constant",
            "constant_with_warmup",
        ],
    )
    parser.add_argument("--lr_warmup_steps", type=int, default=500)
    parser.add_argument("--lr_num_cycles", type=int, default=1)
    parser.add_argument("--lr_power", type=float, default=1.0)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--max_grad_norm", default=1.0, type=float)

    parser.add_argument("--lambda_l2", default=1.0, type=float)
    parser.add_argument("--lambda_lpips", default=2.0, type=float)
    parser.add_argument("--lambda_gan", default=1.0, type=float)
    parser.add_argument("--lora_rank", default=4, type=int)
    parser.add_argument("--neg_prompt", default=DEFAULT_NEG_PROMPT, type=str)

    parser.add_argument("--allow_tf32", action="store_true")
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="fp16",
        choices=["no", "fp16", "bf16"],
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help='Tracker backend supported by accelerate, for example "tensorboard" or "wandb".',
    )
    parser.add_argument("--enable_xformers_memory_efficient_attention", action="store_true")
    parser.add_argument("--set_grads_to_none", action="store_true")

    return parser.parse_args(input_args)


def collect_trainable_parameters(model_gen):
    params = []
    for name, param in model_gen.unet.named_parameters():
        if "lora" in name:
            params.append(param)
    params += list(model_gen.unet.conv_in.parameters())
    for name, param in model_gen.vae.named_parameters():
        if "lora" in name:
            params.append(param)
    return params


def main(args):
    logging_dir = Path(args.output_dir, args.logging_dir)
    project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=project_config,
        kwargs_handlers=[ddp_kwargs],
    )

    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        os.makedirs(os.path.join(args.output_dir, "checkpoints"), exist_ok=True)

    model_gen = OSEDiff_gen(args, device=accelerator.device)
    model_gen.set_train()
    model_gen.vae.set_adapter(["default_encoder"])
    model_gen.unet.set_adapter(["default_encoder", "default_decoder", "default_others"])

    if args.enable_xformers_memory_efficient_attention:
        if not is_xformers_available():
            raise ValueError("xformers is not available. Install xformers or remove the flag.")
        model_gen.unet.enable_xformers_memory_efficient_attention()

    if args.gradient_checkpointing:
        model_gen.unet.enable_gradient_checkpointing()

    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    net_lpips = lpips.LPIPS(net="vgg").to(accelerator.device)
    net_lpips.requires_grad_(False)

    ssim_loss_fn = SSIMLoss(loss_weight=1.0, data_range=1.0, size_average=True)
    ssim_loss_fn.to(accelerator.device)

    net_disc_ldr = vision_aided_loss.Discriminator(
        cv_type="clip",
        loss_type="multilevel_sigmoid",
        device=accelerator.device,
    )
    net_disc_ldr.cv_ensemble.requires_grad_(False)

    layers_to_opt = collect_trainable_parameters(model_gen)
    optimizer = torch.optim.AdamW(
        layers_to_opt,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    optimizer_disc = torch.optim.AdamW(
        net_disc_ldr.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    lr_scheduler_disc = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer_disc,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    dataset_train = PairedCaptionDataset(root_folders=args.root_folders, args=args)
    dl_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.dataloader_num_workers,
    )

    (
        model_gen,
        optimizer,
        dl_train,
        lr_scheduler,
        net_disc_ldr,
        optimizer_disc,
        lr_scheduler_disc,
    ) = accelerator.prepare(
        model_gen,
        optimizer,
        dl_train,
        lr_scheduler,
        net_disc_ldr,
        optimizer_disc,
        lr_scheduler_disc,
    )
    net_lpips = accelerator.prepare(net_lpips)
    trainable_params = [param for group in optimizer.param_groups for param in group["params"]]

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    if accelerator.is_main_process:
        accelerator.init_trackers(args.tracker_project_name, config=dict(vars(args)))

    progress_bar = tqdm(
        range(args.max_train_steps),
        initial=0,
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )

    camera_model = CameraModel()
    global_step = 0

    for _epoch in range(args.num_training_epochs):
        for batch in dl_train:
            with accelerator.accumulate(model_gen, net_disc_ldr):
                x_tgt_0_1 = batch["output_pixel_values"]
                x_tgt = (x_tgt_0_1 * 2.0) - 1.0
                batch_size = x_tgt_0_1.size(0)

                exposure_value = float(
                    torch.empty(1, device=accelerator.device, dtype=weight_dtype).uniform_(-2.0, 2.0)
                )
                exposure = torch.full(
                    (batch_size, 1, 1, 1),
                    abs(exposure_value),
                    device=accelerator.device,
                    dtype=weight_dtype,
                )
                gamma = torch.normal(0.9, 0.1, size=exposure.shape, device=exposure.device, dtype=weight_dtype)
                beta = torch.normal(0.6, 0.1, size=exposure.shape, device=exposure.device, dtype=weight_dtype)

                if exposure_value < 0:
                    ldr_tmp = camera_model(1 - x_tgt_0_1, exposure, gamma, beta)
                    x_src_0_1 = 1 - ldr_tmp
                    real_exposed_0_1 = camera_model(x_tgt_0_1, exposure, gamma, beta)
                    batch["prompt"] = batch["prompt-"]
                else:
                    x_src_0_1 = camera_model(x_tgt_0_1, exposure, gamma, beta)
                    real_exposed_0_1 = camera_model(1 - x_tgt_0_1, exposure, gamma, beta)
                    real_exposed_0_1 = 1 - real_exposed_0_1
                    batch["prompt"] = batch["prompt+"]

                x_src = x_src_0_1 * 2.0 - 1.0
                real_exposed_neg1_1 = real_exposed_0_1 * 2.0 - 1.0

                x_tgt_pred, _latents_pred, _prompt_embeds, _neg_prompt_embeds = model_gen(
                    x_src,
                    batch=batch,
                )

                pred_normal_image_0_1 = (x_tgt_pred + 1.0) / 2.0
                if exposure_value < 0:
                    pseudo_exposed_0_1 = camera_model(pred_normal_image_0_1, exposure, gamma, beta)
                else:
                    pred_normal_image_0_1 = 1 - pred_normal_image_0_1
                    pseudo_exposed_0_1 = camera_model(pred_normal_image_0_1, exposure, gamma, beta)
                    pseudo_exposed_0_1 = 1 - pseudo_exposed_0_1
                pseudo_exposed_neg1_1 = pseudo_exposed_0_1 * 2.0 - 1.0

                net_disc_ldr.requires_grad_(False)
                loss_gan = net_disc_ldr(pseudo_exposed_neg1_1, for_G=True).mean() * args.lambda_gan
                loss_ssim = ssim_loss_fn((x_tgt_pred + 1.0) / 2.0, (x_tgt + 1.0) / 2.0)
                loss_l2 = F.mse_loss(x_tgt_pred.float(), x_tgt.float(), reduction="mean") * args.lambda_l2
                loss_lpips = net_lpips(x_tgt_pred.float(), x_tgt.float()).mean() * args.lambda_lpips
                loss = loss_l2 + loss_lpips + loss_gan + loss_ssim

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(trainable_params, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=args.set_grads_to_none)

                net_disc_ldr.requires_grad_(True)
                optimizer_disc.zero_grad()
                loss_d_real = net_disc_ldr(real_exposed_neg1_1.detach(), for_real=True).mean()
                loss_d_fake = net_disc_ldr(pseudo_exposed_neg1_1.detach(), for_real=False).mean()
                loss_d = (loss_d_real + loss_d_fake) * 0.5

                accelerator.backward(loss_d)
                optimizer_disc.step()
                lr_scheduler_disc.step()
                optimizer_disc.zero_grad(set_to_none=args.set_grads_to_none)

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    logs = {
                        "loss_l2": loss_l2.detach().item(),
                        "loss_lpips": loss_lpips.detach().item(),
                        "loss_gan": loss_gan.detach().item(),
                        "loss_ssim": loss_ssim.detach().item(),
                        "loss_d": loss_d.detach().item(),
                    }
                    progress_bar.set_postfix(**logs)

                    if global_step % args.checkpointing_steps == 1:
                        outf = os.path.join(args.output_dir, "checkpoints", f"model_{global_step}.pkl")
                        accelerator.unwrap_model(model_gen).save_model(outf)

                    accelerator.log(logs, step=global_step)

                if global_step >= args.max_train_steps:
                    return


if __name__ == "__main__":
    main(parse_args())
