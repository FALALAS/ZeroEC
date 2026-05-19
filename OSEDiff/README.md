# ZeroEC OSEDiff Branch

This is the one-step diffusion instantiation of ZeroEC. It follows the
OSEDiff-style backbone and trains with the modified camera degradation and
hallucination consistency objective described in the paper.

## Contents

- `train_osediff_ganp.py`: main training entrypoint.
- `osediff.py`: OSEDiff generator and LoRA initialization used by training.
- `camera.py`: exposure/camera-response simulation.
- `losses.py`: SSIM loss wrapper.
- `dataloaders/paired_dataset.py`: dataset loader for GT images and prompts.
- `models/`: local VAE and UNet definitions adapted from diffusers.

## Method Match

The training script implements the paper's OSD branch:

- Samples exposure `e` continuously from `[-2, 2]`.
- Samples camera response parameters with `gamma ~ N(0.9, 0.1)` and `beta ~ N(0.6, 0.1)`.
- Uses invert-clip-invert degradation for under-exposure simulation.
- Re-degrades the corrected prediction with the opposite exposure.
- Uses MSE, LPIPS, SSIM, and CLIP-guided hallucination consistency GAN loss.

## Installation

```bash
conda create -n osediff_ganp python=3.10 -y
conda activate osediff_ganp
pip install -r requirements.txt
```

`vision_aided_loss` is required by the GAN discriminator. If your package index
does not provide it, install it from the upstream project used by your training
environment.

`xformers` is only required when using
`--enable_xformers_memory_efficient_attention`; remove that flag if you do not
install xformers.

## Data Format

Pass one or more dataset roots with `--root_folders`. Each root must use this
layout:

```text
dataset_root/
  gt/
    image_001.png
    image_002.png
  tag/
    image_001_+2.txt
    image_001_-2.txt
    image_002_+2.txt
    image_002_-2.txt
```

The `+2` prompt is used for over-exposure simulation and the `-2` prompt is used
for under-exposure simulation.

## Training

```bash
accelerate launch train_osediff_ganp.py \
  --root_folders /path/to/dataset_root \
  --pretrained_model_name_or_path /path/to/stable-diffusion-2-1-base \
  --output_dir outputs/osediff_ganp \
  --learning_rate 5e-5 \
  --train_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --checkpointing_steps 1000 \
  --mixed_precision fp16 \
  --report_to tensorboard \
  --enable_xformers_memory_efficient_attention
```

For multiple datasets, pass comma-separated roots:

```bash
--root_folders /path/to/dataset_a,/path/to/dataset_b
```

Checkpoints are saved under:

```text
outputs/osediff_ganp/checkpoints/model_<step>.pkl
```

## Release Notes

This folder intentionally excludes:

- Training outputs such as `experience/`, logs, result CSV files, and metrics plots.
- Private dataset paths and machine-specific default model paths.
- Pretrained weights and generated checkpoints.
- Inference scripts, ablation scripts, RAM/DAPE assets, and unrelated dataset loaders.

Before publishing, review licensing for the pretrained Stable Diffusion weights
and all third-party dependencies you expect users to download separately.
