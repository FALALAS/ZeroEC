# ZeroEC CoTF Branch

This is the lightweight CoTF/LUT instantiation of ZeroEC. It keeps the BasicSR
training path and adds the same on-the-fly exposure degradation and
hallucination consistency objective used by the paper.

## Contents

- `train.py`: BasicSR training entrypoint.
- `test.py`: BasicSR testing entrypoint.
- `archs/co_lut_arch.py`: CoTF/CoNet generator architecture.
- `models/ft2_model.py`: training model with supervised losses and vision-aided GAN loss.
- `data/paired_resize_msec_dataset.py`: on-the-fly exposure degradation dataset.
- `losses/ssim_loss.py`: SSIM loss registered for BasicSR.
- `options/train/train_msec.yml`: cleaned training config.
- `options/test/test_msec.yml`: cleaned test config.

## Method Match

The released CoTF branch follows the paper's CoTF instantiation:

- Samples exposure `e` continuously from `[-2, 2]`.
- Samples camera response parameters with `gamma ~ N(0.9, 0.1)` and `beta ~ N(0.6, 0.1)`.
- Generates the degraded input and the opposite-exposure target on the fly from ordinary GT images.
- Re-degrades the network prediction with the opposite exposure in `models/ft2_model.py`.
- Optimizes the CoTF reconstruction losses plus the CLIP-guided hallucination consistency GAN loss.

## Installation

```bash
conda create -n cotf python=3.8 -y
conda activate cotf
pip install -r requirements.txt
```

`vision_aided_loss` is required by the discriminator. If your package index does
not provide it, install it from the upstream project used by your training
environment.

## Dataset

The default training option expects this layout:

```text
datasets/msec/
  train/
    gt/
      image_001.png
  test/
    INPUT_IMAGES/
      image_001.png
    expert_c_testing_set/
      image_001.png
```

Training samples are generated from GT images on the fly with the same exposure
simulation used by the OSEDiff-GANP training code.

Set `noise_std` in `options/train/train_msec.yml` only when training for noisier
low-light settings. The default `0.0` matches cleaner MSEC/SICEV2-style usage.

## Training

Edit `options/train/train_msec.yml` if your dataset paths or batch size differ,
then run:

```bash
python train.py -opt options/train/train_msec.yml
```

## Testing

Place model weights outside the repository, for example under `pretrained/`, and
set `path.pretrain_network_g` in `options/test/test_msec.yml`.

```bash
python test.py -opt options/test/test_msec.yml
```

## Release Notes

This folder intentionally excludes:

- `experiments/`, `results/`, `tb_logger/`, and archived runs.
- `.pth` pretrained weights and generated checkpoints.
- Machine-specific absolute paths.
- Alternative dataset variants and old ablation files not used by the main MSEC GAN training path.
