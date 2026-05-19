# ZeroEC

Official implementation of **ZeroEC: A Zero-Reference Framework for Robust Exposure Correction via Hallucination Consistency Learning**.

Paper link will be added when available.

Ao Li<sup>1</sup>, Zhenyu Wang<sup>2,*</sup>, Mingtao Feng<sup>1</sup>, Tao Huang<sup>2</sup>, Yufan Zhu<sup>1</sup>, Yuxin Feng<sup>2</sup>, Fangfang Wu<sup>3</sup>, Weisheng Dong<sup>1</sup>

<sup>1</sup>School of Artificial Intelligence, Xidian University  
<sup>2</sup>Hangzhou Institute of Technology, Xidian University  
<sup>3</sup>School of Computer Science and Technology, Xidian University  
<sup>*</sup>Corresponding author

## Overview

ZeroEC is a zero-reference learning framework for general exposure correction. It does not require curated paired or unpaired exposure-specific training data. Instead, it trains correction models from ordinary natural images by combining:

- A modified camera model that continuously samples exposure gain and camera response parameters to synthesize both over- and under-exposure degradations.
- An invert-clip-invert strategy for realistic under-exposure simulation and shadow detail loss.
- Hallucination consistency learning, where the corrected image is re-degraded with the opposite exposure and constrained against the deterministic opposite-exposure target.
- A model-agnostic pipeline instantiated with both a one-step diffusion backbone and a lightweight CoTF/LUT backbone.

## Code Structure

- `OSEDiff/`: one-step diffusion instantiation of ZeroEC, following the OSEDiff-style backbone with LoRA training.
- `CoTF/`: CoTF/LUT instantiation of ZeroEC, implemented in BasicSR style with the same on-the-fly degradation and hallucination consistency objective.

Each subfolder has its own `README.md` and `requirements.txt`.

## Datasets

ZeroEC trains on ordinary natural images, such as Flickr2K or MS COCO, rather than exposure-specific datasets. Evaluation in the paper uses:

- MSEC: https://github.com/mahmoudnafifi/Exposure_Correction
- SICEV2
- LOL v1

Datasets are not included in this repository. Download them separately and update the option files or command-line paths locally.

## Installation

Use the environment for the branch you want to run.

### OSEDiff Branch

```bash
cd OSEDiff
pip install -r requirements.txt
```

### CoTF Branch

```bash
cd CoTF
pip install -r requirements.txt
```

`vision_aided_loss` is required by both training pipelines for the CLIP-guided discriminator. If it is unavailable from your package index, install it from the upstream project used by your environment.

## Training

### OSEDiff

Prepare a dataset root containing `gt/` images and `tag/` prompt files. See `OSEDiff/README.md` for the exact layout.

```bash
cd OSEDiff
accelerate launch train_osediff_ganp.py \
  --root_folders /path/to/dataset_root \
  --pretrained_model_name_or_path /path/to/stable-diffusion-2-1-base \
  --output_dir outputs/osediff_ganp
```

### CoTF

Edit `CoTF/options/train/train_msec.yml` to point to your natural-image training folder, then run:

```bash
cd CoTF
python train.py -opt options/train/train_msec.yml
```

The CoTF config exposes `noise_std`. Keep it at `0.0` for cleaner MSEC/SICEV2-style settings, and set a positive value for noisier low-light settings such as LOL v1.

## Method Alignment

The released code follows the paper's main training logic:

- Exposure offset is sampled continuously from `[-2, 2]`.
- Camera response parameters are sampled as `gamma ~ N(0.9, 0.1)` and `beta ~ N(0.6, 0.1)`.
- For positive exposure, the degraded input is `C(I, e, beta, gamma)` and the opposite-exposure target is `1 - C(1 - I, e, beta, gamma)`.
- For negative exposure, the degraded input is `1 - C(1 - I, |e|, beta, gamma)` and the opposite-exposure target is `C(I, |e|, beta, gamma)`.
- The prediction is re-degraded with the opposite exposure and supervised by the hallucination consistency discriminator.

## Release Notes

This repository intentionally excludes datasets, pretrained weights, checkpoints, logs, cached files, and archived experiment outputs. Download datasets and pretrained models separately.

## Citation

```bibtex
@article{li2026zeroec,
  title={ZeroEC: A Zero-Reference Framework for Robust Exposure Correction via Hallucination Consistency Learning},
  author={Li, Ao and Wang, Zhenyu and Feng, Mingtao and Huang, Tao and Zhu, Yufan and Feng, Yuxin and Wu, Fangfang and Dong, Weisheng},
  journal={arXiv preprint},
  year={2026}
}
```
