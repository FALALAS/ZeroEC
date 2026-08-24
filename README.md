# ZeroEC

Official implementation of **ZeroEC: A Zero-Reference Framework for Robust Exposure Correction via Hallucination Consistency Learning**.

Paper link will be added when available.

[Ao Li](https://liaosite.github.io/)<sup>1</sup>, Jinqiang Yao<sup>4</sup>, Zhenyu Wang<sup>2,&dagger;</sup>, Tao Huang<sup>2</sup>, Yufan Zhu<sup>1</sup>, Fangfang Wu<sup>3</sup>, Weisheng Dong<sup>1,&dagger;</sup>

<sup>1</sup>School of Artificial Intelligence, Xidian University  
<sup>2</sup>Hangzhou Institute of Technology, Xidian University  
<sup>3</sup>School of Computer Science and Technology, Xidian University  
<sup>4</sup>CICO Research Global Co., Ltd., Hangzhou, China<br>
<sup>&dagger;</sup>Corresponding author

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
- SICEV2: https://share.weiyun.com/C2aJ1Cti
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
