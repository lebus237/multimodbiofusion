# Multimodal Biometric Fusion

PyTorch implementation of **"Artificial Intelligence-Enabled Deep Learning Model for Multimodal Biometric Fusion"**  
*Multimedia Tools and Applications (2024)* — DOI: [10.1007/s11042-024-18509-0](https://doi.org/10.1007/s11042-024-18509-0)

---

## Overview

This project implements a three-level multimodal biometric fusion system combining **face**, **fingerprint**, and **iris** modalities.

| Fusion Level | Methods | Result |
|---|---|---|
| Pixel | Channel / Intensity / Spatial | +2.2 pp over single-modal |
| Feature | Joint Representation Layer | +2.2 pp over single-modal |
| Score | Rank-1 / Modality Evaluation | +3.5 pp → **99.6% Rank-1** |

---

## Project Structure

```
multimodal_biometric_fusion/
├── config/
│   └── config.yaml          # All hyper-parameters
├── src/
│   ├── data/
│   │   ├── dataset.py       # PyTorch dataset & data loaders
│   │   └── preprocessing.py # Low-quality image simulation
│   ├── models/
│   │   ├── backbones.py     # VGG16, ResNet50, DenseNet169
│   │   ├── pixel_fusion.py  # Channel, Intensity, Spatial fusion
│   │   ├── feature_fusion.py# 3-branch CNN + Joint FC layer
│   │   └── score_fusion.py  # Rank-1 & Modality evaluation
│   ├── training/
│   │   ├── losses.py        # ArcFace margin loss
│   │   └── trainer.py       # Training loop
│   └── evaluation/
│       └── metrics.py       # mAP, Rank-N, CMC curve
└── scripts/
    ├── prepare_dataset.py   # Build virtual homologous dataset
    ├── train_pixel.py       # Train pixel-level fusion models
    ├── train_feature.py     # Train feature-level fusion model
    └── evaluate.py          # Full evaluation + comparison table
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare dataset

Download the source datasets:
- **CASIA-IrisV4-Interval** — iris data for 2,712 identities
- **CASIA-FingerprintV5** — fingerprint data for 500 individuals (expanded)
- **WebFace 260M** (subset) — face data

Then run:
```bash
python scripts/prepare_dataset.py \
  --iris_dir      /path/to/CASIA-IrisV4-Interval \
  --fingerprint_dir /path/to/CASIA-FingerprintV5 \
  --face_dir      /path/to/WebFace260M_subset \
  --output_dir    data/virtual_dataset
```

### 3. Train

**Pixel-level fusion** (channel, intensity, or spatial):
```bash
python scripts/train_pixel.py \
  --config config/config.yaml \
  --backbone vgg16 \
  --fusion_type channel
```

**Feature-level fusion**:
```bash
python scripts/train_feature.py \
  --config config/config.yaml \
  --backbone vgg16
```

### 4. Evaluate all methods

```bash
python scripts/evaluate.py \
  --config config/config.yaml \
  --checkpoint_dir checkpoints/
```

This produces:
- CMC curves for all methods
- mAP / Rank-1/5/10 comparison table
- Score-fusion results (no additional training needed)

---

## Architecture Details

### Pixel-Level Fusion

Three strategies are applied before the backbone network:

| Method | Description | Input Size |
|---|---|---|
| Channel | Concat face+iris+fp along channels | [B, 9, 224, 224] |
| Intensity | Weighted sum: N₁·face + N₂·iris + N₃·fp | [B, 3, 224, 224] |
| Spatial | Horizontal stitch → resize to 224×224 | [B, 3, 224, 224] |

### Feature-Level Fusion

```
Face ──► Branch_F ──► 512-d
Iris ──► Branch_I ──► 512-d ──► Concat[1536] ──► FC ──► 512-d
Iris ──► Branch_P ──► 512-d
```

All branches and the joint layer are optimised jointly via backpropagation.

### Score-Level Fusion (Inference-only)

**Modality Evaluation (Eq. 4):**
```
D_t = Σ_q s[q]
```

**Rank-1 Evaluation (Eq. 5):**
```
D_t = s[rank1] - Σ_{q≠rank1} s[q]
```

The modality with the highest D_t is trusted for the final decision.

---

## Backbone Networks

All four backbones from the paper are supported:
- `vgg16` — standard VGG-16
- `vgg16_bn` — VGG-16 with batch normalisation before activation
- `resnet50` — ResNet-50
- `densenet169` — DenseNet-169

---

## Citation

```bibtex
@article{byeon2024multimodal,
  title   = {Artificial intelligence-Enabled deep learning model for multimodal biometric fusion},
  author  = {Byeon, Haewon and Raina, Vikas and Sandhu, Mukta and Shabaz, Mohammad and others},
  journal = {Multimedia Tools and Applications},
  volume  = {83},
  pages   = {80105--80128},
  year    = {2024},
  doi     = {10.1007/s11042-024-18509-0}
}
```
