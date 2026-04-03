"""
backbones.py
============
Backbone network definitions for multimodal biometric feature extraction.

Four backbones are supported as in the paper §5.2:
  - VGG-16 (vgg16)
  - VGG-16 with BatchNorm before activation (vgg16_bn)
  - ResNet-50 (resnet50)
  - DenseNet-169 (densenet169)

Each backbone is modified to output a 512-dimensional embedding vector
(paper §3.3: "one-dimensional feature vector whose dimension is [512, 1]").
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

EMBED_DIM: int = 512  # output embedding dimension (paper §3.3)


# ── Helpers ──────────────────────────────────────────────────────────────────


def _replace_classifier_vgg(model: nn.Module, in_channels: int = 3) -> nn.Module:
    """
    Replace VGG classifier with a 512-dim embedding head.
    Optionally patches the first conv layer for non-3-channel inputs.
    """
    if in_channels != 3:
        old_conv = model.features[0]
        model.features[0] = nn.Conv2d(
            in_channels,
            old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=old_conv.bias is not None,
        )
    model.classifier = nn.Sequential(
        nn.Linear(25088, 4096),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.5),
        nn.Linear(4096, 1024),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.5),
        nn.Linear(1024, EMBED_DIM),
    )
    return model


# ── Backbone factory ─────────────────────────────────────────────────────────


def build_backbone(
    name: str,
    pretrained: bool = True,
    in_channels: int = 3,
) -> nn.Module:
    """
    Build a backbone network with a 512-dim embedding head.

    Parameters
    ----------
    name : str
        One of ``'vgg16'``, ``'vgg16_bn'``, ``'resnet50'``, ``'densenet169'``.
    pretrained : bool
        Load ImageNet weights.
    in_channels : int
        Number of input channels.  Use 9 for channel-fusion (paper §3.2).

    Returns
    -------
    nn.Module
        Modified backbone with embedding head.
    """
    weights_map = {
        "vgg16": models.VGG16_Weights.IMAGENET1K_V1,
        "vgg16_bn": models.VGG16_BN_Weights.IMAGENET1K_V1,
        "resnet50": models.ResNet50_Weights.IMAGENET1K_V2,
        "densenet169": models.DenseNet169_Weights.IMAGENET1K_V1,
    }

    if name == "vgg16":
        weights = weights_map["vgg16"] if pretrained else None
        model = models.vgg16(weights=weights)
        model = _replace_classifier_vgg(model, in_channels)

    elif name == "vgg16_bn":
        # VGG-16 with batch normalisation before activation (paper §5.2)
        weights = weights_map["vgg16_bn"] if pretrained else None
        model = models.vgg16_bn(weights=weights)
        model = _replace_classifier_vgg(model, in_channels)

    elif name == "resnet50":
        weights = weights_map["resnet50"] if pretrained else None
        model = models.resnet50(weights=weights)
        if in_channels != 3:
            old_conv = model.conv1
            model.conv1 = nn.Conv2d(
                in_channels,
                old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=False,
            )
        model.fc = nn.Sequential(
            nn.Linear(model.fc.in_features, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, EMBED_DIM),
        )

    elif name == "densenet169":
        weights = weights_map["densenet169"] if pretrained else None
        model = models.densenet169(weights=weights)
        if in_channels != 3:
            old_conv = model.features.conv0
            model.features.conv0 = nn.Conv2d(
                in_channels,
                old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=False,
            )
        model.classifier = nn.Sequential(
            nn.Linear(model.classifier.in_features, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, EMBED_DIM),
        )

    else:
        raise ValueError(
            f"Unknown backbone '{name}'. "
            "Choose from: vgg16, vgg16_bn, resnet50, densenet169."
        )

    return model


# ── Single-modality branch ────────────────────────────────────────────────────


class ModalityBranch(nn.Module):
    """
    A single-modality processing branch.

    Wraps a backbone and applies L2 normalisation to the output embedding
    so cosine similarity = dot product of normalised vectors.

    Parameters
    ----------
    backbone_name : str
        Name of the backbone (see ``build_backbone``).
    pretrained : bool
        Load ImageNet weights.
    """

    def __init__(self, backbone_name: str = "vgg16", pretrained: bool = True) -> None:
        super().__init__()
        self.backbone_name = backbone_name
        self.backbone = build_backbone(backbone_name, pretrained=pretrained)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor  [B, C, H, W]

        Returns
        -------
        torch.Tensor  [B, 512] — L2-normalised embedding
        """
        feat = self.backbone(x)
        return F.normalize(feat, p=2, dim=1)

    def __repr__(self) -> str:
        return f"ModalityBranch(backbone={self.backbone_name}, embed_dim={EMBED_DIM})"
