"""
pixel_fusion.py
===============
Pixel-level fusion methods (paper §3.2).

Three strategies are implemented:

1. Channel Fusion
   Grayscale iris + grayscale fingerprint are expanded to 3-channel each,
   then concatenated with the colour face image along the channel axis:
       [B, 3, 224, 224] × 3  →  [B, 9, 224, 224]

2. Intensity Fusion
   Weighted sum of three modalities (learnable weights N1, N2, N3):
       Input_i = N1·Face + N2·Iris + N3·Fingerprint
       Output shape: [B, 3, 224, 224]

3. Spatial Fusion
   Three images are stitched horizontally and resized back to 224×224:
       Stitch: [B, 3, 224, 224×3=672]  →  resize  →  [B, 3, 224, 224]
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbones import EMBED_DIM, build_backbone

# ── Fusion modules ────────────────────────────────────────────────────────────


class ChannelFusion(nn.Module):
    """
    Channel-wise concatenation of face (RGB), iris, and fingerprint.

    Iris and fingerprint may arrive as 1-channel (grayscale) images.
    They are replicated to 3-channel before concatenation.
    Output channels = 9.
    """

    def forward(
        self,
        face: torch.Tensor,
        iris: torch.Tensor,
        fingerprint: torch.Tensor,
    ) -> torch.Tensor:
        # Ensure all modalities have 3 channels
        if iris.shape[1] == 1:
            iris = iris.expand(-1, 3, -1, -1)
        if fingerprint.shape[1] == 1:
            fingerprint = fingerprint.expand(-1, 3, -1, -1)

        return torch.cat([face, iris, fingerprint], dim=1)  # [B, 9, H, W]


class IntensityFusion(nn.Module):
    """
    Weighted pixel-intensity fusion (paper Eq. 1):
        Input_i = N1·Picture(Face_i) + N2·Picture(Iris_i) + N3·Picture(Fingerprint_i)

    N1, N2, N3 are learnable scalar parameters initialised to 1/3.
    A softmax is applied during the forward pass to ensure the weights
    sum to 1 and remain positive throughout training.
    """

    def __init__(self) -> None:
        super().__init__()
        # Raw logits; softmax keeps weights summing to 1
        self.logits = nn.Parameter(torch.zeros(3))

    def forward(
        self,
        face: torch.Tensor,
        iris: torch.Tensor,
        fingerprint: torch.Tensor,
    ) -> torch.Tensor:
        # Ensure all modalities have 3 channels
        if iris.shape[1] == 1:
            iris = iris.expand(-1, 3, -1, -1)
        if fingerprint.shape[1] == 1:
            fingerprint = fingerprint.expand(-1, 3, -1, -1)

        w = F.softmax(self.logits, dim=0)  # [3] — sum to 1
        return w[0] * face + w[1] * iris + w[2] * fingerprint  # [B, 3, H, W]

    @property
    def weights(self) -> Tuple[float, float, float]:
        """Return current N1, N2, N3 weights as a tuple."""
        w = F.softmax(self.logits, dim=0).detach().cpu()
        return float(w[0]), float(w[1]), float(w[2])


class SpatialFusion(nn.Module):
    """
    Horizontal stitch of face, iris, and fingerprint followed by a
    bilinear resize back to the original spatial size.

    Input:  3 × [B, 3, H, W]
    After stitch: [B, 3, H, 3W]
    After resize: [B, 3, H, W]   (same as a single input image)
    """

    def __init__(self, target_size: Tuple[int, int] = (224, 224)) -> None:
        super().__init__()
        self.target_size = target_size

    def forward(
        self,
        face: torch.Tensor,
        iris: torch.Tensor,
        fingerprint: torch.Tensor,
    ) -> torch.Tensor:
        if iris.shape[1] == 1:
            iris = iris.expand(-1, 3, -1, -1)
        if fingerprint.shape[1] == 1:
            fingerprint = fingerprint.expand(-1, 3, -1, -1)

        stitched = torch.cat([face, iris, fingerprint], dim=3)  # [B, 3, H, 3W]
        resized = F.interpolate(
            stitched,
            size=self.target_size,
            mode="bilinear",
            align_corners=False,
        )
        return resized  # [B, 3, H, W]


# ── End-to-end pixel-level fusion model ──────────────────────────────────────


class PixelFusionModel(nn.Module):
    """
    End-to-end model for pixel-level fusion (paper §3.2).

    Pipeline:
        (face, iris, fingerprint)  →  fusion module  →  backbone  →  embedding

    Parameters
    ----------
    backbone_name : str
        One of ``'vgg16'``, ``'vgg16_bn'``, ``'resnet50'``, ``'densenet169'``.
    fusion_type : str
        One of ``'channel'``, ``'intensity'``, ``'spatial'``.
    pretrained : bool
        Load ImageNet weights in the backbone.
    """

    def __init__(
        self,
        backbone_name: str = "vgg16",
        fusion_type: str = "channel",
        pretrained: bool = True,
    ) -> None:
        super().__init__()

        if fusion_type == "channel":
            self.fusion = ChannelFusion()
            in_channels = 9
        elif fusion_type == "intensity":
            self.fusion = IntensityFusion()
            in_channels = 3
        elif fusion_type == "spatial":
            self.fusion = SpatialFusion()
            in_channels = 3
        else:
            raise ValueError(
                f"Unknown fusion_type '{fusion_type}'. "
                "Choose from: channel, intensity, spatial."
            )

        self.fusion_type = fusion_type
        self.backbone_name = backbone_name
        self.backbone = build_backbone(
            backbone_name, pretrained=pretrained, in_channels=in_channels
        )

    def forward(
        self,
        face: torch.Tensor,
        iris: torch.Tensor,
        fingerprint: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        face, iris, fingerprint : torch.Tensor  [B, 3, 224, 224]

        Returns
        -------
        torch.Tensor  [B, 512] — L2-normalised embedding
        """
        fused = self.fusion(face, iris, fingerprint)
        feat = self.backbone(fused)
        return F.normalize(feat, p=2, dim=1)

    def __repr__(self) -> str:
        return (
            f"PixelFusionModel("
            f"backbone={self.backbone_name}, "
            f"fusion={self.fusion_type}, "
            f"embed_dim={EMBED_DIM})"
        )
