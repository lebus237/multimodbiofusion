"""
feature_fusion.py
=================
Feature-level fusion method (paper §3.3).

Architecture:
    Face       ──► FaceBranch(VGG16)        ──► 512-d embedding
    Iris       ──► IrisBranch(VGG16)        ──► 512-d embedding
    Fingerprint ──► FingerprintBranch(VGG16) ──► 512-d embedding
                                                         │
                                               Concat [1536-d]
                                                         │
                                           JointRepresentationLayer
                                           (FC 1536 → 512, BN, ReLU)
                                                         │
                                                 512-d fused embedding

Formula (paper §3.3):
    Feature_i = Concat(Feature(Face_i), Feature(Iris_i), Feature(Fingerprint_i))
    → FC → [512, 1]

All three modality branches and the joint layer are trained jointly through
backpropagation, constructing first-order dependency relationships between
modalities.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbones import EMBED_DIM, ModalityBranch

CONCAT_DIM: int = EMBED_DIM * 3  # 512 × 3 = 1536


# ── Joint Representation Layer ────────────────────────────────────────────────


class JointRepresentationLayer(nn.Module):
    """
    Fully-connected layer that fuses concatenated modality embeddings.

    Input  : [B, 1536]  (concat of three 512-d embeddings)
    Output : [B, 512]   (fused embedding, L2-normalised)

    The paper uses a single FC layer with BN + ReLU for dimensionality
    reduction (§3.3).
    """

    def __init__(
        self,
        in_dim: int = CONCAT_DIM,
        out_dim: int = EMBED_DIM,
    ) -> None:
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_dim, out_dim * 2),  # 1536 → 1024
            nn.BatchNorm1d(out_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(out_dim * 2, out_dim),  # 1024 → 512
            nn.BatchNorm1d(out_dim),
            nn.ReLU(inplace=True),
        )

    def forward(
        self,
        face_feat: torch.Tensor,
        iris_feat: torch.Tensor,
        fp_feat: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        face_feat, iris_feat, fp_feat : torch.Tensor  [B, 512]

        Returns
        -------
        torch.Tensor  [B, 512] — L2-normalised fused embedding
        """
        concat = torch.cat([face_feat, iris_feat, fp_feat], dim=1)  # [B, 1536]
        fused = self.fc(concat)  # [B, 512]
        return F.normalize(fused, p=2, dim=1)


# ── Feature-level fusion model ────────────────────────────────────────────────


class FeatureFusionModel(nn.Module):
    """
    Three-branch feature-level fusion model (paper §3.3).

    Each modality has its own backbone branch.  The branches are
    concatenated and fed into a joint representation layer.  All
    parameters are optimised jointly through backpropagation.

    Parameters
    ----------
    backbone_name : str
        Backbone architecture shared by all three branches.
    pretrained : bool
        Load ImageNet weights.

    Returns (training mode)
    -----------------------
    fused_feat  : torch.Tensor [B, 512]  — joint embedding
    face_feat   : torch.Tensor [B, 512]  — per-modality embedding (auxiliary)
    iris_feat   : torch.Tensor [B, 512]
    fp_feat     : torch.Tensor [B, 512]

    Returns (eval mode)
    -------------------
    fused_feat  : torch.Tensor [B, 512]
    """

    def __init__(
        self,
        backbone_name: str = "vgg16",
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        self.backbone_name = backbone_name

        # Three independent modality branches (paper §3.3)
        self.face_branch = ModalityBranch(backbone_name, pretrained)
        self.iris_branch = ModalityBranch(backbone_name, pretrained)
        self.fingerprint_branch = ModalityBranch(backbone_name, pretrained)

        # Joint representation layer
        self.joint_layer = JointRepresentationLayer()

    def forward(
        self,
        face: torch.Tensor,
        iris: torch.Tensor,
        fingerprint: torch.Tensor,
    ) -> Tuple[torch.Tensor, ...] | torch.Tensor:
        """
        Parameters
        ----------
        face, iris, fingerprint : torch.Tensor  [B, 3, 224, 224]

        Returns
        -------
        During training:
            (fused_feat, face_feat, iris_feat, fp_feat)  — all [B, 512]
        During evaluation:
            fused_feat  [B, 512]
        """
        face_feat = self.face_branch(face)
        iris_feat = self.iris_branch(iris)
        fp_feat = self.fingerprint_branch(fingerprint)
        fused = self.joint_layer(face_feat, iris_feat, fp_feat)

        if self.training:
            return fused, face_feat, iris_feat, fp_feat
        return fused

    def encode_modalities(
        self,
        face: torch.Tensor,
        iris: torch.Tensor,
        fingerprint: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Always returns (fused, face, iris, fingerprint) embeddings,
        regardless of training/eval mode.  Useful for score-fusion evaluation.
        """
        face_feat = self.face_branch(face)
        iris_feat = self.iris_branch(iris)
        fp_feat = self.fingerprint_branch(fingerprint)
        fused = self.joint_layer(face_feat, iris_feat, fp_feat)
        return fused, face_feat, iris_feat, fp_feat

    def __repr__(self) -> str:
        return (
            f"FeatureFusionModel(backbone={self.backbone_name}, embed_dim={EMBED_DIM})"
        )
