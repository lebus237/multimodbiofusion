"""
losses.py
=========
Loss functions for training the multimodal biometric fusion models.

ArcFaceLoss  — primary loss for biometric embedding learning.
TripletLoss  — optional auxiliary loss.
CombinedLoss — weighted sum of ArcFace + Triplet.

Reference: ArcFace: Additive Angular Margin Loss (Deng et al., CVPR 2019)
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# ── ArcFace Loss ──────────────────────────────────────────────────────────────


class ArcFaceLoss(nn.Module):
    """
    Additive Angular Margin Loss (ArcFace).

    Adds a fixed margin m in the angular space to improve the
    discriminability of learned embeddings.

    Parameters
    ----------
    embed_dim : int
        Dimensionality of the embedding (512 in the paper).
    n_classes : int
        Number of identity classes (2712 in the paper).
    scale : float
        Feature scale s (paper uses 64.0).
    margin : float
        Angular margin m (paper uses 0.50 rad ≈ 28.6°).
    """

    def __init__(
        self,
        embed_dim: int = 512,
        n_classes: int = 2712,
        scale: float = 64.0,
        margin: float = 0.50,
    ) -> None:
        super().__init__()
        self.scale = scale
        self.margin = margin
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)  # threshold for numerical stability
        self.mm = math.sin(math.pi - margin) * margin

        # Class-centre weight matrix — normalised during forward
        self.weight = nn.Parameter(torch.FloatTensor(n_classes, embed_dim))
        nn.init.xavier_uniform_(self.weight)

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        embeddings : torch.Tensor  [B, embed_dim]
            L2-normalised feature vectors.
        labels     : torch.Tensor  [B]  (long)
            Ground-truth class indices.

        Returns
        -------
        torch.Tensor — scalar loss value
        """
        # Cosine similarity between embeddings and weight centres
        cosine = F.linear(
            F.normalize(embeddings, p=2, dim=1),
            F.normalize(self.weight, p=2, dim=1),
        )  # [B, n_classes]

        # Add angular margin to the target class
        sine = torch.sqrt(1.0 - cosine.pow(2).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m  # cos(θ + m)
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1.0)

        logits = one_hot * phi + (1.0 - one_hot) * cosine
        logits = logits * self.scale

        return self.criterion(logits, labels)


# ── Triplet Loss ──────────────────────────────────────────────────────────────


class TripletLoss(nn.Module):
    """
    Batch hard triplet loss with cosine distance.

    For each anchor in the batch, the hardest positive (furthest same-class)
    and hardest negative (closest different-class) are selected.

    Parameters
    ----------
    margin : float
        Triplet margin.
    """

    def __init__(self, margin: float = 0.30) -> None:
        super().__init__()
        self.margin = margin

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # Pairwise cosine distance = 1 - cosine_similarity
        emb_n = F.normalize(embeddings, p=2, dim=1)
        dist = 1.0 - emb_n @ emb_n.T  # [B, B]

        labels_eq = labels.unsqueeze(0) == labels.unsqueeze(1)  # [B, B]
        labels_ne = ~labels_eq

        # Hardest positive: maximum distance among same-class pairs
        pos_dist = (dist * labels_eq.float()).max(dim=1).values

        # Hardest negative: minimum distance among different-class pairs
        neg_dist = dist.clone()
        neg_dist[labels_eq] = float("inf")
        neg_dist = neg_dist.min(dim=1).values

        loss = F.relu(pos_dist - neg_dist + self.margin)
        return loss.mean()


# ── Combined Loss ─────────────────────────────────────────────────────────────


class CombinedLoss(nn.Module):
    """
    Weighted combination: α·ArcFace + β·Triplet.

    Parameters
    ----------
    embed_dim, n_classes, scale, margin_arc : ArcFace hyper-parameters.
    margin_triplet : float  — triplet margin.
    alpha : float           — weight for ArcFace.
    beta  : float           — weight for Triplet.
    """

    def __init__(
        self,
        embed_dim: int = 512,
        n_classes: int = 2712,
        scale: float = 64.0,
        margin_arc: float = 0.50,
        margin_triplet: float = 0.30,
        alpha: float = 1.0,
        beta: float = 0.1,
    ) -> None:
        super().__init__()
        self.arcface = ArcFaceLoss(embed_dim, n_classes, scale, margin_arc)
        self.triplet = TripletLoss(margin_triplet)
        self.alpha = alpha
        self.beta = beta

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return self.alpha * self.arcface(embeddings, labels) + self.beta * self.triplet(
            embeddings, labels
        )
