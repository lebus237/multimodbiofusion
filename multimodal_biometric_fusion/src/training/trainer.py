"""
trainer.py
==========
Training loops for all three fusion levels.

Trainer             — generic trainer for PixelFusionModel / ModalityBranch.
FeatureFusionTrainer — specialized trainer for FeatureFusionModel with
                       multi-task loss (joint + per-modality auxiliary losses).
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from .losses import ArcFaceLoss, CombinedLoss

# ── Utility ───────────────────────────────────────────────────────────────────


def _set_seed(seed: int) -> None:
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ── Generic Trainer ───────────────────────────────────────────────────────────


class Trainer:
    """
    Generic trainer for PixelFusionModel and single-modality ModalityBranch
    networks.

    Parameters
    ----------
    model      : nn.Module  — the model to train.
    n_classes  : int        — number of identity classes.
    lr         : float      — initial learning rate.
    weight_decay : float
    scale, margin : float   — ArcFace hyper-parameters.
    epochs     : int
    warmup_epochs : int     — number of linear warm-up epochs.
    device     : str
    save_dir   : str        — directory to save checkpoints.
    log_dir    : str        — TensorBoard log directory.
    seed       : int
    """

    def __init__(
        self,
        model: nn.Module,
        n_classes: int = 2712,
        lr: float = 1e-4,
        weight_decay: float = 5e-4,
        scale: float = 64.0,
        margin: float = 0.50,
        epochs: int = 100,
        warmup_epochs: int = 5,
        device: str = "cuda",
        save_dir: str = "checkpoints",
        log_dir: str = "logs",
        seed: int = 42,
    ) -> None:
        _set_seed(seed)
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.epochs = epochs
        self.warmup_epochs = warmup_epochs
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir)

        # Loss
        self.loss_fn = ArcFaceLoss(
            embed_dim=512, n_classes=n_classes, scale=scale, margin=margin
        ).to(self.device)

        # Optimiser (model params + ArcFace centre params)
        self.optimizer = Adam(
            list(model.parameters()) + list(self.loss_fn.parameters()),
            lr=lr,
            weight_decay=weight_decay,
        )

        # LR scheduler: linear warm-up → cosine annealing
        if warmup_epochs > 0:
            warmup = LinearLR(
                self.optimizer,
                start_factor=1e-3,
                total_iters=warmup_epochs,
            )
            cosine = CosineAnnealingLR(
                self.optimizer,
                T_max=max(1, epochs - warmup_epochs),
            )
            self.scheduler = SequentialLR(
                self.optimizer,
                schedulers=[warmup, cosine],
                milestones=[warmup_epochs],
            )
        else:
            self.scheduler = CosineAnnealingLR(self.optimizer, T_max=epochs)

    # ── Training ──────────────────────────────────────────────────────────────

    def train_epoch(self, loader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0

        for face, iris, fp, labels in loader:
            face, iris, fp, labels = (
                face.to(self.device),
                iris.to(self.device),
                fp.to(self.device),
                labels.to(self.device),
            )
            self.optimizer.zero_grad()
            embeddings = self.model(face, iris, fp)
            loss = self.loss_fn(embeddings, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
            self.optimizer.step()
            total_loss += loss.item()

        self.scheduler.step()
        return total_loss / max(1, len(loader))

    @torch.no_grad()
    def val_epoch(self, loader: DataLoader) -> float:
        self.model.eval()
        total_loss = 0.0
        for face, iris, fp, labels in loader:
            face, iris, fp, labels = (
                face.to(self.device),
                iris.to(self.device),
                fp.to(self.device),
                labels.to(self.device),
            )
            embeddings = self.model(face, iris, fp)
            loss = self.loss_fn(embeddings, labels)
            total_loss += loss.item()
        return total_loss / max(1, len(loader))

    # ── Full training loop ────────────────────────────────────────────────────

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        save_every: int = 10,
        run_name: str = "model",
    ) -> Dict[str, list]:
        history: Dict[str, list] = {"train_loss": [], "val_loss": []}

        for epoch in range(1, self.epochs + 1):
            t0 = time.time()
            train_loss = self.train_epoch(train_loader)
            val_loss = self.val_epoch(val_loader)
            elapsed = time.time() - t0

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)

            self.writer.add_scalar("Loss/train", train_loss, epoch)
            self.writer.add_scalar("Loss/val", val_loss, epoch)
            self.writer.add_scalar(
                "LR",
                self.optimizer.param_groups[0]["lr"],
                epoch,
            )

            print(
                f"[{run_name}] Epoch {epoch:3d}/{self.epochs} | "
                f"train={train_loss:.4f}  val={val_loss:.4f} | "
                f"{elapsed:.1f}s"
            )

            if epoch % save_every == 0 or epoch == self.epochs:
                self.save(f"{run_name}_epoch{epoch:03d}.pt")

        self.writer.close()
        return history

    def save(self, filename: str) -> None:
        path = self.save_dir / filename
        torch.save(
            {
                "model_state": self.model.state_dict(),
                "loss_state": self.loss_fn.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
            },
            path,
        )
        print(f"  ✓ Saved checkpoint → {path}")

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.loss_fn.load_state_dict(ckpt["loss_state"])
        print(f"  ✓ Loaded checkpoint ← {path}")


# ── Feature-Fusion Trainer ────────────────────────────────────────────────────


class FeatureFusionTrainer(Trainer):
    """
    Specialised trainer for FeatureFusionModel.

    Adds per-modality auxiliary losses to reinforce each branch.
    Loss = loss(fused) + λ·[loss(face) + loss(iris) + loss(fp)]

    Parameters
    ----------
    modality_weight : float
        Weight λ for each auxiliary modality loss (default 0.3 per paper).
    """

    def __init__(
        self,
        model: nn.Module,
        n_classes: int = 2712,
        lr: float = 1e-4,
        weight_decay: float = 5e-4,
        scale: float = 64.0,
        margin: float = 0.50,
        epochs: int = 100,
        warmup_epochs: int = 5,
        device: str = "cuda",
        save_dir: str = "checkpoints",
        log_dir: str = "logs",
        seed: int = 42,
        modality_weight: float = 0.3,
    ) -> None:
        super().__init__(
            model,
            n_classes,
            lr,
            weight_decay,
            scale,
            margin,
            epochs,
            warmup_epochs,
            device,
            save_dir,
            log_dir,
            seed,
        )
        self.modality_weight = modality_weight

    def train_epoch(self, loader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0

        for face, iris, fp, labels in loader:
            face, iris, fp, labels = (
                face.to(self.device),
                iris.to(self.device),
                fp.to(self.device),
                labels.to(self.device),
            )
            self.optimizer.zero_grad()

            # FeatureFusionModel returns (fused, face_feat, iris_feat, fp_feat)
            output = self.model(face, iris, fp)
            fused_feat, face_feat, iris_feat, fp_feat = output

            loss_fused = self.loss_fn(fused_feat, labels)
            loss_face = self.loss_fn(face_feat, labels)
            loss_iris = self.loss_fn(iris_feat, labels)
            loss_fp = self.loss_fn(fp_feat, labels)

            loss = loss_fused + self.modality_weight * (loss_face + loss_iris + loss_fp)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
            self.optimizer.step()
            total_loss += loss.item()

        self.scheduler.step()
        return total_loss / max(1, len(loader))

    @torch.no_grad()
    def val_epoch(self, loader: DataLoader) -> float:
        self.model.eval()
        total_loss = 0.0
        for face, iris, fp, labels in loader:
            face, iris, fp, labels = (
                face.to(self.device),
                iris.to(self.device),
                fp.to(self.device),
                labels.to(self.device),
            )
            # eval mode returns only fused embedding
            fused_feat = self.model(face, iris, fp)
            loss = self.loss_fn(fused_feat, labels)
            total_loss += loss.item()
        return total_loss / max(1, len(loader))
