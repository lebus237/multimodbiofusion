"""
dataset.py
==========
PyTorch Dataset and DataLoader factory for the virtual homologous
multimodal biometric dataset described in paper §5.1.

Dataset layout on disk (produced by scripts/prepare_dataset.py):

    data/virtual_dataset/
        {identity_id:04d}/
            face/
                {img_id:02d}.jpg
            iris/
                {img_id:02d}.jpg
            fingerprint/
                {img_id:02d}.jpg

2712 identities, ≥5 images per identity per modality.
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from .preprocessing import (
    bgr_to_rgb,
    preprocess_face,
    preprocess_fingerprint,
    preprocess_iris,
)

# ── Default augmentation / normalisation transforms ─────────────────────────

_NORM = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
)

TRAIN_TRANSFORMS = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
        transforms.RandomAffine(degrees=10, translate=(0.05, 0.05)),
        transforms.ToTensor(),
        _NORM,
    ]
)

EVAL_TRANSFORMS = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.ToTensor(),
        _NORM,
    ]
)


# ── Dataset ──────────────────────────────────────────────────────────────────


class MultimodalBiometricDataset(Dataset):
    """
    Virtual homologous multimodal dataset.

    Each ``__getitem__`` call returns a tuple
    ``(face_tensor, iris_tensor, fingerprint_tensor, label)``
    where each tensor has shape ``(3, 224, 224)``.

    Parameters
    ----------
    root_dir : str | Path
        Root directory of the virtual dataset.
    split : str
        One of ``"train"``, ``"val"``, ``"test"``.
    transform_face : callable, optional
        Transform applied to the face image.  Defaults to
        ``TRAIN_TRANSFORMS`` for train and ``EVAL_TRANSFORMS`` otherwise.
    transform_iris : callable, optional
        Transform applied to the iris image.
    transform_fp : callable, optional
        Transform applied to the fingerprint image.
    train_ratio : float
        Fraction of identities used for training.
    val_ratio : float
        Fraction of identities used for validation.
    seed : int
        Random seed for reproducible splits.
    """

    def __init__(
        self,
        root_dir: str | Path,
        split: str = "train",
        transform_face: Optional[object] = None,
        transform_iris: Optional[object] = None,
        transform_fp: Optional[object] = None,
        train_ratio: float = 0.80,
        val_ratio: float = 0.10,
        seed: int = 42,
    ) -> None:
        assert split in ("train", "val", "test"), (
            f"split must be 'train', 'val', or 'test', got '{split}'"
        )
        self.root_dir = Path(root_dir)
        self.split = split

        is_train = split == "train"
        self.transform_face = transform_face or (
            TRAIN_TRANSFORMS if is_train else EVAL_TRANSFORMS
        )
        self.transform_iris = transform_iris or (
            TRAIN_TRANSFORMS if is_train else EVAL_TRANSFORMS
        )
        self.transform_fp = transform_fp or (
            TRAIN_TRANSFORMS if is_train else EVAL_TRANSFORMS
        )

        self.samples: List[Dict] = []
        self.label_to_idx: Dict[str, int] = {}
        self._build_index(train_ratio, val_ratio, seed)

    # ── Index building ───────────────────────────────────────────────────────

    def _build_index(self, train_ratio: float, val_ratio: float, seed: int) -> None:
        """
        Scans the root directory and builds a flat list of samples.
        Each sample is a dict with keys: face, iris, fingerprint, label.
        """
        identities = sorted([d for d in self.root_dir.iterdir() if d.is_dir()])
        if not identities:
            raise RuntimeError(
                f"No identity sub-directories found in '{self.root_dir}'. "
                "Run scripts/prepare_dataset.py first."
            )

        # Deterministic shuffle then split
        rng = random.Random(seed)
        rng.shuffle(identities)

        n = len(identities)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        splits = {
            "train": identities[:n_train],
            "val": identities[n_train : n_train + n_val],
            "test": identities[n_train + n_val :],
        }
        chosen = splits[self.split]

        for idx, identity_dir in enumerate(chosen):
            identity_id = identity_dir.name
            self.label_to_idx[identity_id] = idx

            face_dir = identity_dir / "face"
            iris_dir = identity_dir / "iris"
            fp_dir = identity_dir / "fingerprint"

            if not (face_dir.exists() and iris_dir.exists() and fp_dir.exists()):
                continue  # skip incomplete identities

            face_imgs = sorted(face_dir.glob("*.jpg")) + sorted(face_dir.glob("*.png"))
            iris_imgs = sorted(iris_dir.glob("*.jpg")) + sorted(iris_dir.glob("*.png"))
            fp_imgs = sorted(fp_dir.glob("*.jpg")) + sorted(fp_dir.glob("*.png"))

            # Use the minimum count to keep tuples aligned
            n_samples = min(len(face_imgs), len(iris_imgs), len(fp_imgs))
            for i in range(n_samples):
                self.samples.append(
                    {
                        "face": str(face_imgs[i]),
                        "iris": str(iris_imgs[i]),
                        "fingerprint": str(fp_imgs[i]),
                        "label": idx,
                    }
                )

    # ── Dataset interface ────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        s = self.samples[idx]

        face_img = bgr_to_rgb(preprocess_face(s["face"]))
        iris_img = bgr_to_rgb(preprocess_iris(s["iris"]))
        fp_img = bgr_to_rgb(preprocess_fingerprint(s["fingerprint"]))

        face_t = self.transform_face(face_img)
        iris_t = self.transform_iris(iris_img)
        fp_t = self.transform_fp(fp_img)
        label = torch.tensor(s["label"], dtype=torch.long)

        return face_t, iris_t, fp_t, label

    @property
    def num_classes(self) -> int:
        return len(self.label_to_idx)


# ── DataLoader factory ────────────────────────────────────────────────────────


def build_dataloaders(
    root_dir: str | Path,
    batch_size: int = 64,
    num_workers: int = 4,
    pin_memory: bool = True,
    train_ratio: float = 0.80,
    val_ratio: float = 0.10,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader, int]:
    """
    Build train / val / test DataLoaders.

    Returns
    -------
    train_loader, val_loader, test_loader, num_classes
    """
    train_ds = MultimodalBiometricDataset(
        root_dir, "train", train_ratio=train_ratio, val_ratio=val_ratio, seed=seed
    )
    val_ds = MultimodalBiometricDataset(
        root_dir, "val", train_ratio=train_ratio, val_ratio=val_ratio, seed=seed
    )
    test_ds = MultimodalBiometricDataset(
        root_dir, "test", train_ratio=train_ratio, val_ratio=val_ratio, seed=seed
    )

    def _loader(ds, shuffle):
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=shuffle,
        )

    return (
        _loader(train_ds, shuffle=True),
        _loader(val_ds, shuffle=False),
        _loader(test_ds, shuffle=False),
        train_ds.num_classes,
    )
