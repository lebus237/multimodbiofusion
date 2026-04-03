#!/usr/bin/env python
"""
prepare_dataset.py
==================
Build the virtual homologous multimodal dataset from raw public datasets.

Source datasets (see paper §5.1):
  - CASIA-IrisV4-Interval  → iris modality
  - CASIA-FingerprintV5    → fingerprint modality
  - WebFace 260M (subset)  → face modality

Output directory layout:
  {output_dir}/
      {identity_id:04d}/
          face/
              00.jpg  01.jpg  …
          iris/
              00.jpg  01.jpg  …
          fingerprint/
              00.jpg  01.jpg  …

Usage
-----
python scripts/prepare_dataset.py \
    --iris_dir        /path/to/CASIA-IrisV4-Interval \
    --fingerprint_dir /path/to/CASIA-FingerprintV5 \
    --face_dir        /path/to/WebFace260M_subset \
    --output_dir      data/virtual_dataset \
    --num_identities  2712 \
    --min_samples     5 \
    --seed            42
"""

from __future__ import annotations

import argparse
import os
import random
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from tqdm import tqdm

# Add parent directory to path so we can import src
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data.preprocessing import (
    preprocess_face,
    preprocess_fingerprint,
    preprocess_iris,
)

# ── Helpers ───────────────────────────────────────────────────────────────────


def collect_identity_paths(
    root: Path,
    min_samples: int = 5,
) -> Dict[str, List[Path]]:
    """
    Scan a dataset directory and return {identity_id: [image_paths]}.

    Handles two common layouts:
      Layout A: root/{identity_id}/{img}.jpg
      Layout B: root/{img_with_id_prefix}.jpg  (e.g. "0001_01.jpg")
    """
    identities: Dict[str, List[Path]] = {}

    if not root.exists():
        print(f"  ⚠  Directory not found: {root}")
        return identities

    # Try Layout A first
    for subdir in sorted(root.iterdir()):
        if subdir.is_dir():
            imgs = sorted(
                list(subdir.glob("*.jpg"))
                + list(subdir.glob("*.png"))
                + list(subdir.glob("*.bmp"))
            )
            if len(imgs) >= min_samples:
                identities[subdir.name] = imgs

    # Fallback: flat directory — group by filename prefix (first 4 digits)
    if not identities:
        flat_imgs = sorted(
            list(root.glob("*.jpg"))
            + list(root.glob("*.png"))
            + list(root.glob("*.bmp"))
        )
        for img in flat_imgs:
            id_key = img.stem[:4]
            identities.setdefault(id_key, []).append(img)
        identities = {k: v for k, v in identities.items() if len(v) >= min_samples}

    return identities


def save_image(img: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img)


# ── Main builder ──────────────────────────────────────────────────────────────


def build_dataset(
    iris_dir: Path,
    fingerprint_dir: Path,
    face_dir: Path,
    output_dir: Path,
    num_identities: int = 2712,
    min_samples: int = 5,
    seed: int = 42,
) -> None:
    print("\n══════════════════════════════════════════")
    print("  Building Virtual Multimodal Dataset")
    print("══════════════════════════════════════════")

    rng = random.Random(seed)

    # Collect paths
    print(f"\n[1/4] Scanning iris directory:        {iris_dir}")
    iris_ids = collect_identity_paths(iris_dir, min_samples)
    print(f"      → {len(iris_ids)} valid identities found")

    print(f"[2/4] Scanning fingerprint directory: {fingerprint_dir}")
    fp_ids = collect_identity_paths(fingerprint_dir, min_samples)
    print(f"      → {len(fp_ids)} valid identities found")

    print(f"[3/4] Scanning face directory:        {face_dir}")
    face_ids = collect_identity_paths(face_dir, min_samples)
    print(f"      → {len(face_ids)} valid identities found")

    # Take the first min(available, num_identities) identities from each
    iris_keys = sorted(iris_ids.keys())[:num_identities]
    fp_keys = sorted(fp_ids.keys())[:num_identities]
    face_keys = sorted(face_ids.keys())[:num_identities]

    n = min(len(iris_keys), len(fp_keys), len(face_keys), num_identities)
    if n == 0:
        print("\n  ERROR: No identities found. Check your dataset paths.\n")
        print("  Hint: Run with --demo to generate a small synthetic dataset.")
        sys.exit(1)

    print(f"\n[4/4] Writing {n} identities to: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    skipped = 0
    for virtual_id in tqdm(range(n), desc="Processing identities"):
        out_base = output_dir / f"{virtual_id:04d}"

        iris_paths = iris_ids[iris_keys[virtual_id]]
        fp_paths = fp_ids[fp_keys[virtual_id]]
        face_paths = face_ids[face_keys[virtual_id]]

        n_samples = min(len(iris_paths), len(fp_paths), len(face_paths))
        if n_samples < min_samples:
            skipped += 1
            continue

        for sample_idx in range(n_samples):
            try:
                iris_img = preprocess_iris(iris_paths[sample_idx])
                fp_img = preprocess_fingerprint(fp_paths[sample_idx])
                face_img = preprocess_face(face_paths[sample_idx])
            except (FileNotFoundError, cv2.error) as e:
                continue

            save_image(face_img, out_base / "face" / f"{sample_idx:02d}.jpg")
            save_image(iris_img, out_base / "iris" / f"{sample_idx:02d}.jpg")
            save_image(fp_img, out_base / "fingerprint" / f"{sample_idx:02d}.jpg")

    created = len(list(output_dir.iterdir()))
    print(f"\n  ✓ Dataset built: {created} identities, {skipped} skipped")
    print(f"  ✓ Output: {output_dir}\n")


# ── Demo: synthetic dataset ───────────────────────────────────────────────────


def build_demo_dataset(
    output_dir: Path,
    num_identities: int = 50,
    samples_per_identity: int = 8,
    seed: int = 42,
) -> None:
    """
    Generate a tiny synthetic dataset with random noise images for testing
    the pipeline without real data.
    """
    print(
        f"\n  Generating DEMO dataset ({num_identities} identities × "
        f"{samples_per_identity} samples)…"
    )
    rng = np.random.default_rng(seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    for i in tqdm(range(num_identities), desc="Generating identities"):
        base = output_dir / f"{i:04d}"
        for s in range(samples_per_identity):
            # Random noise images as placeholder biometrics
            face_img = rng.integers(0, 255, (224, 224, 3), dtype=np.uint8)
            iris_img = rng.integers(0, 255, (224, 224, 3), dtype=np.uint8)
            fp_img = rng.integers(0, 255, (224, 224, 3), dtype=np.uint8)

            save_image(face_img, base / "face" / f"{s:02d}.jpg")
            save_image(iris_img, base / "iris" / f"{s:02d}.jpg")
            save_image(fp_img, base / "fingerprint" / f"{s:02d}.jpg")

    print(f"  ✓ Demo dataset ready at: {output_dir}\n")


# ── CLI ───────────────────────────────────────────────────────────────────────


def parse_args():
    p = argparse.ArgumentParser(
        description="Build virtual homologous multimodal biometric dataset."
    )
    p.add_argument(
        "--iris_dir", type=str, default="", help="Path to CASIA-IrisV4-Interval dataset"
    )
    p.add_argument(
        "--fingerprint_dir",
        type=str,
        default="",
        help="Path to CASIA-FingerprintV5 dataset",
    )
    p.add_argument(
        "--face_dir", type=str, default="", help="Path to WebFace260M subset"
    )
    p.add_argument(
        "--output_dir",
        type=str,
        default="data/virtual_dataset",
        help="Output directory",
    )
    p.add_argument("--num_identities", type=int, default=2712)
    p.add_argument("--min_samples", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--demo",
        action="store_true",
        help="Generate a small synthetic demo dataset instead",
    )
    p.add_argument("--demo_identities", type=int, default=50)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    out = Path(args.output_dir)

    if args.demo:
        build_demo_dataset(out, args.demo_identities, seed=args.seed)
    else:
        build_dataset(
            iris_dir=Path(args.iris_dir),
            fingerprint_dir=Path(args.fingerprint_dir),
            face_dir=Path(args.face_dir),
            output_dir=out,
            num_identities=args.num_identities,
            min_samples=args.min_samples,
            seed=args.seed,
        )
