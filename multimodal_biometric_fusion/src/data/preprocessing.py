"""
preprocessing.py
================
Image preprocessing and low-quality simulation for multimodal biometric data.

Paper §5.1.1:
  - Resolution reduced to 1/4 of original.
  - Iris images retain only 20% of original quality.
  - Fingerprint images retain 30% of original quality.
  - Face images are not degraded (cleaned from WebFace260M).
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Union, Optional, Tuple

# ── Constants from the paper ────────────────────────────────────────────────
IMG_SIZE: Tuple[int, int] = (224, 224)
IRIS_QUALITY: float = 0.20          # 20% JPEG quality
FINGERPRINT_QUALITY: float = 0.30   # 30% JPEG quality
RESOLUTION_FACTOR: float = 0.25     # downsample to 1/4 before upsampling

# ImageNet normalisation (applied later in transforms, NOT here)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


# ── Low-quality simulation ───────────────────────────────────────────────────

def simulate_low_quality(
    img: np.ndarray,
    quality: float,
    resolution_factor: float = RESOLUTION_FACTOR,
) -> np.ndarray:
    """
    Degrade an image to simulate real-world crime-scene conditions.

    Pipeline:
      1. Downsample to ``resolution_factor`` of original size.
      2. Upsample back to original size (introduces blur/pixelation).
      3. Apply JPEG compression at the given quality level.

    Parameters
    ----------
    img : np.ndarray
        BGR image loaded via cv2.
    quality : float
        Target quality fraction in [0, 1]. E.g. 0.20 → JPEG quality 20.
    resolution_factor : float
        Downsampling factor before upsampling.  Default 0.25 (1/4).

    Returns
    -------
    np.ndarray
        Degraded BGR image, same spatial size as input.
    """
    if img is None:
        raise ValueError("Input image is None — check the file path.")

    h, w = img.shape[:2]

    # Step 1 & 2: resolution degradation
    small_h = max(1, int(h * resolution_factor))
    small_w = max(1, int(w * resolution_factor))
    small = cv2.resize(img, (small_w, small_h), interpolation=cv2.INTER_AREA)
    degraded = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)

    # Step 3: JPEG compression artefacts
    quality_pct = max(1, min(100, int(quality * 100)))
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality_pct]
    ok, buf = cv2.imencode(".jpg", degraded, encode_param)
    if not ok:
        return degraded  # fallback if encoding fails
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


# ── Per-modality preprocessors ───────────────────────────────────────────────

def _load_image(path: Union[str, Path]) -> np.ndarray:
    """Load an image from disk and raise a clear error if missing."""
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return img


def _to_3channel(img: np.ndarray) -> np.ndarray:
    """Convert grayscale image to 3-channel BGR if needed."""
    if img.ndim == 2 or img.shape[2] == 1:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img


def preprocess_face(
    img_path: Union[str, Path, np.ndarray],
    target_size: Tuple[int, int] = IMG_SIZE,
) -> np.ndarray:
    """
    Load and resize a face image.
    No quality degradation is applied (images come from cleaned WebFace260M).

    Returns
    -------
    np.ndarray
        BGR image of shape (H, W, 3).
    """
    img = _load_image(img_path) if not isinstance(img_path, np.ndarray) else img_path
    img = _to_3channel(img)
    return cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)


def preprocess_iris(
    img_path: Union[str, Path, np.ndarray],
    target_size: Tuple[int, int] = IMG_SIZE,
    quality: float = IRIS_QUALITY,
    resolution_factor: float = RESOLUTION_FACTOR,
) -> np.ndarray:
    """
    Load, degrade, and resize an iris image per paper §5.1.1.

    Returns
    -------
    np.ndarray
        BGR image of shape (H, W, 3).
    """
    img = _load_image(img_path) if not isinstance(img_path, np.ndarray) else img_path
    img = _to_3channel(img)
    img = simulate_low_quality(img, quality, resolution_factor)
    return cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)


def preprocess_fingerprint(
    img_path: Union[str, Path, np.ndarray],
    target_size: Tuple[int, int] = IMG_SIZE,
    quality: float = FINGERPRINT_QUALITY,
    resolution_factor: float = RESOLUTION_FACTOR,
) -> np.ndarray:
    """
    Load, degrade, and resize a fingerprint image per paper §5.1.1.

    Returns
    -------
    np.ndarray
        BGR image of shape (H, W, 3).
    """
    img = _load_image(img_path) if not isinstance(img_path, np.ndarray) else img_path
    img = _to_3channel(img)
    img = simulate_low_quality(img, quality, resolution_factor)
    return cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)


# ── Utility: BGR → RGB tensor-ready array ────────────────────────────────────

def bgr_to_rgb(img: np.ndarray) -> np.ndarray:
    """Convert OpenCV BGR array to RGB (required before ToTensor)."""
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
