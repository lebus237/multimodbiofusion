"""
metrics.py
==========
Evaluation metrics for multimodal biometric retrieval.

Functions:
    extract_embeddings   — run a model over a DataLoader, collect embeddings.
    compute_cmc_map      — CMC curve and mAP from embedding arrays.
    plot_cmc_curve       — matplotlib CMC curve figure.
    print_results_table  — pretty-print Rank-1/5/10 + mAP table.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader
from tqdm import tqdm

# ── Embedding extraction ──────────────────────────────────────────────────────


@torch.no_grad()
def extract_embeddings(
    model: nn.Module,
    loader: DataLoader,
    device: str = "cuda",
    return_all_modalities: bool = False,
) -> Tuple[np.ndarray, ...]:
    """
    Run ``model`` in eval mode over ``loader`` and collect embeddings.

    Parameters
    ----------
    model   : nn.Module
        The fusion model.  Can be PixelFusionModel or FeatureFusionModel.
    loader  : DataLoader
        DataLoader yielding (face, iris, fp, labels) batches.
    device  : str
    return_all_modalities : bool
        If True and model is FeatureFusionModel, also return per-modality
        embeddings via ``model.encode_modalities()``.

    Returns
    -------
    If return_all_modalities is False:
        (embeddings [N, D], labels [N])
    If return_all_modalities is True:
        (fused [N,D], face [N,D], iris [N,D], fp [N,D], labels [N])
    """
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    model = model.to(dev)
    model.eval()

    all_fused, all_face, all_iris, all_fp, all_labels = [], [], [], [], []

    for face, iris, fp, labels in tqdm(
        loader, desc="Extracting embeddings", leave=False
    ):
        face = face.to(dev)
        iris = iris.to(dev)
        fp = fp.to(dev)

        if return_all_modalities and hasattr(model, "encode_modalities"):
            fused, face_feat, iris_feat, fp_feat = model.encode_modalities(
                face, iris, fp
            )
            all_face.append(face_feat.cpu().numpy())
            all_iris.append(iris_feat.cpu().numpy())
            all_fp.append(fp_feat.cpu().numpy())
        else:
            fused = model(face, iris, fp)
            # If model returns a tuple (training mode artefact), take first element
            if isinstance(fused, tuple):
                fused = fused[0]

        all_fused.append(fused.cpu().numpy())
        all_labels.append(labels.numpy())

    fused_arr = np.concatenate(all_fused, axis=0)
    labels_arr = np.concatenate(all_labels, axis=0)

    if return_all_modalities and all_face:
        return (
            fused_arr,
            np.concatenate(all_face, axis=0),
            np.concatenate(all_iris, axis=0),
            np.concatenate(all_fp, axis=0),
            labels_arr,
        )
    return fused_arr, labels_arr


# ── CMC + mAP ─────────────────────────────────────────────────────────────────


def compute_cmc_map(
    query_embeddings: np.ndarray,
    query_labels: np.ndarray,
    gallery_embeddings: np.ndarray,
    gallery_labels: np.ndarray,
    max_rank: int = 10,
) -> Tuple[np.ndarray, float]:
    """
    Compute CMC curve and mAP for open-set biometric retrieval.

    Parameters
    ----------
    query_embeddings   : np.ndarray  [N_q, D]
    query_labels       : np.ndarray  [N_q]
    gallery_embeddings : np.ndarray  [N_g, D]
    gallery_labels     : np.ndarray  [N_g]
    max_rank           : int

    Returns
    -------
    cmc : np.ndarray  [max_rank]  — cumulative Rank-N accuracy
    mAP : float                   — mean average precision
    """
    # Cosine similarity matrix [N_q, N_g]
    q_norm = query_embeddings / (
        np.linalg.norm(query_embeddings, axis=1, keepdims=True) + 1e-12
    )
    g_norm = gallery_embeddings / (
        np.linalg.norm(gallery_embeddings, axis=1, keepdims=True) + 1e-12
    )
    sim_matrix = q_norm @ g_norm.T  # [N_q, N_g]

    # Sort gallery by descending similarity for each query
    sorted_idx = np.argsort(-sim_matrix, axis=1)  # [N_q, N_g]

    cmc = np.zeros(max_rank)
    ap_scores = []

    for q in range(len(query_labels)):
        q_label = query_labels[q]
        ranked_labels = gallery_labels[sorted_idx[q]]  # [N_g]
        matches = (ranked_labels == q_label).astype(int)

        # CMC: first correct match
        for r in range(min(max_rank, len(matches))):
            if matches[r] == 1:
                cmc[r:] += 1
                break

        # AP
        if matches.sum() > 0:
            ranked_sims = sim_matrix[q, sorted_idx[q]]
            ap_scores.append(average_precision_score(matches, ranked_sims))

    cmc = cmc / len(query_labels)
    mAP = float(np.mean(ap_scores)) if ap_scores else 0.0
    return cmc, mAP


# ── Plotting ──────────────────────────────────────────────────────────────────


def plot_cmc_curve(
    results: Dict[str, Tuple[np.ndarray, float]],
    save_path: Optional[str] = None,
    title: str = "CMC Curve — Multimodal Biometric Fusion",
) -> plt.Figure:
    """
    Plot CMC curves for multiple methods.

    Parameters
    ----------
    results : dict
        Maps method name → (cmc array, mAP).
    save_path : str, optional
        If provided, save the figure to this path.
    title : str

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=(9, 6))

    colors = plt.cm.tab10(np.linspace(0, 1, max(len(results), 1)))

    for (name, (cmc, mAP)), color in zip(results.items(), colors):
        ranks = np.arange(1, len(cmc) + 1)
        label = f"{name} (mAP={mAP * 100:.1f}%, R1={cmc[0] * 100:.1f}%)"
        ax.plot(ranks, cmc * 100, marker="o", markersize=4, label=label, color=color)

    ax.set_xlabel("Rank", fontsize=13)
    ax.set_ylabel("Recognition Rate (%)", fontsize=13)
    ax.set_title(title, fontsize=14)
    ax.set_xlim(1, len(next(iter(results.values()))[0]))
    ax.set_ylim(0, 102)
    ax.grid(True, alpha=0.35)
    ax.legend(fontsize=10)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
        print(f"  ✓ CMC curve saved → {save_path}")

    return fig


# ── Pretty-print table ────────────────────────────────────────────────────────


def print_results_table(results: Dict[str, Tuple[np.ndarray, float]]) -> None:
    """
    Print a formatted table of Rank-1, Rank-5, Rank-10 and mAP.

    Parameters
    ----------
    results : dict
        Maps method name → (cmc array [max_rank], mAP).
    """
    header = f"{'Method':<35}{'Rank-1':>8}{'Rank-5':>8}{'Rank-10':>9}{'mAP':>8}"
    sep = "-" * len(header)
    print(f"\n{sep}\n{header}\n{sep}")

    for name, (cmc, mAP) in results.items():
        r1 = cmc[0] * 100 if len(cmc) > 0 else 0.0
        r5 = cmc[4] * 100 if len(cmc) > 4 else 0.0
        r10 = cmc[9] * 100 if len(cmc) > 9 else 0.0
        print(f"{name:<35}{r1:>7.2f}%{r5:>7.2f}%{r10:>8.2f}%{mAP * 100:>7.2f}%")
    print(sep + "\n")
