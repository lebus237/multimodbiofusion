#!/usr/bin/env python
"""
evaluate.py
===========
Full evaluation of all trained multimodal biometric fusion models.

Produces:
  1. CMC curves for all methods (saved as PNG).
  2. Formatted Rank-1 / Rank-5 / Rank-10 / mAP results table.
  3. Score-fusion results (rank1 and modality evaluation) from a trained
     feature-fusion model — no extra training required.
  4. Per-method results CSV file.

Usage
-----
# Evaluate ALL checkpoints found under --checkpoint_dir
python scripts/evaluate.py --config config/config.yaml \
    --checkpoint_dir checkpoints/

# Evaluate a single pixel-level checkpoint
python scripts/evaluate.py --config config/config.yaml \
    --model_type  pixel \
    --fusion_type channel \
    --backbone    vgg16 \
    --checkpoint  checkpoints/pixel_channel_vgg16/pixel_channel_vgg16_epoch100.pt

# Evaluate a single feature-level checkpoint (also runs score fusion)
python scripts/evaluate.py --config config/config.yaml \
    --model_type feature \
    --backbone   vgg16 \
    --checkpoint checkpoints/feature_vgg16/feature_vgg16_epoch100.pt
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import yaml

# Allow imports from the project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.dataset import build_dataloaders
from src.evaluation.metrics import (
    compute_cmc_map,
    extract_embeddings,
    plot_cmc_curve,
    print_results_table,
)
from src.models.feature_fusion import FeatureFusionModel
from src.models.pixel_fusion import PixelFusionModel
from src.models.score_fusion import ScoreFusion

# Type alias
ResultsDict = Dict[str, Tuple[np.ndarray, float]]


# ── Argument parsing ──────────────────────────────────────────────────────────


def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate trained multimodal biometric fusion models.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to YAML configuration file.",
    )
    p.add_argument(
        "--checkpoint_dir",
        type=str,
        default=None,
        help=(
            "Root directory containing one sub-directory per trained model "
            "(e.g. checkpoints/pixel_channel_vgg16/, checkpoints/feature_vgg16/). "
            "Evaluates the most-recent .pt file in every sub-directory."
        ),
    )
    p.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to a single checkpoint file (.pt) to evaluate.",
    )
    p.add_argument(
        "--model_type",
        type=str,
        default=None,
        choices=["pixel", "feature"],
        help="Type of model for --checkpoint. Required when --checkpoint is used.",
    )
    p.add_argument(
        "--backbone",
        type=str,
        default=None,
        help="Backbone name (vgg16 | vgg16_bn | resnet50 | densenet169). "
        "Used with --checkpoint.",
    )
    p.add_argument(
        "--fusion_type",
        type=str,
        default=None,
        choices=["channel", "intensity", "spatial"],
        help="Pixel-fusion type. Required when --model_type pixel.",
    )
    p.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Override dataset root directory from config.",
    )
    p.add_argument(
        "--output_dir",
        type=str,
        default="results",
        help="Directory where plots and CSV are saved.",
    )
    p.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Compute device: cuda | cpu.",
    )
    p.add_argument(
        "--score_method",
        type=str,
        default=None,
        choices=["rank1", "modality", "both"],
        help="Score-fusion evaluation method. 'both' runs rank1 and modality.",
    )
    return p.parse_args()


# ── Model loaders ─────────────────────────────────────────────────────────────


def load_pixel_model(
    checkpoint_path: str,
    backbone: str,
    fusion_type: str,
    device: str,
) -> PixelFusionModel:
    """Load a PixelFusionModel from a checkpoint file."""
    model = PixelFusionModel(
        backbone_name=backbone,
        fusion_type=fusion_type,
        pretrained=False,
    )
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def load_feature_model(
    checkpoint_path: str,
    backbone: str,
    device: str,
) -> FeatureFusionModel:
    """Load a FeatureFusionModel from a checkpoint file."""
    model = FeatureFusionModel(
        backbone_name=backbone,
        pretrained=False,
    )
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


# ── Per-model evaluation helpers ──────────────────────────────────────────────


def evaluate_pixel_checkpoint(
    checkpoint_path: str,
    backbone: str,
    fusion_type: str,
    test_loader,
    n_query: int,
    max_rank: int,
    device: str,
) -> Tuple[np.ndarray, float]:
    """
    Evaluate one pixel-fusion checkpoint.

    Returns
    -------
    cmc : np.ndarray [max_rank]
    mAP : float
    """
    print(f"  Loading pixel model: backbone={backbone}  fusion={fusion_type}")
    model = load_pixel_model(checkpoint_path, backbone, fusion_type, device)

    embs, labels = extract_embeddings(model, test_loader, device=device)
    q_emb, q_lbl = embs[:n_query], labels[:n_query]
    g_emb, g_lbl = embs[n_query:], labels[n_query:]

    if len(g_emb) == 0:
        print("    ⚠  Not enough test samples for query/gallery split.")
        return np.zeros(max_rank), 0.0

    cmc, mAP = compute_cmc_map(q_emb, q_lbl, g_emb, g_lbl, max_rank=max_rank)
    return cmc, mAP


def evaluate_feature_checkpoint(
    checkpoint_path: str,
    backbone: str,
    test_loader,
    n_query: int,
    max_rank: int,
    score_methods: list[str],
    device: str,
) -> ResultsDict:
    """
    Evaluate one feature-fusion checkpoint.

    Returns a dict with results for:
      - Feature Fusion (joint)
      - Single-modal: Face / Iris / Fingerprint
      - Score Fusion (rank1) and/or (modality)
    """
    print(f"  Loading feature model: backbone={backbone}")
    model = load_feature_model(checkpoint_path, backbone, device)

    (
        fused_emb,
        face_emb,
        iris_emb,
        fp_emb,
        labels,
    ) = extract_embeddings(
        model, test_loader, device=device, return_all_modalities=True
    )

    results: ResultsDict = {}

    # ── Embedding-based methods ───────────────────────────────────────────────
    for display_name, emb in [
        ("Feature Fusion (joint)", fused_emb),
        ("Single-modal: Face", face_emb),
        ("Single-modal: Iris", iris_emb),
        ("Single-modal: Fingerprint", fp_emb),
    ]:
        q_e = emb[:n_query]
        q_l = labels[:n_query]
        g_e = emb[n_query:]
        g_l = labels[n_query:]
        if len(g_e) == 0:
            continue
        cmc, mAP = compute_cmc_map(q_e, q_l, g_e, g_l, max_rank=max_rank)
        results[display_name] = (cmc, mAP)

    # ── Score-level fusion ────────────────────────────────────────────────────
    q_face = face_emb[:n_query]
    q_iris = iris_emb[:n_query]
    q_fp = fp_emb[:n_query]
    g_face = face_emb[n_query:]
    g_iris = iris_emb[n_query:]
    g_fp = fp_emb[n_query:]
    g_lbl = labels[n_query:]
    q_lbl = labels[:n_query]

    if len(g_lbl) == 0:
        return results

    for method in score_methods:
        sf = ScoreFusion(method=method)
        scores_dict = {
            "face": sf.build_score_matrix(q_face, g_face),
            "iris": sf.build_score_matrix(q_iris, g_iris),
            "fingerprint": sf.build_score_matrix(q_fp, g_fp),
        }
        pred_labels = sf.fuse_with_labels(scores_dict, g_lbl)
        rank1_acc = float(np.mean(pred_labels == q_lbl))

        # Hard-decision fusion gives a definitive rank-1 answer;
        # represent as a flat CMC (rank-N accuracy >= rank-1 for N>1).
        cmc_pseudo = np.full(max_rank, rank1_acc)
        results[f"Score Fusion ({method})"] = (cmc_pseudo, rank1_acc)

    return results


# ── Auto-discovery from checkpoint_dir ────────────────────────────────────────


def _parse_run_name(
    run_name: str,
) -> Optional[Tuple[str, str, Optional[str]]]:
    """
    Parse a run directory name back into (model_type, backbone, fusion_type).

    Expected formats:
      pixel_{fusion_type}_{backbone}  →  ('pixel', backbone, fusion_type)
      feature_{backbone}              →  ('feature', backbone, None)

    Returns None if the name cannot be parsed.
    """
    if run_name.startswith("pixel_"):
        rest = run_name[len("pixel_") :]
        for ft in ("channel", "intensity", "spatial"):
            if rest.startswith(ft + "_"):
                backbone = rest[len(ft) + 1 :]
                return "pixel", backbone, ft
        return None  # unrecognised
    elif run_name.startswith("feature_"):
        backbone = run_name[len("feature_") :]
        return "feature", backbone, None
    return None


def evaluate_all_in_dir(
    checkpoint_dir: Path,
    test_loader,
    n_query: int,
    max_rank: int,
    score_methods: list[str],
    device: str,
) -> ResultsDict:
    """
    Scan checkpoint_dir for model sub-directories, evaluate the latest
    checkpoint in each, and return the merged results dict.
    """
    all_results: ResultsDict = {}

    if not checkpoint_dir.exists():
        print(f"  ⚠  checkpoint_dir does not exist: {checkpoint_dir}")
        return all_results

    sub_dirs = sorted(d for d in checkpoint_dir.iterdir() if d.is_dir())
    if not sub_dirs:
        print(f"  ⚠  No sub-directories found in {checkpoint_dir}")
        return all_results

    for model_dir in sub_dirs:
        parsed = _parse_run_name(model_dir.name)
        if parsed is None:
            print(f"  ⚠  Skipping unrecognised directory: {model_dir.name}")
            continue

        model_type, backbone, fusion_type = parsed
        checkpoints = sorted(model_dir.glob("*.pt"))
        if not checkpoints:
            print(f"  ⚠  No .pt files found in {model_dir}")
            continue

        latest = checkpoints[-1]
        print(f"\n  ── {model_dir.name} ──")
        print(f"     Checkpoint: {latest.name}")

        try:
            if model_type == "pixel":
                display = f"Pixel ({fusion_type}, {backbone})"
                cmc, mAP = evaluate_pixel_checkpoint(
                    str(latest),
                    backbone,
                    fusion_type,
                    test_loader,
                    n_query,
                    max_rank,
                    device,
                )
                all_results[display] = (cmc, mAP)

            else:  # feature
                results = evaluate_feature_checkpoint(
                    str(latest),
                    backbone,
                    test_loader,
                    n_query,
                    max_rank,
                    score_methods,
                    device,
                )
                # Prefix with backbone to avoid key collisions when multiple
                # backbones are evaluated
                for key, val in results.items():
                    prefixed = f"{key} [{backbone}]"
                    all_results[prefixed] = val

        except Exception as exc:
            print(f"  ✗ Failed to evaluate {model_dir.name}: {exc}")

    return all_results


# ── CSV export ────────────────────────────────────────────────────────────────


def save_csv(results: ResultsDict, csv_path: Path) -> None:
    """Write results to a CSV file."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["Method", "Rank-1 (%)", "Rank-5 (%)", "Rank-10 (%)", "mAP (%)"]
        )
        for name, (cmc, mAP) in results.items():
            r1 = f"{cmc[0] * 100:.4f}"
            r5 = f"{cmc[4] * 100:.4f}" if len(cmc) > 4 else "N/A"
            r10 = f"{cmc[9] * 100:.4f}" if len(cmc) > 9 else "N/A"
            writer.writerow([name, r1, r5, r10, f"{mAP * 100:.4f}"])
    print(f"  ✓ Results CSV  → {csv_path}")


# ── Main ──────────────────────────────────────────────────────────────────────


def main():
    args = parse_args()

    # ── Load configuration ────────────────────────────────────────────────────
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"  ERROR: Config not found: {config_path}")
        sys.exit(1)

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    data_dir = args.data_dir or cfg["data"]["root_dir"]
    max_rank = cfg["evaluation"]["max_rank"]
    query_ratio = cfg["evaluation"].get("query_ratio", 0.20)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Resolve score-fusion methods to run
    cfg_score_method = cfg["score_fusion"].get("method", "rank1")
    if args.score_method == "both":
        score_methods = ["rank1", "modality"]
    elif args.score_method:
        score_methods = [args.score_method]
    else:
        score_methods = [cfg_score_method]

    print(f"\n{'=' * 62}")
    print(f"  Multimodal Biometric Fusion — Evaluation")
    print(f"  Data dir     : {data_dir}")
    print(f"  Output dir   : {output_dir}")
    print(f"  Max rank     : {max_rank}")
    print(f"  Query ratio  : {query_ratio:.0%}")
    print(f"  Score method : {score_methods}")
    print(f"  Device       : {args.device}")
    print(f"{'=' * 62}\n")

    # ── Build test DataLoader ─────────────────────────────────────────────────
    print("  Building test DataLoader…")
    try:
        _, _, test_loader, n_classes = build_dataloaders(
            root_dir=data_dir,
            batch_size=cfg["training"].get("batch_size", 64),
            num_workers=cfg["training"].get("num_workers", 4),
            pin_memory=cfg["training"].get("pin_memory", True),
            train_ratio=cfg["data"]["train_split"],
            val_ratio=cfg["data"]["val_split"],
            seed=cfg["training"].get("seed", 42),
        )
    except RuntimeError as e:
        print(f"\n  ERROR: {e}")
        print(
            "  Hint: Run  python scripts/prepare_dataset.py --demo  "
            "to create a synthetic dataset."
        )
        sys.exit(1)

    n_test = len(test_loader.dataset)
    n_query = max(1, int(n_test * query_ratio))
    print(f"  Test set: {n_test} samples | {n_classes} identities")
    print(f"  Queries : {n_query}  |  Gallery: {n_test - n_query}\n")

    # ── Run evaluations ───────────────────────────────────────────────────────
    all_results: ResultsDict = {}

    # Mode A: evaluate a single checkpoint
    if args.checkpoint:
        ckpt_path = args.checkpoint
        backbone = args.backbone or cfg["model"]["backbone"]
        model_type = args.model_type

        if model_type is None:
            print(
                "  ERROR: --model_type (pixel | feature) is required "
                "when --checkpoint is specified."
            )
            sys.exit(1)

        print(f"  Evaluating single checkpoint: {ckpt_path}")

        if model_type == "pixel":
            fusion_type = args.fusion_type or cfg["model"]["pixel_fusion_type"]
            display = f"Pixel ({fusion_type}, {backbone})"
            cmc, mAP = evaluate_pixel_checkpoint(
                ckpt_path,
                backbone,
                fusion_type,
                test_loader,
                n_query,
                max_rank,
                args.device,
            )
            all_results[display] = (cmc, mAP)

        else:  # feature
            results = evaluate_feature_checkpoint(
                ckpt_path,
                backbone,
                test_loader,
                n_query,
                max_rank,
                score_methods,
                args.device,
            )
            all_results.update(results)

    # Mode B: auto-discover all checkpoints in a directory
    elif args.checkpoint_dir:
        all_results = evaluate_all_in_dir(
            checkpoint_dir=Path(args.checkpoint_dir),
            test_loader=test_loader,
            n_query=n_query,
            max_rank=max_rank,
            score_methods=score_methods,
            device=args.device,
        )

    else:
        # Mode C: try the default checkpoint directory from config
        default_dir = Path(cfg["training"]["save_dir"])
        if default_dir.exists():
            print(f"  No --checkpoint or --checkpoint_dir given.")
            print(f"  Scanning default save_dir: {default_dir}\n")
            all_results = evaluate_all_in_dir(
                checkpoint_dir=default_dir,
                test_loader=test_loader,
                n_query=n_query,
                max_rank=max_rank,
                score_methods=score_methods,
                device=args.device,
            )
        else:
            print(
                "  ERROR: Please specify --checkpoint, --checkpoint_dir, "
                "or ensure config.training.save_dir exists."
            )
            sys.exit(1)

    # ── Nothing found ─────────────────────────────────────────────────────────
    if not all_results:
        print("\n  No results were produced. Check checkpoint paths and try again.")
        sys.exit(0)

    # ── Print results table ───────────────────────────────────────────────────
    print()
    print_results_table(all_results)

    # ── CMC curve plot ────────────────────────────────────────────────────────
    cmc_path = str(output_dir / "cmc_curves.png")
    fig = plot_cmc_curve(
        all_results,
        save_path=cmc_path,
        title="CMC Curve — Multimodal Biometric Fusion",
    )

    # ── Save CSV ──────────────────────────────────────────────────────────────
    csv_path = output_dir / "results.csv"
    save_csv(all_results, csv_path)

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n  ✓ CMC plot saved → {cmc_path}")
    print(f"  ✓ CSV saved      → {csv_path}")

    # Print the best-performing method
    best_name = max(all_results, key=lambda k: all_results[k][0][0])
    best_r1 = all_results[best_name][0][0] * 100
    best_map = all_results[best_name][1] * 100
    print(
        f"\n  Best Rank-1: {best_name}\n"
        f"             R1={best_r1:.2f}%  mAP={best_map:.2f}%\n"
    )


if __name__ == "__main__":
    main()
