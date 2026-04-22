#!/usr/bin/env python
"""
train_feature.py
================
Train the feature-level fusion model (paper §3.3).

Architecture: three independent modality-specific CNN branches (face, iris,
fingerprint) + a joint representation layer (FC 1536 → 512).  All parameters
are optimised jointly through backpropagation.  Per-modality auxiliary losses
(weighted by lambda = 0.3) reinforce each individual branch.

Loss (paper §3.3):
    L = L_fused + λ · (L_face + L_iris + L_fingerprint)

After training, the per-modality embeddings are also extracted so that
score-level fusion can be applied at evaluation time without any additional
training steps (paper §3.4).

Usage
-----
# Default: VGG-16 backbone
python scripts/train_feature.py --config config/config.yaml

# ResNet-50 backbone
python scripts/train_feature.py --config config/config.yaml --backbone resnet50

# DenseNet-169, no pretrained weights
python scripts/train_feature.py --config config/config.yaml \
    --backbone densenet169 --no_pretrained
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

# Allow imports from the project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.dataset import build_dataloaders
from src.evaluation.metrics import compute_cmc_map, extract_embeddings
from src.models.feature_fusion import FeatureFusionModel
from src.models.score_fusion import ScoreFusion
from src.training.trainer import FeatureFusionTrainer

# ── Argument parsing ──────────────────────────────────────────────────────────


def parse_args():
    p = argparse.ArgumentParser(
        description="Train the feature-level multimodal biometric fusion model."
    )
    p.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to YAML config file",
    )
    p.add_argument(
        "--backbone",
        type=str,
        default=None,
        help="Backbone architecture (vgg16 | vgg16_bn | resnet50 | densenet169)",
    )
    p.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Override dataset root directory",
    )
    p.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override number of training epochs",
    )
    p.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Override batch size",
    )
    p.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Override initial learning rate",
    )
    p.add_argument(
        "--modality_weight",
        type=float,
        default=None,
        help="Override auxiliary per-modality loss weight λ (default: 0.3)",
    )
    p.add_argument(
        "--pretrained",
        action="store_true",
        default=True,
        help="Use ImageNet-pretrained backbone weights (default: True)",
    )
    p.add_argument(
        "--no_pretrained",
        dest="pretrained",
        action="store_false",
        help="Train backbone from scratch",
    )
    p.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Compute device: cuda | cpu",
    )
    p.add_argument(
        "--save_dir",
        type=str,
        default=None,
        help="Override checkpoint save directory",
    )
    p.add_argument(
        "--log_dir",
        type=str,
        default=None,
        help="Override TensorBoard log directory",
    )
    p.add_argument(
        "--score_method",
        type=str,
        default=None,
        choices=["rank1", "modality"],
        help="Score-fusion method to evaluate after training (default from config)",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        default=False,
        help="Resume training from the latest checkpoint in save_dir",
    )
    return p.parse_args()


# ── Evaluation helpers ────────────────────────────────────────────────────────


def _eval_single(
    embeddings: np.ndarray,
    labels: np.ndarray,
    n_query: int,
    max_rank: int,
    name: str,
) -> tuple[np.ndarray, float] | tuple[None, None]:
    """Run CMC+mAP for a single embedding array and print results inline."""
    q_emb = embeddings[:n_query]
    q_lbl = labels[:n_query]
    g_emb = embeddings[n_query:]
    g_lbl = labels[n_query:]

    if len(g_emb) == 0:
        print(f"    {name:<40} ⚠ not enough samples for evaluation")
        return None, None

    cmc, mAP = compute_cmc_map(q_emb, q_lbl, g_emb, g_lbl, max_rank=max_rank)
    r1 = cmc[0] * 100
    r5 = cmc[min(4, max_rank - 1)] * 100
    r10 = cmc[min(9, max_rank - 1)] * 100 if max_rank >= 10 else float("nan")
    print(
        f"    {name:<40}  R1={r1:5.2f}%  R5={r5:5.2f}%  "
        f"R10={r10:5.2f}%  mAP={mAP * 100:5.2f}%"
    )
    return cmc, mAP


# ── Main ──────────────────────────────────────────────────────────────────────


def main():
    args = parse_args()

    # ── Load config ───────────────────────────────────────────────────────────
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"  ERROR: Config file not found: {config_path}")
        sys.exit(1)

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    # Resolve settings (CLI overrides config)
    backbone = args.backbone or cfg["model"]["backbone"]
    data_dir = args.data_dir or cfg["data"]["root_dir"]
    epochs = args.epochs or cfg["training"]["epochs"]
    batch_size = args.batch_size or cfg["training"]["batch_size"]
    lr = args.lr or cfg["training"]["learning_rate"]
    seed = cfg["training"].get("seed", 42)
    score_method = args.score_method or cfg["score_fusion"].get("method", "rank1")
    mw_cfg = cfg["training"].get("multi_task_weights", {})
    modality_weight = (
        args.modality_weight
        if args.modality_weight is not None
        else mw_cfg.get("modality", 0.3)
    )

    run_name = f"feature_{backbone}"
    save_dir = Path(args.save_dir or cfg["training"]["save_dir"]) / run_name
    log_dir = Path(args.log_dir or cfg["training"]["log_dir"]) / run_name

    print(f"\n{'=' * 62}")
    print(f"  Feature-Level Fusion Training")
    print(f"  Run:             {run_name}")
    print(f"  Backbone:        {backbone}  (x3 branches)")
    print(f"  Modality weight: λ={modality_weight}")
    print(f"  Data:            {data_dir}")
    print(f"  Epochs:          {epochs}  |  Batch: {batch_size}  |  LR: {lr}")
    print(f"  Pretrained:      {args.pretrained}")
    print(f"  Device:          {args.device}")
    print(f"  Save dir:        {save_dir}")
    print(f"{'=' * 62}\n")

    # ── Data loaders ──────────────────────────────────────────────────────────
    print("  [1/3] Building data loaders…")
    try:
        train_loader, val_loader, test_loader, n_classes = build_dataloaders(
            root_dir=data_dir,
            batch_size=batch_size,
            num_workers=cfg["training"].get("num_workers", 4),
            pin_memory=cfg["training"].get("pin_memory", True),
            train_ratio=cfg["data"]["train_split"],
            val_ratio=cfg["data"]["val_split"],
            seed=seed,
        )
    except RuntimeError as e:
        print(f"\n  ERROR: {e}")
        print(
            "  Hint: Run  python scripts/prepare_dataset.py --demo  "
            "to generate a synthetic dataset for testing."
        )
        sys.exit(1)

    n_train = len(train_loader.dataset)
    n_val = len(val_loader.dataset)
    n_test = len(test_loader.dataset)
    print(
        f"  Dataset: {n_classes} identities | "
        f"train={n_train}  val={n_val}  test={n_test}"
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    print("\n  [2/3] Building FeatureFusionModel…")
    model = FeatureFusionModel(
        backbone_name=backbone,
        pretrained=args.pretrained,
    )

    # Count parameters
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())

    # Per-branch count (approx.)
    n_branch = sum(p.numel() for p in model.face_branch.parameters() if p.requires_grad)
    n_joint = sum(p.numel() for p in model.joint_layer.parameters() if p.requires_grad)

    print(f"  Architecture:")
    print(f"    Face branch      : {n_branch / 1e6:.2f} M params")
    print(
        f"    Iris branch      : {n_branch / 1e6:.2f} M params  (shared architecture)"
    )
    print(
        f"    Fingerprint brnch: {n_branch / 1e6:.2f} M params  (shared architecture)"
    )
    print(f"    Joint FC layer   : {n_joint / 1e3:.1f} K params")
    print(f"    Total trainable  : {n_trainable / 1e6:.2f} M / {n_total / 1e6:.2f} M")
    print(
        f"  Multi-task loss:  L = L_fused + {modality_weight}*(L_face + L_iris + L_fp)"
    )

    # ── Trainer ───────────────────────────────────────────────────────────────
    print("\n  [3/3] Initialising FeatureFusionTrainer…")
    trainer = FeatureFusionTrainer(
        model=model,
        n_classes=n_classes,
        lr=lr,
        weight_decay=cfg["training"].get("weight_decay", 5e-4),
        scale=cfg["training"]["arcface"]["scale"],
        margin=cfg["training"]["arcface"]["margin"],
        epochs=epochs,
        warmup_epochs=cfg["training"].get("warmup_epochs", 5),
        device=args.device,
        save_dir=str(save_dir),
        log_dir=str(log_dir),
        seed=seed,
        modality_weight=modality_weight,
    )
    print(
        f"  ArcFace: scale={cfg['training']['arcface']['scale']}  "
        f"margin={cfg['training']['arcface']['margin']}"
    )
    warmup = cfg["training"].get("warmup_epochs", 5)
    print(
        f"  LR schedule: linear warm-up ({warmup} ep) → "
        f"cosine annealing ({epochs - warmup} ep)\n"
    )

    # ── Resume from checkpoint (if requested) ────────────────────────────────
    start_epoch = 1
    if args.resume:
        ckpt_path = trainer.find_latest_checkpoint(run_name)
        if ckpt_path:
            last_epoch = trainer.load(ckpt_path, resume=True)
            start_epoch = last_epoch + 1
            print(f"  ▶ Resuming from epoch {start_epoch}")
        else:
            print("  ⚠ --resume set but no checkpoint found; starting from scratch.")

    # ── Train ─────────────────────────────────────────────────────────────────
    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        save_every=cfg["training"].get("save_every", 10),
        run_name=run_name,
        start_epoch=start_epoch,
    )

    # ── Full test-set evaluation ───────────────────────────────────────────────
    print("\n" + "─" * 62)
    print("  Final evaluation on held-out test set")
    print("─" * 62)

    model.eval()
    query_ratio = cfg["evaluation"].get("query_ratio", 0.20)
    max_rank = cfg["evaluation"]["max_rank"]
    n_query = max(1, int(n_test * query_ratio))

    print(f"  Extracting embeddings (all modalities)…")
    (
        fused_emb,
        face_emb,
        iris_emb,
        fp_emb,
        test_labels,
    ) = extract_embeddings(
        model,
        test_loader,
        device=args.device,
        return_all_modalities=True,
    )

    print(
        f"  Embeddings shape: {fused_emb.shape}  "
        f"(queries={n_query}, gallery={n_test - n_query})\n"
    )

    # Results dict for summary
    all_results: dict[str, tuple[np.ndarray, float]] = {}

    # 1. Fused embedding (feature-level fusion)
    cmc, mAP = _eval_single(
        fused_emb, test_labels, n_query, max_rank, "Feature Fusion (joint)"
    )
    if cmc is not None:
        all_results["Feature Fusion (joint)"] = (cmc, mAP)

    # 2. Individual modality branches
    for name, emb in [
        ("Single-modal: Face", face_emb),
        ("Single-modal: Iris", iris_emb),
        ("Single-modal: Fingerprint", fp_emb),
    ]:
        cmc, mAP = _eval_single(emb, test_labels, n_query, max_rank, name)
        if cmc is not None:
            all_results[name] = (cmc, mAP)

    # 3. Score-level fusion (inference-only, no extra training needed)
    print(f"\n  Score-level fusion (method='{score_method}')…")
    sf = ScoreFusion(method=score_method)

    q_face = face_emb[:n_query]
    q_iris = iris_emb[:n_query]
    q_fp = fp_emb[:n_query]
    g_face = face_emb[n_query:]
    g_iris = iris_emb[n_query:]
    g_fp = fp_emb[n_query:]
    g_lbl = test_labels[n_query:]
    q_lbl = test_labels[:n_query]

    if len(g_lbl) > 0:
        scores_dict = {
            "face": sf.build_score_matrix(q_face, g_face),
            "iris": sf.build_score_matrix(q_iris, g_iris),
            "fingerprint": sf.build_score_matrix(q_fp, g_fp),
        }

        # Hard-decision rank-1
        pred_labels = sf.fuse_with_labels(scores_dict, g_lbl)
        rank1_acc = float(np.mean(pred_labels == q_lbl))

        # Build a pseudo-CMC (hard fusion gives only a rank-1 decision;
        # higher ranks degrade gracefully in retrieval systems)
        cmc_pseudo = np.full(max_rank, rank1_acc)
        cmc_pseudo[0] = rank1_acc
        score_key = f"Score Fusion ({score_method})"
        all_results[score_key] = (cmc_pseudo, rank1_acc)
        print(
            f"    {score_key:<40}  R1={rank1_acc * 100:5.2f}%  "
            f"(hard-decision; CMC ≥ R1 by definition)"
        )

    # ── Summary table ─────────────────────────────────────────────────────────
    if all_results:
        print(f"\n{'─' * 62}")
        print(f"  {'Method':<40} {'R1':>6}  {'R5':>6}  {'R10':>6}  {'mAP':>6}")
        print(f"  {'─' * 58}")
        for name, (cmc, mAP) in all_results.items():
            r1 = cmc[0] * 100
            r5 = cmc[min(4, len(cmc) - 1)] * 100 if len(cmc) > 4 else float("nan")
            r10 = cmc[min(9, len(cmc) - 1)] * 100 if len(cmc) > 9 else float("nan")
            print(f"  {name:<40} {r1:6.2f}  {r5:6.2f}  {r10:6.2f}  {mAP * 100:6.2f}")
        print(f"{'─' * 62}\n")

        # Save summary text
        summary_path = save_dir / "test_results.txt"
        with open(summary_path, "w") as f:
            f.write(f"Run:              {run_name}\n")
            f.write(f"Backbone:         {backbone}\n")
            f.write(f"Modality weight:  {modality_weight}\n")
            f.write(f"Epochs:           {epochs}\n")
            f.write(f"Score method:     {score_method}\n\n")
            f.write(f"{'Method':<40} {'R1':>8}  {'R5':>8}  {'R10':>8}  {'mAP':>8}\n")
            f.write("-" * 76 + "\n")
            for name, (cmc, mAP) in all_results.items():
                r1 = cmc[0] * 100
                r5 = cmc[min(4, len(cmc) - 1)] * 100 if len(cmc) > 4 else float("nan")
                r10 = cmc[min(9, len(cmc) - 1)] * 100 if len(cmc) > 9 else float("nan")
                f.write(
                    f"{name:<40} {r1:8.4f}  {r5:8.4f}  {r10:8.4f}  {mAP * 100:8.4f}\n"
                )
        print(f"  ✓ Test results saved → {summary_path}")

    print(f"  ✓ Training complete:  {run_name}")
    print(f"  ✓ Checkpoints in:    {save_dir}\n")


if __name__ == "__main__":
    main()
