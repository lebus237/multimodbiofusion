#!/usr/bin/env python
"""
train_pixel.py
==============
Train a pixel-level fusion model (paper §3.2).

Supports all three pixel fusion types (channel, intensity, spatial) and all
four backbone architectures (vgg16, vgg16_bn, resnet50, densenet169).

Usage
-----
# Channel fusion with VGG-16 (default)
python scripts/train_pixel.py --config config/config.yaml

# Intensity fusion with ResNet-50
python scripts/train_pixel.py --config config/config.yaml \
    --backbone resnet50 --fusion_type intensity

# Spatial fusion with DenseNet-169
python scripts/train_pixel.py --config config/config.yaml \
    --backbone densenet169 --fusion_type spatial
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.dataset import build_dataloaders
from src.evaluation.metrics import compute_cmc_map, extract_embeddings
from src.models.pixel_fusion import PixelFusionModel
from src.training.trainer import Trainer

# ── Argument parsing ──────────────────────────────────────────────────────────


def parse_args():
    p = argparse.ArgumentParser(description="Train pixel-level biometric fusion model.")
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
        help="Override backbone (vgg16 | vgg16_bn | resnet50 | densenet169)",
    )
    p.add_argument(
        "--fusion_type",
        type=str,
        default=None,
        help="Override fusion type (channel | intensity | spatial)",
    )
    p.add_argument(
        "--data_dir", type=str, default=None, help="Override data root directory"
    )
    p.add_argument(
        "--epochs", type=int, default=None, help="Override number of training epochs"
    )
    p.add_argument("--batch_size", type=int, default=None, help="Override batch size")
    p.add_argument("--lr", type=float, default=None, help="Override learning rate")
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
        "--device", type=str, default="cuda", help="Device to use: cuda | cpu"
    )
    p.add_argument(
        "--save_dir", type=str, default=None, help="Override checkpoint save directory"
    )
    p.add_argument(
        "--log_dir", type=str, default=None, help="Override TensorBoard log directory"
    )
    return p.parse_args()


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

    # Apply CLI overrides (CLI takes precedence over config)
    backbone = args.backbone or cfg["model"]["backbone"]
    fusion_type = args.fusion_type or cfg["model"]["pixel_fusion_type"]
    data_dir = args.data_dir or cfg["data"]["root_dir"]
    epochs = args.epochs or cfg["training"]["epochs"]
    batch_size = args.batch_size or cfg["training"]["batch_size"]
    lr = args.lr or cfg["training"]["learning_rate"]
    seed = cfg["training"].get("seed", 42)

    run_name = f"pixel_{fusion_type}_{backbone}"
    save_dir = Path(args.save_dir or cfg["training"]["save_dir"]) / run_name
    log_dir = Path(args.log_dir or cfg["training"]["log_dir"]) / run_name

    print(f"\n{'=' * 62}")
    print(f"  Pixel-Level Fusion Training")
    print(f"  Run:        {run_name}")
    print(f"  Backbone:   {backbone}")
    print(f"  Fusion:     {fusion_type}")
    print(f"  Data:       {data_dir}")
    print(f"  Epochs:     {epochs}  |  Batch: {batch_size}  |  LR: {lr}")
    print(f"  Pretrained: {args.pretrained}")
    print(f"  Device:     {args.device}")
    print(f"  Save dir:   {save_dir}")
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
        print(f"\n  ERROR loading dataset: {e}")
        print(
            "  Hint: Run scripts/prepare_dataset.py --demo first to create a test dataset."
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
    print("\n  [2/3] Building model…")
    model = PixelFusionModel(
        backbone_name=backbone,
        fusion_type=fusion_type,
        pretrained=args.pretrained,
    )
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_params_all = sum(p.numel() for p in model.parameters())
    print(f"  Model:   {model}")
    print(
        f"  Params:  {n_params / 1e6:.2f} M trainable / {n_params_all / 1e6:.2f} M total"
    )

    # ── Trainer ───────────────────────────────────────────────────────────────
    print("\n  [3/3] Initialising trainer…")
    trainer = Trainer(
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
    )
    print(
        f"  ArcFace: scale={cfg['training']['arcface']['scale']}  "
        f"margin={cfg['training']['arcface']['margin']}"
    )
    print(
        f"  Warmup:  {cfg['training'].get('warmup_epochs', 5)} epochs → "
        f"cosine annealing for remaining {epochs - cfg['training'].get('warmup_epochs', 5)} epochs\n"
    )

    # ── Train ─────────────────────────────────────────────────────────────────
    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        save_every=cfg["training"].get("save_every", 10),
        run_name=run_name,
    )

    # ── Final evaluation on test set ──────────────────────────────────────────
    print("\n" + "─" * 50)
    print("  Final evaluation on held-out test set…")
    print("─" * 50)

    model.eval()
    test_embs, test_labels = extract_embeddings(
        model,
        test_loader,
        device=args.device,
    )

    query_ratio = cfg["evaluation"].get("query_ratio", 0.20)
    max_rank = cfg["evaluation"]["max_rank"]
    n_query = max(1, int(n_test * query_ratio))

    q_emb = test_embs[:n_query]
    q_lbl = test_labels[:n_query]
    g_emb = test_embs[n_query:]
    g_lbl = test_labels[n_query:]

    if len(g_emb) == 0:
        print(
            "  ⚠  Not enough test samples for query/gallery split. "
            "Increase dataset size or reduce query_ratio."
        )
    else:
        cmc, mAP = compute_cmc_map(q_emb, q_lbl, g_emb, g_lbl, max_rank=max_rank)
        print(f"\n  {'Metric':<12}{'Value':>10}")
        print(f"  {'─' * 22}")
        print(f"  {'Rank-1':<12}{cmc[0] * 100:>9.2f}%")
        if max_rank >= 5:
            print(f"  {'Rank-5':<12}{cmc[4] * 100:>9.2f}%")
        if max_rank >= 10:
            print(f"  {'Rank-10':<12}{cmc[9] * 100:>9.2f}%")
        print(f"  {'mAP':<12}{mAP * 100:>9.2f}%")
        print(f"  {'─' * 22}\n")

        # Save test metrics to a simple text summary
        summary_path = save_dir / "test_results.txt"
        with open(summary_path, "w") as f:
            f.write(f"Run:         {run_name}\n")
            f.write(f"Backbone:    {backbone}\n")
            f.write(f"Fusion type: {fusion_type}\n")
            f.write(f"Epochs:      {epochs}\n")
            f.write(f"Rank-1:      {cmc[0] * 100:.4f}%\n")
            if max_rank >= 5:
                f.write(f"Rank-5:      {cmc[4] * 100:.4f}%\n")
            if max_rank >= 10:
                f.write(f"Rank-10:     {cmc[9] * 100:.4f}%\n")
            f.write(f"mAP:         {mAP * 100:.4f}%\n")
        print(f"  ✓ Test results saved → {summary_path}")

    print(f"\n  ✓ Training complete: {run_name}")
    print(f"  ✓ Checkpoints in:   {save_dir}\n")


if __name__ == "__main__":
    main()
