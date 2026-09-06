"""Train the single-object detector from a terminal.

Mirrors the notebook experiment for people who would rather not run one, and
gives the long training run somewhere to live where progress stays visible.

    python scripts/train_detector.py --epochs 12
    python scripts/train_detector.py --fast-dev-run
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from detector import (DetectionLoss, SingleObjectDetector,  # noqa: E402
                      SyntheticDetectionDataset, evaluate)
from synthdata import load_object_sprites  # noqa: E402

# Disjoint seed ranges, spaced so splits can grow without ever colliding.
SPLIT_SEEDS = {"train": 0, "val": 1_000_000, "test": 2_000_000}


def build_loaders(args, sprites):
    sizes = ((32, 16, 16) if args.fast_dev_run
             else (args.train_size, args.val_size, args.test_size))

    datasets = {
        "train": SyntheticDetectionDataset(SPLIT_SEEDS["train"], sizes[0],
                                           args.image_size, augment=True, sprites=sprites),
        "val": SyntheticDetectionDataset(SPLIT_SEEDS["val"], sizes[1],
                                         args.image_size, sprites=sprites),
        "test": SyntheticDetectionDataset(SPLIT_SEEDS["test"], sizes[2],
                                          args.image_size, sprites=sprites),
    }
    return {
        name: DataLoader(dataset, batch_size=args.batch_size,
                         shuffle=(name == "train"), num_workers=args.workers)
        for name, dataset in datasets.items()
    }


def run_epoch(model, loader, criterion, device, optimizer=None, progress_every=50):
    """One pass over `loader`; trains when an optimizer is supplied.

    Losses are weighted by batch size so a short final batch cannot skew the
    epoch mean against the validation figure it is compared with.
    """
    training = optimizer is not None
    model.train(training)

    totals = {"total": 0.0, "class": 0.0, "l1": 0.0, "iou": 0.0}
    seen = 0

    with torch.set_grad_enabled(training):
        for batch_idx, (images, classes, boxes) in enumerate(loader, start=1):
            images = images.to(device)
            classes = classes.to(device)
            boxes = boxes.to(device)

            logits, pred_boxes = model(images)
            loss, parts = criterion(logits, pred_boxes, classes, boxes)

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            batch = images.size(0)
            seen += batch
            totals["total"] += loss.item() * batch
            for key, value in parts.items():
                totals[key] += value.item() * batch

            if training and (batch_idx == 1 or batch_idx % progress_every == 0
                             or batch_idx == len(loader)):
                print(f"  batch {batch_idx:04d}/{len(loader)} loss {loss.item():.4f}",
                      flush=True)

    return {key: value / seen for key, value in totals.items()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--train-size", type=int, default=6000)
    parser.add_argument("--val-size", type=int, default=1000)
    parser.add_argument("--test-size", type=int, default=1000)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--fast-dev-run", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    torch.manual_seed(42)

    sprites = load_object_sprites(PROJECT_ROOT / "assets" / "objects")
    print(f"Object source: {'assets/objects' if sprites else 'procedurally drawn'}")

    loaders = build_loaders(args, sprites)
    model = SingleObjectDetector().to(device)
    criterion = DetectionLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    model_dir = args.output_dir / "models"
    metric_dir = args.output_dir / "metrics"
    model_dir.mkdir(parents=True, exist_ok=True)
    metric_dir.mkdir(parents=True, exist_ok=True)
    best_path = model_dir / "best_detector.pth"

    history = {"train": [], "val": []}
    best_val = float("inf")
    stale = 0

    epochs = 1 if args.fast_dev_run else args.epochs
    for epoch in range(1, epochs + 1):
        started = time.time()
        print(f"Epoch {epoch:02d}/{epochs}", flush=True)

        train_metrics = run_epoch(model, loaders["train"], criterion, device, optimizer)
        val_metrics = run_epoch(model, loaders["val"], criterion, device)
        history["train"].append(train_metrics)
        history["val"].append(val_metrics)

        if val_metrics["total"] < best_val:
            best_val = val_metrics["total"]
            stale = 0
            torch.save(model.state_dict(), best_path)
            status = f"improved, wrote {best_path.name}"
        else:
            stale += 1
            status = f"no improvement ({stale}/{args.patience})"

        print(f"  train {train_metrics['total']:.4f} | val {val_metrics['total']:.4f} "
              f"| {(time.time() - started) / 60:.1f} min | {status}", flush=True)

        if stale >= args.patience:
            print(f"Early stopping after {args.patience} epochs without improvement.")
            break

    if best_path.exists():
        model.load_state_dict(torch.load(best_path, map_location=device))

    test_metrics = evaluate(model, loaders["test"], device)
    print("\nTest results:")
    for key, value in test_metrics.items():
        print(f"  {key}: {value}")

    (metric_dir / "training_history.json").write_text(json.dumps(history, indent=2))
    (metric_dir / "test_results.json").write_text(json.dumps(test_metrics, indent=2, default=float))
    print(f"\nWrote metrics to {metric_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
