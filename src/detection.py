"""Single-object detection: source-disjoint data, training and shared evaluation."""
from __future__ import annotations

import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

NAMES = ["Waldo", "Wenda", "Wizard Whitebeard"]
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def generate_dataset(objects_dir, backgrounds_dir, destination, counts=(5000, 1000, 200),
                     size=640, seed=42):
    """Split distinct decoded backgrounds BEFORE compositing; reject bad assets.

    An existing destination is never silently mixed with a new experiment.
    Exact decoded duplicates are removed; near duplicates still need human review.
    """
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(f"Dataset already exists: {destination}. Choose a new directory.")
    if len(counts) != 3 or any(n < 3 for n in counts) or size < 32:
        raise ValueError("Use three split counts >= 3 and image size >= 32.")
    objects = []
    for name in NAMES:
        with Image.open(Path(objects_dir) / f"{name}.png") as raw:
            obj = raw.convert("RGBA")
        bounds = obj.getchannel("A").getbbox()
        if bounds is None:
            raise ValueError(f"Empty object: {name}")
        obj = obj.crop(bounds)
        if max(obj.size) > size - 10:
            raise ValueError(f"{name} is too large for {size}px. Resize the source explicitly.")
        objects.append(obj)
    backgrounds = {}
    review_path = Path(backgrounds_dir) / "review.json"
    review = json.loads(review_path.read_text()) if review_path.exists() else {}
    excluded = set(review.get("excluded", []))
    for path in sorted(Path(backgrounds_dir).glob("*")):
        if path.name in excluded:
            continue
        if path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
            continue
        with Image.open(path) as raw:
            rgb = ImageOps.exif_transpose(raw).convert("RGB")
            digest = hashlib.sha256(str(rgb.size).encode() + rgb.tobytes()).hexdigest()
        backgrounds.setdefault(digest, path)
    items = sorted(backgrounds.items())
    if len(items) < 10:
        raise ValueError("At least 10 distinct, readable backgrounds are required.")
    rng = random.Random(seed)
    rng.shuffle(items)
    train_end = max(1, int(len(items) * .8))
    val_end = min(len(items) - 1, max(train_end + 1, int(len(items) * .9)))
    partitions = [items[:train_end], items[train_end:val_end], items[val_end:]]
    manifest = {"seed": seed, "size": size, "names": NAMES, "background_review": review, "counts": dict(zip(
        ["train", "val", "test"], counts)), "backgrounds": {}, "samples": [],
        "objects": {name: hashlib.sha256(obj.tobytes()).hexdigest()
                    for name, obj in zip(NAMES, objects)}}
    for split, pool, count in zip(["train", "val", "test"], partitions, counts):
        image_dir = destination / split / "images"
        label_dir = destination / split / "labels"
        image_dir.mkdir(parents=True)
        label_dir.mkdir(parents=True)
        manifest["backgrounds"][split] = [{"sha256": h, "file": p.name} for h, p in pool]
        classes = [i % len(NAMES) for i in range(count)]
        rng.shuffle(classes)
        for i, cls in enumerate(classes):
            digest, path = rng.choice(pool)
            with Image.open(path) as raw:
                scene = ImageOps.exif_transpose(raw).convert("RGB").resize((size, size))
            obj = objects[cls]
            w, h = obj.size
            x, y = rng.randint(5, size - w - 5), rng.randint(5, size - h - 5)
            scene.paste(obj, (x, y), obj.getchannel("A"))
            image_path = image_dir / f"{i:05d}.jpg"
            scene.save(image_path, quality=95)
            box = [(x + w / 2) / size, (y + h / 2) / size, w / size, h / size]
            label_path = label_dir / f"{i:05d}.txt"
            label_path.write_text(f"{cls} " + " ".join(f"{v:.10f}" for v in box) + "\n")
            manifest["samples"].append({"split": split, "image": image_path.name,
                "background_sha256": digest, "class": cls,
                "image_sha256": hashlib.sha256(image_path.read_bytes()).hexdigest(),
                "label_sha256": hashlib.sha256(label_path.read_bytes()).hexdigest()})
    write_json(destination / "manifest.json", manifest)
    # JSON is valid YAML; this avoids platform-specific absolute /content paths.
    write_json(destination / "data.yaml", {"path": destination.resolve().as_posix(),
        "train": "train/images", "val": "val/images", "test": "test/images", "names": NAMES})
    return manifest


class ObjectDetectionDataset(Dataset):
    def __init__(self, root, split, size=224, augment=False):
        if augment and split != "train":
            raise ValueError("Augmentation is only allowed on training data.")
        self.root, self.split = Path(root), split
        self.files = sorted((self.root / split / "images").glob("*.jpg"))
        if not self.files:
            raise ValueError(f"No images in {root}/{split}")
        self.augment = augment
        ops = [transforms.Resize((size, size))]
        if augment:
            ops.append(transforms.ColorJitter(brightness=.25, contrast=.25, saturation=.25, hue=.03))
        self.transform = transforms.Compose(ops + [transforms.ToTensor(), transforms.Normalize(MEAN, STD)])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        path = self.files[index]
        label = self.root / self.split / "labels" / (path.stem + ".txt")
        values = np.asarray([float(v) for v in label.read_text().split()], dtype=np.float32)
        if values.shape != (5,) or not np.isfinite(values).all():
            raise ValueError(f"Expected exactly one finite class + box row: {label}")
        cls, cx, cy, w, h = values
        if cls != int(cls) or not 0 <= cls < len(NAMES) or min(w, h) <= 0:
            raise ValueError(f"Invalid class or box: {label}")
        if min(cx-w/2, cy-h/2) < -1e-6 or max(cx+w/2, cy+h/2) > 1+1e-6:
            raise ValueError(f"Box outside image: {label}")
        with Image.open(path) as raw:
            image = raw.convert("RGB")
        if self.augment and random.random() < .5:
            image = ImageOps.mirror(image)
            values[1] = 1 - values[1]
        return self.transform(image), torch.from_numpy(values)


class CustomObjectDetectionModel(nn.Module):
    """ResNet18 features, pooled classification and a spatial bounding-box head."""
    def __init__(self, pretrained=True):
        super().__init__()
        backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])
        self.classifier = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(.3), nn.Linear(256, len(NAMES)))
        self.regressor = nn.Sequential(nn.AdaptiveAvgPool2d((4, 4)), nn.Flatten(), nn.LayerNorm(512 * 16),
            nn.Linear(512 * 16, 256), nn.ReLU(), nn.Linear(256, 4))
        # Keep pretrained feature magnitudes from saturating sigmoid coordinates
        # during the first updates. Start near a centered 20%-size box.
        nn.init.normal_(self.regressor[-1].weight, std=.001)
        nn.init.zeros_(self.regressor[-1].bias)
        with torch.no_grad():
            self.regressor[-1].bias[2:] = torch.logit(torch.tensor(.2))

    def forward(self, images):
        features = self.backbone(images)
        raw = self.regressor(features).sigmoid()
        # Positive widths/heights; constrain centers so the WHOLE box fits [0, 1].
        wh = raw[:, 2:].clamp(min=1e-6, max=1-1e-6)
        xy = wh / 2 + raw[:, :2] * (1 - wh)
        return self.classifier(features), torch.cat((xy, wh), dim=1)


def composite_loss(logits, boxes, target):
    classification = nn.functional.cross_entropy(logits, target[:, 0].long())
    regression = nn.functional.smooth_l1_loss(boxes, target[:, 1:])
    return classification + 5 * regression, classification, regression


def box_iou(a, b):
    """Aligned normalized cx,cy,w,h boxes; invalid predictions have zero IoU."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.shape != (4,) or b.shape != (4,) or not np.isfinite([a, b]).all():
        return 0.0
    if min(a[2], a[3], b[2], b[3]) <= 0:
        return 0.0
    lo = np.maximum(a[:2] - a[2:] / 2, b[:2] - b[2:] / 2)
    hi = np.minimum(a[:2] + a[2:] / 2, b[:2] + b[2:] / 2)
    intersection = np.maximum(hi - lo, 0).prod()
    union = a[2:].prod() + b[2:].prod() - intersection
    return float(intersection / union)


def detection_metrics(predictions, targets, threshold=.5):
    """One top-confidence prediction per image, or None for YOLO abstentions.

    AP uses all-points interpolated precision, not COCO's 101-point sampler.
    Every target counts, including class 0 and images with no prediction.
    """
    if len(predictions) != len(targets) or not targets:
        raise ValueError("Predictions and nonempty targets must have equal length.")
    records, ious = [], []
    correct = 0
    for pred, target in zip(predictions, targets):
        if pred is None:
            ious.append(0.0)
            continue
        iou = box_iou(pred[2], target[1])
        same_class = int(pred[0]) == int(target[0])
        ious.append(iou)
        correct += same_class
        records.append((int(pred[0]), float(pred[1]), same_class and iou >= threshold))
    tp = sum(r[2] for r in records)
    precision = tp / len(records) if records else 0.0
    recall = tp / len(targets)
    aps = {}
    for cls in range(len(NAMES)):
        support = sum(int(t[0]) == cls for t in targets)
        if not support:
            continue
        ranked = sorted((r for r in records if r[0] == cls), key=lambda r: -r[1])
        hits = np.cumsum([r[2] for r in ranked])
        if len(hits):
            precisions = hits / np.arange(1, len(hits) + 1)
            envelope = np.maximum.accumulate(precisions[::-1])[::-1]
            increments = np.diff(np.r_[0., hits / support])
            aps[NAMES[cls]] = float((envelope * increments).sum())
        else:
            aps[NAMES[cls]] = 0.0
    return {"samples": len(targets), "predictions": len(records), "true_positives": int(tp),
        "precision": precision, "recall": recall,
        "f1": 2*precision*recall/(precision+recall) if precision+recall else 0.,
        "class_accuracy": correct/len(targets), "mean_iou": float(np.mean(ious)),
        "map50_all_points": float(np.mean(list(aps.values()))), "ap50_by_class": aps,
        "iou_threshold": threshold, "protocol": "one-top-confidence-box-per-image"}


@torch.inference_mode()
def evaluate(model, loader, device):
    model.eval()
    predictions, targets, elapsed = [], [], 0.0
    for i, (images, labels) in enumerate(loader):
        images = images.to(device)
        if i == 0:
            model(images)  # Untimed warm-up.
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        logits, boxes = model(images)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed += time.perf_counter() - start
        scores, classes = logits.softmax(1).max(1)
        predictions.extend((int(c), float(s), b.tolist()) for c,s,b in
                           zip(classes.cpu(), scores.cpu(), boxes.cpu()))
        targets.extend((int(t[0]), t[1:].tolist()) for t in labels)
    metrics = detection_metrics(predictions, targets)
    metrics["forward_ms_per_image"] = elapsed * 1000 / len(targets)
    metrics["timing_scope"] = "warm forward only; excludes loading, transfers and postprocessing"
    metrics["device"] = str(device)
    return metrics, predictions


def run_epoch(model, loader, device, optimizer=None):
    model.train(optimizer is not None)
    totals, count = np.zeros(3), 0
    with torch.set_grad_enabled(optimizer is not None):
        for images, targets in loader:
            images, targets = images.to(device), targets.to(device)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            logits, boxes = model(images)
            losses = composite_loss(logits, boxes, targets)
            if not all(torch.isfinite(v) for v in losses):
                raise FloatingPointError("Nonfinite loss; training stopped.")
            if optimizer is not None:
                losses[0].backward()
                optimizer.step()
            totals += np.asarray([v.item() for v in losses]) * len(images)
            count += len(images)
    return dict(zip(["total", "classification", "box"], (totals/count).tolist()))


def train(dataset, output, epochs=20, batch_size=16, size=224, seed=42,
          pretrained=True, resume=False, patience=5, device=None):
    if epochs < 1 or batch_size < 1 or patience < 1 or size < 32:
        raise ValueError("Require epochs, batch size and patience >= 1 and image size >= 32.")
    seed_everything(seed)
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    output, dataset = Path(output), Path(dataset)
    output.mkdir(parents=True, exist_ok=True)
    fingerprint = hashlib.sha256((dataset / "manifest.json").read_bytes()).hexdigest()
    manifest = json.loads((dataset / "manifest.json").read_text())
    for sample in manifest["samples"]:
        image = dataset / sample["split"] / "images" / sample["image"]
        label = dataset / sample["split"] / "labels" / (image.stem + ".txt")
        for path, key in [(image, "image_sha256"), (label, "label_sha256")]:
            if hashlib.sha256(path.read_bytes()).hexdigest() != sample[key]:
                raise ValueError(f"Dataset file changed since generation: {path}")
    config = {"dataset_sha256": fingerprint, "size": size, "seed": seed,
              "pretrained": pretrained, "batch_size": batch_size, "patience": patience,
              "architecture": "resnet18-spatial4-layernorm-validbox-v2"}
    loaders = {split: DataLoader(ObjectDetectionDataset(dataset, split, size, split == "train"),
        batch_size=batch_size, shuffle=split == "train", num_workers=0)
        for split in ["train", "val", "test"]}
    last_path, best_path = output / "last.pt", output / "best.pt"
    if last_path.exists() and not resume:
        raise FileExistsError("Run exists: use resume=True or choose a new output directory.")
    if resume and not last_path.exists():
        raise FileNotFoundError("Cannot resume: last.pt is missing.")
    model = CustomObjectDetectionModel(pretrained=pretrained and not resume).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    history, start, best, stale = [], 0, float("inf"), 0
    if resume:
        state = torch.load(last_path, map_location="cpu", weights_only=False)
        if state["config"] != config:
            raise ValueError("Resume configuration or dataset differs from checkpoint.")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        history, start, best, stale = state["history"], state["epoch"]+1, state["best"], state["stale"]
        random.setstate(state["python_rng"])
        np.random.set_state(state["numpy_rng"])
        torch.set_rng_state(state["torch_rng"])
        if device.type == "cuda" and state["cuda_rng"] is not None:
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        if not best_path.exists():
            raise FileNotFoundError("Cannot resume without the selected best.pt checkpoint.")
    for epoch in range(start, epochs):
        if stale >= patience:
            break
        row = {"epoch": epoch+1, "train": run_epoch(model, loaders["train"], device, optimizer),
               "val": run_epoch(model, loaders["val"], device)}
        history.append(row)
        if row["val"]["total"] < best:
            best, stale = row["val"]["total"], 0
            torch.save(model.state_dict(), best_path)
        else:
            stale += 1
        state = {"config": config, "epoch": epoch, "model": model.state_dict(),
            "optimizer": optimizer.state_dict(), "history": history, "best": best, "stale": stale,
            "python_rng": random.getstate(), "numpy_rng": np.random.get_state(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else None}
        temporary = output / "last.tmp"
        torch.save(state, temporary)
        temporary.replace(last_path)
        write_json(output / "history.json", history)
        print(f"Epoch {epoch+1}: train={row['train']['total']:.5f}, val={row['val']['total']:.5f}", flush=True)
    model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
    metrics, predictions = evaluate(model, loaders["test"], device)
    metrics.update({"config": config, "epochs_completed": len(history),
        "selected_epoch": min(history, key=lambda r: r['val']['total'])['epoch'],
        "parameters": sum(p.numel() for p in model.parameters()),
        "torch_version": str(torch.__version__)})
    write_json(output / "test_results.json", metrics)
    write_json(output / "predictions.json", predictions)
    return model, history, metrics


def train_yolo(dataset, output, epochs=100, size=640, seed=42):
    """Native YOLO metrics plus the SAME single-object protocol as the custom model."""
    from ultralytics import YOLO
    dataset, output = Path(dataset).resolve(), Path(output).resolve()
    model = YOLO("yolov8n.pt")
    model.train(data=str(dataset / "data.yaml"), epochs=epochs, imgsz=size, batch=8,
                project=str(output), name="train", exist_ok=False, seed=seed, workers=0)
    run_dir = Path(model.trainer.save_dir)
    best = YOLO(str(run_dir / "weights" / "best.pt"))
    native = best.val(data=str(dataset / "data.yaml"), split="test", imgsz=size,
                      project=str(output), name="test", workers=0)
    ds = ObjectDetectionDataset(dataset, "test")
    targets = [(int(ds[i][1][0]), ds[i][1][1:].tolist()) for i in range(len(ds))]
    predictions = []
    for result in best.predict(source=[str(p) for p in ds.files], imgsz=size,
                               conf=.001, max_det=1, stream=True, verbose=False):
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            predictions.append(None)
        else:
            i = int(boxes.conf.argmax())
            predictions.append((int(boxes.cls[i]), float(boxes.conf[i]), boxes.xywhn[i].cpu().tolist()))
    metrics = {"shared_protocol": detection_metrics(predictions, targets),
        "native_test": {k: float(v) for k,v in native.results_dict.items()},
        "note": "Native AP and operating thresholds differ from the shared protocol.",
        "confidence_floor": .001, "imgsz": size, "seed": seed,
        "dataset_sha256": hashlib.sha256((dataset / "manifest.json").read_bytes()).hexdigest()}
    write_json(output / "test_results.json", metrics)
    return metrics
