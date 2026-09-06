"""Detection model, dataset wrapper, losses and metrics.

Single-object detection: every image contains exactly one object, so the model
predicts one class and one box rather than running proposals or anchors. That
keeps the pipeline small enough to read end to end while still exercising the
parts that matter — a shared backbone, two heads, a composite objective and
IoU-based evaluation.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms

from synthdata import NUM_CLASSES, Box, horizontal_flip, make_sample

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

_to_tensor = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


class SyntheticDetectionDataset(Dataset):
    """Generate samples on the fly from a deterministic seed range.

    Nothing is written to disk. Each index maps to a fixed seed, so a split is
    reproducible, splits cannot overlap, and regenerating costs no storage.

    Augmentation lives here rather than in a torchvision `Compose` because a
    geometric transform has to move the label with the pixels. The coursework
    version put `RandomHorizontalFlip` in the Compose and left the box alone,
    which trained the model against mirrored targets half the time.
    """

    def __init__(self, seed_start: int, length: int, image_size: int = 224,
                 augment: bool = False, sprites=None):
        self.seed_start = seed_start
        self.length = length
        self.image_size = image_size
        self.augment = augment
        self.sprites = sprites

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int):
        seed = self.seed_start + idx
        image, class_id, box = make_sample(
            seed, self.image_size, sprites=self.sprites
        )

        if self.augment:
            # Seeded off the sample index so an epoch is reproducible.
            if (seed * 2654435761) % 2 == 0:
                image, box = horizontal_flip(image, box)

        target_box = torch.tensor(box.as_tuple(), dtype=torch.float32)
        return _to_tensor(image), torch.tensor(class_id, dtype=torch.long), target_box


class SingleObjectDetector(nn.Module):
    """ResNet-18 backbone with a classification head and a box head.

    Two departures from the coursework model, both aimed at localization:

    - The neck pools to a 3x3 grid instead of 1x1. Collapsing the feature map
      to a single vector averages away *where* things are, which is precisely
      what the box head needs; keeping a coarse grid preserves it at negligible
      cost.
    - The box head ends in a sigmoid. Targets are normalized to [0, 1], so an
      unbounded head starts out predicting impossible boxes and spends early
      training walking back into range.
    """

    def __init__(self, num_classes: int = NUM_CLASSES, grid: int = 3,
                 pretrained: bool = True, freeze_backbone: bool = False):
        super().__init__()
        from torchvision import models

        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.resnet18(weights=weights)
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])

        if freeze_backbone:
            for parameter in self.backbone.parameters():
                parameter.requires_grad = False

        self.pool = nn.AdaptiveAvgPool2d((grid, grid))
        feature_dim = 512 * grid * grid

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )
        self.box_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 4),
            nn.Sigmoid(),
        )

    def forward(self, x):
        features = self.pool(self.backbone(x))
        return self.classifier(features), self.box_head(features)


def boxes_to_corners(boxes: torch.Tensor) -> torch.Tensor:
    """[N, 4] centre form -> [N, 4] corner form, batched."""
    cx, cy, w, h = boxes.unbind(-1)
    return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1)


def box_iou(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Elementwise IoU for two [N, 4] batches of centre-form boxes."""
    p = boxes_to_corners(pred)
    t = boxes_to_corners(target)

    inter_w = (torch.min(p[:, 2], t[:, 2]) - torch.max(p[:, 0], t[:, 0])).clamp(min=0)
    inter_h = (torch.min(p[:, 3], t[:, 3]) - torch.max(p[:, 1], t[:, 1])).clamp(min=0)
    intersection = inter_w * inter_h

    area_p = (pred[:, 2] * pred[:, 3]).clamp(min=0)
    area_t = (target[:, 2] * target[:, 3]).clamp(min=0)
    union = area_p + area_t - intersection
    return intersection / union.clamp(min=1e-9)


class DetectionLoss(nn.Module):
    """Cross-entropy for the class, plus a box term that mixes L1 and IoU.

    Smooth L1 alone treats every coordinate independently and is scale
    sensitive: the same absolute error matters far more for a small object than
    a large one. Adding (1 - IoU) optimises the quantity the metric actually
    reports, while the L1 term keeps gradients useful when boxes do not yet
    overlap and IoU is flat at zero.
    """

    def __init__(self, w_class: float = 1.0, w_l1: float = 5.0, w_iou: float = 2.0):
        super().__init__()
        self.w_class = w_class
        self.w_l1 = w_l1
        self.w_iou = w_iou

    def forward(self, class_logits, pred_boxes, true_classes, true_boxes):
        class_loss = F.cross_entropy(class_logits, true_classes)
        l1_loss = F.smooth_l1_loss(pred_boxes, true_boxes)
        iou_loss = (1.0 - box_iou(pred_boxes, true_boxes)).mean()

        total = self.w_class * class_loss + self.w_l1 * l1_loss + self.w_iou * iou_loss
        return total, {
            "class": class_loss.detach(),
            "l1": l1_loss.detach(),
            "iou": iou_loss.detach(),
        }


@torch.no_grad()
def evaluate(model, loader, device, iou_threshold: float = 0.5) -> dict:
    """Accuracy, mean IoU and detection rate over a loader.

    Every image holds exactly one object and the model emits exactly one
    prediction, so the number of predictions and the number of ground truths are
    equal by construction. Precision and recall are therefore the same number,
    and reporting them as if they could differ hides mistakes: the coursework
    version excluded class 0 from the ground-truth count but not from the
    prediction count, which is what produced its precision 0.645 / recall 1.000.

    A detection counts as correct only when the class is right *and* IoU clears
    the threshold, so the single figure covers both heads.
    """
    model.eval()
    total = 0
    class_correct = 0
    detected = 0
    iou_sum = 0.0
    per_class = {c: [0, 0] for c in range(NUM_CLASSES)}  # [correct, count]

    for images, classes, boxes in loader:
        images = images.to(device, non_blocking=True)
        classes = classes.to(device, non_blocking=True)
        boxes = boxes.to(device, non_blocking=True)

        logits, pred_boxes = model(images)
        predicted = logits.argmax(dim=1)
        ious = box_iou(pred_boxes, boxes)

        correct = (predicted == classes) & (ious >= iou_threshold)

        total += images.size(0)
        class_correct += (predicted == classes).sum().item()
        detected += correct.sum().item()
        iou_sum += ious.sum().item()

        for class_id in range(NUM_CLASSES):
            mask = classes == class_id
            per_class[class_id][0] += correct[mask].sum().item()
            per_class[class_id][1] += mask.sum().item()

    return {
        "samples": total,
        "class_accuracy": class_correct / max(total, 1),
        "mean_iou": iou_sum / max(total, 1),
        "detection_rate": detected / max(total, 1),
        "per_class_detection_rate": {
            c: (hits / count if count else float("nan"))
            for c, (hits, count) in per_class.items()
        },
    }
