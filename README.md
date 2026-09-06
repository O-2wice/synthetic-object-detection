# Simplified Object Detection

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/O-2wice/synthetic-object-detection/blob/main/notebooks/object-detection.ipynb)
![Runtime](https://img.shields.io/badge/runtime-CPU%20or%20GPU-blue)
![Framework](https://img.shields.io/badge/framework-PyTorch-orange)

Find one object in a cluttered scene: say what it is, and where.

The detector is written from scratch rather than assembled from a detection
library, so every part is visible: a ResNet-18 backbone, a classification head,
a box-regression head, and a composite objective that trains both at once. A
YOLOv8 baseline is trained on the same data for comparison.

## Quick Start

```powershell
pip install -r requirements.txt
```

Open `notebooks/object-detection.ipynb` and run it top to bottom, or train from
a terminal:

```powershell
python scripts/train_detector.py --epochs 12
python scripts/train_detector.py --fast-dev-run   # smoke test
```

Training resumes from a per-epoch checkpoint, so an interrupted Colab session
costs one epoch rather than the whole run.

## The Data Is Generated, Not Collected

Each image places one object at a random position on procedurally drawn clutter.
The label is the class and the bounding box in normalized centre form.

The dataset is a pure function of a seed. It regenerates identically on any
machine, splits provably cannot overlap because each owns a disjoint seed range,
it costs no storage, and the object's exact position is known rather than
estimated, so labels are correct by construction.

To use real cut-outs instead, drop transparent PNGs named `0_*.png`, `1_*.png`
and `2_*.png` into `assets/objects/`. Placement, labelling and augmentation are
identical either way.

## Method

A ResNet-18 backbone feeds two heads that share its features:

- a classifier over the three object classes
- a box regressor producing `cx, cy, w, h`, normalized to `[0, 1]`

The objective sums three terms: cross-entropy on the class, smooth L1 on the
box, and `1 - IoU`. L1 alone is scale sensitive, treating an error of 0.05 as
equally bad on a large object and a small one; the IoU term optimises what the
metric actually reports, while L1 keeps a usable gradient when the boxes do not
yet overlap.

Reported metrics are class accuracy, mean IoU, and detection rate, the fraction
of images where the class is right *and* IoU clears 0.5.

## Corrections to the Original

This project was rebuilt from a coursework notebook. Three defects mattered:

**Augmentation moved the pixels but not the labels.** The training transform
applied `RandomHorizontalFlip` and `RandomRotation` through a torchvision
`Compose`, which cannot see the bounding box. Roughly half of every epoch taught
the model that the object was on the side it was no longer on; a flipped image
overlaps its unflipped label by about `0.12` IoU. Geometric augmentation now
lives in the dataset, where it transforms the label alongside the image.

**Evaluation excluded one class.** Padding rows were filtered with
`true_classes != 0`, and class 0 was a real class, so those images contributed a
prediction but no ground truth. This is what produced the reported precision
`0.6450` against recall `1.0000`: with one prediction and one ground truth per
image those two numbers are arithmetically forced to be equal, so the gap was
the bug, not a result.

**Localization was pooled away.** The neck collapsed the feature map to `1x1`
before the box head, averaging out the spatial information the head exists to
use. It now pools to `3x3`. The box head also ends in a sigmoid, matching the
`[0, 1]` range of the targets.

## Layout

```text
index.qmd                      # the write-up, rendered to docs/
notebooks/
  object-detection.ipynb       # the experiment
scripts/
  train_detector.py            # same run, from a terminal
src/
  synthdata.py                 # dataset generation, boxes, IoU
  detector.py                  # model, loss, metrics
outputs/
  models/                      # checkpoints (gitignored)
  metrics/                     # results and history
```
