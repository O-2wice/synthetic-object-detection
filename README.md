# Simplified Object Detection

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/O-2wice/synthetic-object-detection/blob/main/notebooks/object-detection.ipynb)
![Runtime](https://img.shields.io/badge/runtime-CPU%20or%20GPU-blue)
![Framework](https://img.shields.io/badge/framework-PyTorch-orange)

Find a character hidden in a cluttered scene: predict both what it is and where
it is.

No annotated dataset of these characters exists, so one is built. Object
cut-outs are composited at random positions onto crawled doodle backgrounds, and
the placement coordinates become the labels. A detector is then written from
scratch — a pretrained CNN backbone with a classification head and a
bounding-box head — and a YOLOv8 model is trained on the same data as a
reference point.

## Quick Start

Open `notebooks/object-detection.ipynb` and run it top to bottom. A GPU runtime
is worth it: the custom model trains for 20 epochs and YOLOv8 for 100.

The notebook installs what it needs (`icrawler`, `rembg`, `torchsummary`,
`ultralytics`) in its first cells, so nothing has to be prepared beforehand.

## Pipeline

1. Crawl doodle backgrounds from the web with `icrawler`.
2. Load three object cut-outs with their backgrounds removed.
3. Composite one object per background at a random position, recording the class
   and box as YOLO-format labels.
4. Build datasets and dataloaders over the generated splits.
5. Define the detector: a backbone with class and box heads.
6. Train with a composite loss, validation monitoring and early stopping.
7. Evaluate with precision, recall, F1 and IoU, and visualise predictions.
8. Train YOLOv8 on the same data and compare.

## Corrections

The experiment, architecture and epoch counts are unchanged from the original
run. Three defects that affected the results were fixed.

**Augmentation moved the pixels but not the labels.**

```python
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    ...
])
```

A torchvision `Compose` only sees the image. The bounding box lives in a
separate label file and did not move with it, so on roughly half of every epoch
the model was shown a mirrored image and told the object was still on the side
it had just left. The flip now happens inside the dataset, where the label is in
scope and can be mirrored with it. Rotation was dropped rather than
reimplemented: an axis-aligned box cannot represent a rotated object without
growing to cover it, so rotating the label is not well defined here.

**Evaluation excluded one of the three classes.**

```python
valid_mask = true_classes[i] != 0   # meant to drop padding rows
```

Padding rows are `[0, 0, 0, 0, 0]`, but class 0 is Waldo — a real class. Every
Waldo image therefore contributed a prediction while its ground truth was
discarded. That is why the run reported precision `0.6450` against recall
`1.0000`: with exactly one prediction and one ground truth per image those two
numbers are arithmetically forced to be equal, so the gap was the bug rather
than a property of the model. Padding is now identified by its zero-area box.

**The reported model summary did not describe the forward pass.** Images were
generated at 640×640 and fed in at that size, while `torchsummary` was called
with `(3, 224, 224)`. A resize to 224 was added to every split so the two agree.
Because the labels are normalised to the image, resizing does not disturb them.

Also updated: `pretrained=True` to the current `weights=` API.

## Known Limitations

Two design choices were left as they were, because they are the original
author's decisions rather than defects, but both are worth naming:

- The neck pools features to `1×1` before the box head, which averages away the
  spatial information that head exists to use.
- The box head is unbounded while the targets are normalised to `[0, 1]`, so it
  begins by predicting boxes that cannot exist.

Both limit localisation quality and are the first things worth changing next.

## Layout

```text
index.qmd                      # the write-up, rendered to docs/
notebooks/
  object-detection.ipynb       # the experiment
scripts/
  extract_notebook_artifacts.py  # recovers figures and metrics from a run
outputs/
  models/                      # checkpoints (gitignored)
  metrics/                     # results and history
```
