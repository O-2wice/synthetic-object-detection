# Synthetic Object Detection

Find a character in a cluttered scene: predict what it is and where it is.

Three transparent character cut-outs are composited onto backgrounds. Placement
coordinates provide the labels. A custom ResNet18 detector learns classification
and bounding-box regression, with YOLOv8n as a reference on the same generated data.

The [notebook](notebooks/object-detection.ipynb) runs the experiment.
The [write-up](index.qmd) explains the method and evaluation.

**Status:** the repository remains private. The revised character-data benchmark
and YOLO training have not been completed. Historical scores are not presented as
results of the corrected pipeline. Small geometric fixtures are used only for
execution checks; their outputs live in `outputs/validation/`.
The approved replacement character files and reviewed background collection are
saved under `assets/` with source URLs and hashes. Wenda replaces the original
Wilma class; the original Drive links returned 404.

Validation completed: eight regression tests, a seven-cell smoke execution and
a seven-cell real-image pilot. The pilot ran both models; the custom model still
scored zero correct detections at IoU ≥ 0.5 after two epochs. It establishes that
the revised pipeline runs, not that training is complete. See
[validation evidence](outputs/validation/checks.json) and [pilot metrics](outputs/pilot/test_results.json).

## Run

Use Python 3.11 or newer in a dedicated environment:

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
```

Select that environment as the kernel for `notebooks/object-detection.ipynb`.
The saved assets are ready to use; [DATA.md](DATA.md) explains how they were
prepared. Run the notebook from top to bottom. See [COLAB.md](COLAB.md) for GPU setup.

The default experiment generates 5,000 training, 1,000 validation and 200 test
images. The custom detector trains for up to 20 epochs; YOLOv8n trains for up to
100. Full training needs a fresh run before any performance comparison is valid.

For a small execution check without external assets or weight downloads:

```powershell
.venv\Scripts\python scripts/execute_notebook.py --mode smoke
```

This executes the same notebook on geometric fixtures and skips YOLO. It is not
a substitute for running the character-data experiment. Repeating training in
the same output directory requires `RESUME = True` in the notebook; use a new
directory for an independent experiment.

`--mode pilot` runs two custom-model epochs and one YOLO epoch on 96 / 24 / 24
actual character composites. Its outputs live in `outputs/pilot/`, separate from
both smoke checks and the full benchmark.

After data generation, training can also run from a terminal:

```powershell
.venv\Scripts\python scripts/train_detection.py --epochs 20
.venv\Scripts\python scripts/train_detection.py --epochs 20 --resume
.venv\Scripts\python scripts/train_detection.py --yolo --epochs 100 --output outputs/yolo_runs
```

## Method

Source backgrounds are split before compositing, with exact decoded duplicates
removed. Validation and test transformations are deterministic. Training flips
move the image and box together. Invalid or missing labels stop the run.

The custom detector fine-tunes a pretrained ResNet18. Its classification head
pools globally; its box head keeps a 4 × 4 spatial grid. Predicted boxes are
constrained to fit the image. The objective is cross-entropy plus five times
Smooth L1. Validation loss selects the best checkpoint.
Layer normalization and conservative box-head initialization prevent the early
sigmoid saturation found in the first real-data pilot.

Both models use a shared single-object evaluation protocol: retain one prediction
per image, match the class and require IoU ≥ 0.5. YOLO may abstain. Reported AP50
uses all-points interpolation; native YOLO metrics are reported separately.
YOLO sees 640-pixel images and trains longer than the 224-pixel custom detector,
so this is not an equal-budget architecture comparison.

## Report and checks

```powershell
.venv\Scripts\python -m unittest discover -s tests -v
.venv\Scripts\python scripts/build_report.py
quarto render
```

The report renders to `docs/` using saved artifacts; rendering never starts
training or downloads data. Publishing is a separate step and is not enabled.

```text
notebooks/object-detection.ipynb   # narrative and executable experiment
src/detection.py                  # data, model, training, shared metrics
src/visualization.py              # sample, loss and prediction figures
scripts/                          # notebook execution, training, report export
tests/                            # correctness regression checks
data/                             # local assets and generated data (ignored)
assets/                           # downloaded sources, prepared cut-outs, backgrounds
outputs/models/                   # checkpoints and per-run artifacts (ignored)
outputs/metrics/                  # full experiment metrics, when available
outputs/validation/               # explicitly labeled fixture-run evidence
outputs/pilot/                    # short real-image execution run
index.qmd                         # Quarto write-up
```

## Scope

Three fixed character cut-outs, one object per image, fixed object scale and no
occlusion or negative scenes. New backgrounds do not establish generalization
to new poses or real photographs. Near-duplicate backgrounds need manual review.

[AUDIT.md](AUDIT.md) records the corrections and remaining validation work.
[DATA.md](DATA.md) records source requirements. The project originated from an
ELTE Deep Network Development exercise; the task template was by Tamás Takács
and Imre Molnár, and the original implementation was by Robert Ouko Oyombe.
