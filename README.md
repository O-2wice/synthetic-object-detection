# Synthetic Object Detection

[![Read the report](https://img.shields.io/badge/read-the%20report-1b7f5a)](https://o-2wice.github.io/synthetic-object-detection/)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/O-2wice/synthetic-object-detection/blob/main/notebooks/object-detection.ipynb)
![Runtime](https://img.shields.io/badge/runtime-CPU%20or%20GPU-blue)
![Framework](https://img.shields.io/badge/framework-PyTorch-orange)
![Reference](https://img.shields.io/badge/reference-YOLOv8n-purple)

Find one character in a cluttered scene: say which of three it is, and where.
Every label is exact, because the generator places the character rather than
annotating it afterwards.

**[Read the write-up](https://o-2wice.github.io/synthetic-object-detection/)** for the method, the figures and the results.

|                             | Custom ResNet18 | YOLOv8n |
| --------------------------- | --------------: | ------: |
| Trainable parameters        |      13,539,143 | 3,006,233 |
| Correct at IoU 0.5          |     **131/200** | — |
| Mean IoU                    |          0.5746 | — |
| Native mAP50-95             |               — | **0.995** |

The two columns are scored under different protocols, which the report explains;
they are not a single leaderboard.

The [notebook](notebooks/object-detection.ipynb) preserves the original experiment:
synthetic scenes, dataset inspection, ResNet18/VGG16 model comparison, custom
classification and box regression, and YOLOv8n. Its prose presents the project;
its figure layouts follow the original notebook. The architecture follows it
too, with one documented exception: the neck's pooling grid is configurable
via `POOL_GRID`, and `POOL_GRID = 1` reproduces the original exactly. The
checks assert that equivalence rather than asserting the source is identical.

**Completed run:** the executed notebook contains 20 custom-model epochs,
100 YOLO epochs, both held-out evaluations and the final export.

The [Quarto report](index.qmd) is where the experiment is explained and the
results are interpreted: how the labels are constructed, why an augmentation has
to move the bounding box with the pixels, what the two heads each learn, and
what the held-out numbers do and do not establish. Render it with
`quarto render`; rendering reads the saved figures and metrics from
`outputs/completed-run/` and does not execute training.

The [run verification record](notes/RUN.md) documents the imported
notebook, the Drive export and the checks. Earlier pilot results are not used.

Follow [notes/COLAB.md](notes/COLAB.md) to use the prepared local dataset and persistent
checkpoints. [notes/DATA.md](notes/DATA.md) records the saved source assets and the approved
Wenda replacement. Both models use the same frozen 5,000 / 1,000 / 200 scenes.

For local setup, install `requirements.txt` in a Python 3.11+ environment; the
workflow diagram also requires the Graphviz `dot` executable. Focused checks:

```powershell
.venv\Scripts\python scripts/check_original_notebook.py
.venv\Scripts\python scripts/verify_assets.py
.venv\Scripts\python scripts/check_notebook_privacy.py
```

These checks exercise notebook code and saved assets, not full detector training.
The earlier notebook generator, trainer and report builder are disabled to
prevent replacing the preserved notebook or mixing results from different code.
