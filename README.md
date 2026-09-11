# Synthetic Object Detection

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/O-2wice/synthetic-object-detection/blob/main/notebooks/object-detection.ipynb)

Recognize a character in a cluttered scene and locate its bounding box.

The [notebook](notebooks/object-detection.ipynb) preserves the original experiment:
synthetic scenes, dataset inspection, ResNet18/VGG16 model comparison, custom
classification and box regression, and YOLOv8n. Its prose presents the project;
its figure layouts follow the original notebook. The architecture follows it
too, with one documented exception: the neck's pooling grid is configurable
via `POOL_GRID`, and `POOL_GRID = 1` reproduces the original exactly. The
checks assert that equivalence rather than asserting the source is identical.

**Current stage:** preparing for the owner's full Colab run. The repository is public so Colab can fetch its saved assets directly.
The old Quarto write-up has been removed; the new write-up follows that run.
Earlier seven-cell pilot results belong to the discarded rewrite and do not
validate this notebook.

Follow [COLAB.md](COLAB.md) to use the prepared local dataset and persistent
checkpoints. [DATA.md](DATA.md) records the saved source assets and the approved
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
