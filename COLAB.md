# Run with the VS Code Colab extension

Open `notebooks/object-detection.ipynb` in VS Code, select the Colab GPU kernel,
and run from the first cell. Setup downloads the character PNGs, saved backgrounds
and the exact dataset from the public GitHub repository. No token or manual
Google Drive upload is required to load the data.

The source images live under `assets/objects/` and `assets/backgrounds/`. The
larger, fixed dataset is the `synthetic-scenes-v1.zip` asset on GitHub release
`dataset-v1`. Setup checks its SHA-256 hash, and the dataset cell verifies every
image and label before training. Both models use the same 5,000 / 1,000 / 200
scenes prepared locally.

The notebook keeps the original models, 640-pixel images and plotting layouts.
The custom detector trains for up to 20 epochs with early stopping; YOLOv8n
trains for up to 100 epochs. The first download is approximately 578 MiB plus
the small source-asset download. Later cells reuse the local runtime files.

## Checkpoints and remote runtime storage

`USE_DRIVE` defaults to `False` for the VS Code Colab extension. Checkpoints are
written after completed epochs under `/content/synthetic-object-detection/outputs/original-notebook/`.
They survive rerunning cells in the same runtime, but a reset can erase them.

To preserve progress, download the checkpoint files before ending the runtime.
The final export cell creates `outputs/original-notebook/run-artifacts.zip`
with metrics, figures, custom best/last checkpoints, YOLO run files and the dataset
manifest. Save the executed notebook too. For earlier interruption, the same
export cell can be run after the YOLO setup cell has defined its paths, or copy
the checkpoint folders directly.

To resume in a new runtime, run setup and restore the archive's `models/`,
`metrics/`, and `yolo/` folders under `outputs/original-notebook/` before running
training. Keep the custom `best_model.pth` and `last_checkpoint.pth` together.
The custom checkpoint restores Adam, the scheduler, mixed precision, random
states and history. YOLO resumes from its own `last.pt`; an incomplete epoch
runs again. GPU and library differences can affect numerical reproducibility.

In browser Colab, optional `USE_DRIVE = True` mounts Drive and writes checkpoints
there. That route is optional and is not required for GitHub image loading.

The Open in Colab badge opens the current notebook on GitHub. The full notebook
run in your Colab environment still precedes the new Quarto write-up.
