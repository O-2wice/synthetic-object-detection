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

`USE_DRIVE` defaults to `True`. The environment cell mounts Google Drive and
stops before training if mounting fails. In VS Code, use the command palette
command **Colab: Mount Google Drive to Server...**, complete authorization and
rerun the environment cell. This command is documented in the
[official extension README](https://github.com/googlecolab/colab-vscode).

Custom checkpoints are saved after each completed epoch under
`MyDrive/synthetic-object-detection/models/original-notebook/`. YOLO checkpoints
are under `MyDrive/synthetic-object-detection/yolo/train/weights/`. Metrics and
figures are saved under that same project folder on Drive. The image dataset
stays on the runtime's local disk for training speed.

The final export cell creates `outputs/original-notebook/run-artifacts-pool3.zip`
with metrics, figures, custom best/last checkpoints, YOLO run files and the dataset
manifest and copies it to Drive. Save the executed notebook too. The export
cell is also usable after custom training, before the YOLO section. Checkpoints
already saved to Drive do not depend on reaching the export cell.

To resume in a new runtime, mount the same Drive and run setup again. The notebook
finds the saved checkpoints automatically. Keep the custom `best_model.pth` and
`last_checkpoint.pth` together. When moving a YOLO run between filesystems, a
derived resume checkpoint adjusts the saved data path without changing its
weights, optimizer or completed epoch. The original `last.pt` is preserved.
The custom checkpoint restores Adam, the scheduler, mixed precision, random
states and history. YOLO resumes from its own `last.pt`; an incomplete epoch
runs again. GPU and library differences can affect numerical reproducibility.

If you deliberately turn off `USE_DRIVE`, checkpoints return to temporary
runtime storage. Download them before ending that runtime; saving the notebook
does not save weights. Restore the export's `models/`, `metrics/` and `yolo/`
folders under `outputs/original-notebook/` for local resumption. A runtime that
was already deleted cannot be recovered from notebook outputs alone.

Normal local and Colab runs download the same frozen dataset. Ultralytics is
pinned to 8.4.143, the version used in the saved Colab run. Other runtime versions
are recorded in `metrics/runtime.json`; saved random states do not guarantee
identical numerical results across different hardware or libraries.

The Open in Colab badge opens the current notebook on GitHub. The full notebook
run in your Colab environment still precedes the new Quarto write-up.
