# Running on a GPU

The repository is private while the revised experiment is being checked. A public
Colab badge is not a working distribution route yet.

Transfer a ZIP of the working checkout to your own Colab runtime, or copy the
project folder from Google Drive. Keep `src/`, `scripts/` and `notebooks/` together.
In a setup cell, change to that project folder and run:

```python
%pip install -r requirements.txt
```

Choose a GPU runtime and confirm `torch.cuda.is_available()`. Put the three
transparent character PNGs and reviewed backgrounds in the paths in `DATA.md`.
Open `notebooks/object-detection.ipynb`, keep `RUN_MODE = 'experiment'`, and run
top to bottom. The first model run downloads ImageNet ResNet18 weights; YOLO
downloads `yolov8n.pt`.

Use local runtime storage for image data. Point `RUN_DIR` to a persistent Drive
folder if resume checkpoints should survive runtime loss. Set `RESUME = True`
to continue custom-model training with the same data and settings. Restore both
`last.pt` and `best.pt`; the checkpoint verifies the dataset fingerprint.

The YOLO helper starts a fresh run and does not implement interrupted-run resume.

Save the executed notebook and copy `outputs/metrics/`, `outputs/figures/` and the
dataset manifest back to the checkout. Run `scripts/build_report.py` to refresh
the report. `outputs/validation/` contains smoke checks, not benchmark results.

If only the executed notebook comes back, `python scripts/extract_notebook_artifacts.py`
recovers its tagged JSON and figures. It rejects an incomplete run or execution
errors, and keeps pilot/smoke outputs separate from benchmark results.
