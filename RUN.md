# Completed detection run: 11 September 2026

Imported from the owner's Downloads folder:

- `object_detection.ipynb` (14,510,432 bytes)
- `synthetic-object-detection-20260911T140352Z-1-001.zip` (1,124,423,552 bytes)

Both originals are retained under `data/completed-runs/2026-09-11/`. The Drive export is extracted below that directory. The previous working notebook is saved there as `working-notebook-before-import.ipynb`; the executed notebook is now `notebooks/object-detection.ipynb`. Import changed outputs and execution metadata, not executable logic: the downloaded source matches the prior working code by Python AST.

## Evidence checked

- All 31 code cells have execution counts, and no saved error outputs occur.
- Notebook schema, syntax and credential-pattern checks pass. No W&B integration was found.
- The Drive dataset archive matches the published SHA-256, `adf1e2239a402544e09c7449069a36de7615d0ffa7c772f7ed0fa8d2cf520c66`.
- Its manifest matches the saved project manifest and identifies dataset `7107ea37118e53dfaf6c6493e9ff4b27aad3464a0add20a4bc69b2314b07f961`.
- Split label counts are 5,000 / 1,000 / 200. Class counts are recorded in the report.
- The custom history contains epochs 1–20. Epoch 16 has minimum validation loss, 0.0011840921. The last checkpoint metadata records epoch index 19, pool grid 3, batch size 16 and the expected dataset ID.
- YOLO history contains exactly epochs 1–100. Saved output documents resumed training, completion and held-out evaluation on 200 test images.
- The export contains custom best/resume weights and YOLO best/last weights. Completed YOLO weights have optimizer state stripped by Ultralytics, so they are inference artifacts, not a promise of further exact optimizer-state resume.
- Both custom evaluation cells report the same accuracy metrics. Timing varies: 6.86 ms in the first pass versus 6.15 ms in the final pass. The final pass is the exported metric used in the report.
- Original prediction, loss and YOLO test figures were visually inspected. Their layouts are retained.

## Results used in the report

| Measurement | Result |
| --- | ---: |
| Custom precision / recall | 0.6550 |
| Custom correct detections | 131 / 200 |
| Custom mean IoU | 0.574632 |
| Custom all-points mAP@0.5 | 0.523597 |
| YOLO native precision / recall | 1.0000 |
| YOLO native mAP50 / mAP50–95 | 0.9950 |

Raw metrics, original figures and a SHA-256 provenance record are under `outputs/completed-run/`. The report does not use previous pilot outputs, claim an experimentally established benefit from pooling alone, or equate custom AP with native YOLO AP.

These checks validate the consistency of the supplied execution evidence. No training or checkpoint inference was rerun locally. Checkpoint metadata was inspected without unpickling the model objects. Cross-hardware numerical reproducibility and unseen-source generalization are not established by this run.
