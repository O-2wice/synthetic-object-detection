"""Validate the report's saved execution evidence without loading models or training."""
import csv
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/completed-run'
record = json.loads((OUT / 'provenance.json').read_text())
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
assert sha(ROOT / 'notebooks/object-detection.ipynb') == record['notebook_sha256']
for name, digest in record['artifacts'].items():
    assert sha(OUT / name) == digest, name
notebook = json.loads((ROOT / 'notebooks/object-detection.ipynb').read_bytes())
cells = [c for c in notebook['cells'] if c['cell_type'] == 'code']
assert len(cells) == 31
assert [c['execution_count'] for c in cells] == list(range(1, 32))
assert not any(o.get('output_type') == 'error' for c in cells for o in c.get('outputs', []))
custom = json.loads((OUT / 'metrics/custom_test.json').read_text())
yolo = json.loads((OUT / 'metrics/yolo_test.json').read_text())
assert custom['dataset_id'] == yolo['dataset_id'] == record['dataset_id']
metrics = custom['metrics']
assert metrics['num_test_samples'] == 200
assert math.isclose(metrics['precision'] * 200, 131, abs_tol=1e-5)
assert math.isclose(metrics['recall'], metrics['precision'])
assert math.isclose(metrics['mean_iou'], 0.5746320038661361)
assert math.isclose(metrics['mAP@0.5'], 0.5235965585525623)
assert yolo['metrics']['metrics/precision(B)'] == 1
assert yolo['metrics']['metrics/recall(B)'] == 1
assert math.isclose(yolo['metrics']['metrics/mAP50-95(B)'], .995)
history = json.loads((OUT / 'metrics/training_results.json').read_text())
assert [r['epoch'] for r in history] == list(range(1, 21))
assert min(history, key=lambda r: r['val_loss'])['epoch'] == 16
with (OUT / 'yolo/results.csv').open() as f:
    rows = list(csv.DictReader(f))
assert [int(row['epoch']) for row in rows] == list(range(1, 101))
print('PASS: report artifacts match imported hashes; 31 completed cells; histories and test metrics consistent.')
print('No model weights loaded and no training executed.')
