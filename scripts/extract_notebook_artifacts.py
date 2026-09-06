"""Recover explicitly tagged run artifacts from a saved notebook (e.g. from Colab)."""
import argparse
import base64
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--notebook", type=Path, default=ROOT / "notebooks/object-detection.ipynb")
parser.add_argument("--check", action="store_true")
args = parser.parse_args()
notebook = json.loads(args.notebook.read_text(encoding="utf-8"))
payloads = []
figures = {}
for cell in notebook["cells"]:
    source = cell["source"]
    source = source if isinstance(source, str) else "".join(source)
    text = ""
    for output in cell.get("outputs", []):
        if output.get("output_type") == "stream":
            value = output["text"]
            text += value if isinstance(value, str) else "".join(value)
        encoded = output.get("data", {}).get("image/png")
        if encoded:
            for name in ["dataset-samples", "loss-curves", "predictions"]:
                if f"'{name}.png'" in source:
                    figures[name] = encoded
    for match in re.finditer(r"RUN_ARTIFACTS_BEGIN\s*(.*?)\s*RUN_ARTIFACTS_END", text, re.S):
        payloads.append(json.loads(match.group(1)))
if len(payloads) != 1:
    raise SystemExit("Expected exactly one complete tagged run. Finish the notebook before extraction.")
payload = payloads[0]
mode = payload["mode"]
if mode not in {"smoke", "pilot", "experiment"} or payload["custom"].get("run_mode") != mode:
    raise SystemExit("Invalid or inconsistent run-mode tags.")
if payload["custom"].get("benchmark_result") != (mode == "experiment"):
    raise SystemExit("Benchmark status does not match the run mode.")
if any(o.get("output_type") == "error" for c in notebook["cells"] for o in c.get("outputs", [])):
    raise SystemExit("Notebook contains an execution error; artifacts were not extracted.")
base = ROOT / "outputs" / {"smoke": "validation", "pilot": "pilot", "experiment": "metrics"}[mode]
figure_dir = ROOT / "outputs/figures" if mode == "experiment" else base / "figures"
print(f"Run: {mode}; epochs: {len(payload['history'])}; figures: {list(figures)}")
if not args.check:
    base.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    for name, value in [("test_results", payload["custom"]), ("training_history", payload["history"]),
                        ("yolo_results", payload["yolo"])]:
        if value is not None:
            (base / f"{name}.json").write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    for name, encoded in figures.items():
        (figure_dir / f"{name}.png").write_bytes(base64.b64decode(encoded))
    print(f"Extracted to {base}")
