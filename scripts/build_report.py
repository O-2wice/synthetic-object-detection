"""Build the saved-results section without training or mixing smoke and full runs."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
full = ROOT / "outputs/metrics/test_results.json"
yolo = ROOT / "outputs/metrics/yolo_results.json"
lines = []
if full.exists():
    metrics = json.loads(full.read_text())
    if metrics.get("run_mode") != "experiment" or metrics.get("benchmark_result") is not True:
        raise ValueError("Refusing to present a smoke or untagged result as a benchmark.")
    lines += ["The following scores come from the corrected character-data test run.", "",
              "| Metric | Custom detector |", "| --- | --- |"]
    for key in ["samples", "precision", "recall", "f1", "mean_iou", "map50_all_points"]:
        value = metrics[key]
        lines.append(f"| {key} | {value:.4f} |" if isinstance(value, float) else f"| {key} | {value} |")
    for name, caption in [("loss-curves", "Training and validation losses."),
                          ("predictions", "Green: ground truth. Red: predicted box.")]:
        path = ROOT / f"outputs/figures/{name}.png"
        if path.exists():
            lines += ["", f"![{caption}]({path.relative_to(ROOT).as_posix()})"]
    if yolo.exists():
        reference = json.loads(yolo.read_text())
        if reference["dataset_sha256"] != metrics["config"]["dataset_sha256"]:
            raise ValueError("YOLO and custom results use different dataset manifests.")
        lines += ["", "Shared-protocol YOLO test results:", "",
                  "| Metric | YOLOv8n |", "| --- | --- |"]
        for key in ["precision", "recall", "f1", "mean_iou", "map50_all_points"]:
            lines.append(f"| {key} | {reference['shared_protocol'][key]:.4f} |")
    else:
        lines += ["", "YOLO training and held-out evaluation are still pending."]
else:
    lines += ["**The corrected character-data benchmark is pending.** No historical",
              "scores are reused, and the repository remains private.", ""]
smoke = ROOT / "outputs/validation/test_results.json"
if smoke.exists():
    metrics = json.loads(smoke.read_text())
    if metrics.get("run_mode") != "smoke":
        raise ValueError("Unexpected run type in validation artifacts.")
    lines += ["", "### Execution check", "",
        "The notebook completed a two-epoch CPU execution check on 24 training,",
        "6 validation and 6 test geometric fixtures. This exercises the custom",
        "pipeline through generation, training, checkpoint selection, evaluation",
        "and figure export. YOLO is skipped in this mode.", "",
        "Fixture scores are not evidence of character detection accuracy."]
    path = ROOT / "outputs/validation/figures/dataset-samples.png"
    if path.exists():
        lines += ["", "![Geometric execution fixtures with ground-truth boxes. Class names denote test slots, not character artwork.](outputs/validation/figures/dataset-samples.png)"]
pilot_path = ROOT / "outputs/pilot/test_results.json"
pilot_yolo = ROOT / "outputs/pilot/yolo_results.json"
if pilot_path.exists() and pilot_yolo.exists():
    pilot = json.loads(pilot_path.read_text())
    reference = json.loads(pilot_yolo.read_text())
    if pilot.get("run_mode") != "pilot" or pilot.get("benchmark_result") is not False:
        raise ValueError("Incorrect pilot run tags.")
    if reference.get("run_mode") != "pilot" or reference["dataset_sha256"] != pilot["config"]["dataset_sha256"]:
        raise ValueError("Pilot results are from different runs.")
    lines += ["", "### Real-image pilot", "",
        "A short run used 96 training, 24 validation and 24 test character composites,",
        "with two custom-model epochs and one YOLO epoch. Both models were evaluated",
        "on the held-out test images. This validates execution of the real-data path;",
        "it is not the planned 20/100-epoch benchmark.", "",
        "| Shared-protocol metric | Custom (2 epochs) | YOLOv8n (1 epoch) |",
        "| --- | --- | --- |"]
    for key in ["precision", "recall", "mean_iou", "map50_all_points"]:
        lines.append(f"| {key} | {pilot[key]:.4f} | {reference['shared_protocol'][key]:.4f} |")
    lines += ["", "![Real character composites used in the pilot.](outputs/pilot/figures/dataset-samples.png)",
              "", "![Pilot predictions: green is ground truth; red is the custom model prediction.](outputs/pilot/figures/predictions.png)"]
(ROOT / "_results.qmd").write_text("\n".join(lines) + "\n", encoding="utf-8")
print("Updated _results.qmd from saved artifacts.")
