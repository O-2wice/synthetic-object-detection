"""Execute the notebook with this interpreter; preserve outputs on success or failure."""
import argparse
import os
from pathlib import Path
import sys

import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--mode", choices=["smoke", "pilot", "experiment"], default="smoke")
parser.add_argument("--output-notebook", type=Path, help="Save execution evidence separately from the source notebook.")
args = parser.parse_args()
os.environ["DETECTION_RUN_MODE"] = args.mode
path = ROOT / "notebooks/object-detection.ipynb"
notebook = nbformat.read(path, as_version=4)
output_path = args.output_notebook or path
output_path.parent.mkdir(parents=True, exist_ok=True)
manager = KernelManager(kernel_name="python3")
manager.kernel_spec.argv = [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"]
def cell_started(cell, cell_index):
    if cell.cell_type == "code":
        print(f"Executing cell {cell_index}...", flush=True)


def cell_finished(cell, cell_index, execute_reply):
    nbformat.write(notebook, output_path)
    print(f"Completed cell {cell_index}.", flush=True)


client = NotebookClient(notebook, km=manager, timeout=86400,
    on_cell_start=cell_started, on_cell_executed=cell_finished,
    resources={"metadata": {"path": str(ROOT)}})
try:
    client.execute()
finally:
    nbformat.write(notebook, output_path)
print(f"Executed {sum(c.cell_type == 'code' for c in notebook.cells)} cells in {args.mode} mode.")
