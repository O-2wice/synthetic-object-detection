"""Check saved source and prepared-image hashes; optionally restore missing downloads."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--restore-missing", action="store_true")
args = parser.parse_args()
characters = json.loads((ROOT / "assets/sources/characters.json").read_text())
backgrounds = json.loads((ROOT / "assets/sources/backgrounds.json").read_text())
records = [(ROOT / "assets/sources/WaldoCharacterSheets20.pdf", characters["pdf_sha256"], characters["url"])]
records += [(ROOT / "assets/objects" / (row["name"] + ".png"), row["sha256"], None)
            for row in characters["objects"]]
records += [(ROOT / "assets/backgrounds" / row["file"], row["sha256"], row["url"])
            for row in backgrounds["images"]]
for path, expected, url in records:
    if not path.exists() and args.restore_missing and url:
        import requests
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        if hashlib.sha256(response.content).hexdigest() != expected:
            raise ValueError(f"Remote source changed; not restoring {path.name}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(response.content)
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Restore the source or rerun prepare_publisher_assets.py.")
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError(f"Hash mismatch: {path}")
print(f"Verified {len(records)} saved source and image files.")
