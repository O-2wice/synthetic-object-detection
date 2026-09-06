"""Collect candidate backgrounds while preserving source URL and file hashes."""
import argparse
import hashlib
import json
from pathlib import Path

from icrawler import ImageDownloader
from icrawler.builtin import BingImageCrawler

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--count", type=int, default=40)
args = parser.parse_args()
destination = ROOT / "assets/backgrounds"
destination.mkdir(parents=True, exist_ok=True)
if any(destination.glob("*.jpg")):
    raise FileExistsError("Saved collection already exists; do not overwrite its source mapping with a new search.")
sources = {}


class SourceDownloader(ImageDownloader):
    def get_filename(self, task, default_ext):
        name = super().get_filename(task, default_ext)
        sources[name] = task["file_url"]
        return name


crawler = BingImageCrawler(downloader_cls=SourceDownloader,
    storage={"root_dir": str(destination)}, downloader_threads=1)
crawler.crawl(keyword="doodle pattern background", max_num=args.count)
records = [{"file": name, "url": url,
            "sha256": hashlib.sha256((destination / name).read_bytes()).hexdigest()}
           for name,url in sources.items() if (destination / name).exists()]
(ROOT / "assets/sources").mkdir(parents=True, exist_ok=True)
(ROOT / "assets/sources/backgrounds.json").write_text(json.dumps({
    "query": "doodle pattern background", "retrieved": "2026-09-06", "images": records,
    "status": "Candidate web images for a private experiment; not a licensed redistribution dataset."
}, indent=2), encoding="utf-8")
print(f"Saved {len(records)} source records.")
