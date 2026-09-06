"""Reproduce approved Waldo/Wenda/Whitebeard cut-outs from the publisher PDF."""
import hashlib
import json
from pathlib import Path

from PIL import Image
import pymupdf
import requests

ROOT = Path(__file__).resolve().parents[1]
SOURCE = "https://waldo.candlewick.com/pdf/WaldoCharacterSheets20.pdf"
source_dir, object_dir = ROOT / "assets/sources", ROOT / "assets/objects"
source_dir.mkdir(parents=True, exist_ok=True)
object_dir.mkdir(parents=True, exist_ok=True)
pdf_path = source_dir / "WaldoCharacterSheets20.pdf"
if not pdf_path.exists():
    existing = ROOT / "data/downloads/WaldoCharacterSheets20.pdf"
    if existing.exists():
        pdf_path.write_bytes(existing.read_bytes())
    else:
        response = requests.get(SOURCE, timeout=60)
        response.raise_for_status()
        pdf_path.write_bytes(response.content)
doc = pymupdf.open(pdf_path)
specs = [("Waldo", 0, [150, 35, 462, 756]),
         ("Wenda", 2, [182, 35, 432, 756]),
         ("Wizard Whitebeard", 3, [148, 20, 488, 774])]
records = []
for name, page_index, crop in specs:
    page = doc[page_index]
    # Text removal leaves the character's vector drawing intact. In particular,
    # the Whitebeard heading overlaps the enclosing artwork rectangle.
    for block in page.get_text("blocks"):
        page.add_redact_annot(pymupdf.Rect(block[:4]), fill=False)
    page.apply_redactions(images=0, graphics=0)
    pix = page.get_pixmap(matrix=pymupdf.Matrix(2, 2), clip=pymupdf.Rect(crop), alpha=True)
    image = Image.frombytes("RGBA", (pix.width, pix.height), pix.samples)
    bounds = image.getchannel("A").getbbox()
    image = image.crop(bounds)
    image.thumbnail((160, 160), Image.Resampling.LANCZOS)
    path = object_dir / f"{name}.png"
    image.save(path)
    records.append({"name": name, "page": page_index+1, "crop_points": crop,
                    "size": list(image.size), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
manifest = {"url": SOURCE, "downloaded": "2026-09-06",
    "pdf_sha256": hashlib.sha256(pdf_path.read_bytes()).hexdigest(),
    "credit": "Where's Waldo? illustrations by Martin Handford; Candlewick Press character sheets. See original PDF for copyright notice.",
    "preparation": "Remove PDF text only; render clipped vector artwork with alpha at 2x; crop alpha bounds; fit within 160x160 using Lanczos.",
    "class_change": "Wenda replaces Wilma with project-owner approval; original Drive sources returned 404.",
    "objects": records}
(source_dir / "characters.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
print(json.dumps(records, indent=2))
