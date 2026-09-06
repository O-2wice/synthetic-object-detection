# Data preparation

The fresh experiment uses Waldo, Wenda and Wizard Whitebeard. Wenda replaces the
original Wilma class with the project owner's approval because the old links
are unavailable. Source files and prepared assets are saved inside the project.

```text
assets/objects/Waldo.png
assets/objects/Wenda.png
assets/objects/Wizard Whitebeard.png
assets/backgrounds/*.jpg
assets/backgrounds/review.json
assets/sources/WaldoCharacterSheets20.pdf
assets/sources/characters.json
assets/sources/backgrounds.json
```

The character source is [Candlewick Press's character sheets](https://waldo.candlewick.com/pdf/WaldoCharacterSheets20.pdf).
The original PDF is retained with its attribution. `scripts/prepare_publisher_assets.py`
reproduces the PNGs: remove text from the PDF rendering, isolate each character's
vector artwork, preserve its alpha channel and fit it within 160 × 160 pixels.
The script records crop rectangles, page numbers and file hashes. The generator
keeps those prepared dimensions when compositing onto 640-pixel backgrounds.

`scripts/crawl_backgrounds.py` downloaded 40 doodle-pattern candidates and recorded
their source URLs and SHA-256 hashes. Contact-sheet review excluded 18 related
variants, previews and unsuitable compositions, leaving 22 backgrounds.
The exclusion list is saved in `assets/backgrounds/review.json`; generation reads
it before splitting sources. Saved files, not a repeated search, reproduce this
collection. These sources have not been cleared for public redistribution.

Run `python scripts/verify_assets.py` to verify all 44 saved files against their
hashes. `--restore-missing` re-downloads missing source files only when their
contents still match the saved hash. Recreate missing prepared PNGs with
`python scripts/prepare_publisher_assets.py`. A new crawl cannot reproduce a
search engine's earlier results and deliberately refuses to overwrite this set.

The original notebook referenced these Google Drive files:

| Class | Original source |
| --- | --- |
| Waldo | [Original Drive link](https://drive.google.com/file/d/1n0JXOdBBV_gQkMmL51eAsJllVW3sejy5/view) |
| Wilma | [Original Drive link](https://drive.google.com/file/d/1SW4Q8HjtYUaW-Xii0in3crU_pzFBDON5/view) |
| Wizard Whitebeard | [Original Drive link](https://drive.google.com/file/d/1-xJC3ygQEclyiOSYZSFrNox_vxEzgwoM/view) |

All three direct-download URLs returned HTTP 404 during the September 6, 2026
review. They are retained as provenance, not working mirrors. Replacement sources
must be recorded separately; a new asset collection is a new experiment.

The original backgrounds came from doodle searches too, but were not saved in
the available checkout. The new collection is not the original training dataset.
Review any replacement collection for near duplicates and target characters.

At least ten distinct readable backgrounds are required. The generator removes
exact decoded duplicates, splits source identities 80% / 10% / 10%, and only then
samples backgrounds for composites. A manifest records object hashes, background
hashes, class counts, generation seed and hashes of every generated image/label.
Training verifies image and label hashes before using the dataset.

Generated composites remain under the ignored `data/` directory. Choose a new
generated directory when changing source assets or preprocessing.
Reusing cut-outs across splits is intentional; reusing backgrounds is not.

Geometric smoke fixtures are created by `scripts/smoke_assets.py`. Their filenames
occupy the same class slots but they are not character data. They never populate
the benchmark results table.
