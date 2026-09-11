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
it before generation. Saved files, not a repeated search, reproduce this
collection. Original source URLs and attribution remain recorded with the saved assets.

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

The preserved notebook samples from the same background pool for all three
splits, as the original did. It creates 5,000 training, 1,000 validation and 200
test scenes at 640 × 640 pixels, each with one randomly selected cut-out and
its normalized bounding box. Test scenes are new composites of familiar source
artwork; they do not measure generalization to unseen backgrounds.

The fixed local archive is `data/distribution/synthetic-scenes-v1.zip`.
`assets/sources/dataset-archive.sha256` identifies the archive, and
`assets/sources/dataset-manifest.json` records source hashes, split sizes, seed
and hashes of all generated images and labels. Colab extracts and verifies this
same archive; it does not generate a substitute dataset.

Generated composites remain under the ignored `data/` directory. Choose a new
generated directory when changing source assets or preprocessing.
Both cut-outs and source backgrounds are reused across splits, matching the
original generation method.

Earlier smoke/pilot artifacts belong to the discarded rewrite and are not
validation results for the preserved notebook.

The same dataset archive is also available as the synthetic-scenes-v1.zip
asset on GitHub release dataset-v1. Colab downloads it automatically alongside
the source PNGs and verifies its saved checksum.
