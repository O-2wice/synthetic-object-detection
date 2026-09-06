# Detection pipeline review

The requested nested source path was absent. The original was available as
`HW1_Image_Object detection_2 (3).ipynb` in the parent workspace. The existing
project notebook still carried the exercise structure and embedded results.
The original workspace notebook is unchanged; previous project versions remain
in Git history.

| Finding | Revised behavior |
| --- | --- |
| Model summary called a class before its definition | Import model before construction |
| `models` overwritten with plot labels | No module-name collision |
| Stochastic validation flips and color jitter | Training-only augmentation |
| Shared background pool across splits | Split exact source identities before compositing |
| Invalid assets could cause an endless generation loop | Validate assets before generation |
| Missing labels became padded class-0 targets | Require exactly one valid target |
| Visualization discarded class 0 | Draw every valid class, including Waldo |
| 640px test transform recreated after 224px training | Common dataset and input-size configuration |
| Evaluation could fall back to random weights | Load successfully trained best checkpoint |
| Box head discarded spatial layout | Retain a 4 × 4 feature grid (architecture change) |
| Unbounded boxes | Constrain the entire box to the image |
| First revised pilot saturated the box sigmoid | Layer normalization and small output-weight initialization; fixed-batch regression test |
| Asynchronous, per-batch timing | Warmed, synchronized per-image forward timing |
| Last YOLO training CSV row used as performance | Reload best checkpoint and evaluate test split |
| Incompatible metric protocols | Shared one-box protocol; separate native YOLO metrics |
| Old outputs beside changed code | Remove stale scores; label new evidence by run mode |

The spatial head, constrained boxes and AdamW optimizer change the experiment.
Old checkpoints and performance claims do not transfer to this revision.

Two earlier explanations also needed correction: axis-aligned bounds of a
rotated object can be computed; dropping rotation is a simplicity choice.
Global pooling does not make input resolution irrelevant. Using 224px is a
compute/accuracy tradeoff, not equivalence to 640px.

## Validation scope

Regression tests cover class 0, IoU threshold handling, AP confidence ranking,
missing predictions, source separation, valid labels, paired flips, valid model
outputs, gradients and checkpoint resume. Notebook smoke mode executes generation,
data loading, two training epochs, selected-checkpoint evaluation, JSON export
and figures using geometric fixtures. Smoke mode skips YOLO.

Replacement publisher cut-outs and 22 visually reviewed backgrounds are now
saved in `assets/`. The new classes are Waldo, Wenda and Wizard Whitebeard.
Before a public release, run the full character
experiment including YOLO, review convergence and predictions, and check the
rendered report against the actual artifacts. The repository stays private.

Initial checks used the existing colorization environment without modifying it.
The detection project's own environment now uses Python 3.11, PyTorch 2.14.0+cpu,
torchvision 0.29.0+cpu and Ultralytics 8.4.142. `requirements-lock.txt` records the
complete local package versions. No CUDA training result is claimed.

## Implementation references

- [Torchvision ResNet18 weights](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html)
- [Ultralytics training](https://docs.ultralytics.com/modes/train/)
- [Ultralytics validation and test splits](https://docs.ultralytics.com/modes/val/)

Original task template: Tamás Takács and Imre Molnár, ELTE. Original implementation:
Robert Ouko Oyombe.
