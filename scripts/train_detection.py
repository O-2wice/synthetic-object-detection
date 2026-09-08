"""Train from a generated dataset; all options also work in the notebook."""
raise SystemExit('Retired: this trainer uses the discarded architecture. Train with notebooks/object-detection.ipynb in Colab.')
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from detection import train, train_yolo


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("data/experiment-generated-v2"))
    parser.add_argument("--output", type=Path, default=Path("outputs/models/experiment-v3"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--yolo", action="store_true")
    args = parser.parse_args()
    if args.yolo:
        train_yolo(args.dataset, args.output, epochs=args.epochs)
    else:
        train(args.dataset, args.output, epochs=args.epochs,
              batch_size=args.batch_size, resume=args.resume)


if __name__ == "__main__":
    main()
