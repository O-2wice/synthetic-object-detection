"""Figures shared by the notebook and static report."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from detection import MEAN, STD, NAMES


def plot_samples(dataset, destination, predictions=None, count=6):
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    for i, ax in enumerate(axes.flat):
        ax.axis("off")
        if i >= min(count, len(dataset)):
            continue
        tensor, target = dataset[i]
        image = np.clip(tensor.numpy().transpose(1, 2, 0) * STD + MEAN, 0, 1)
        ax.imshow(image)
        h, w = image.shape[:2]
        boxes = [(target[1:].tolist(), "lime", "label")]
        if predictions is not None and predictions[i] is not None:
            boxes.append((predictions[i][2], "red", "prediction"))
        for (cx, cy, bw, bh), color, label in boxes:
            ax.add_patch(Rectangle(((cx-bw/2)*w, (cy-bh/2)*h), bw*w, bh*h,
                                   edgecolor=color, facecolor="none", linewidth=2, label=label))
        ax.set_title(NAMES[int(target[0])])
        ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    Path(destination).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=140)
    return fig


def plot_history(history, destination):
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    for ax, key in zip(axes, ["total", "classification", "box"]):
        for split in ["train", "val"]:
            ax.plot([r["epoch"] for r in history], [r[split][key] for r in history],
                    marker="o", label=split)
        ax.set(xlabel="Epoch", ylabel="Loss", title=key.capitalize())
        ax.legend()
    fig.tight_layout()
    Path(destination).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=140)
    return fig
