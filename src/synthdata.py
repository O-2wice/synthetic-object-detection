"""Procedural dataset for single-object detection.

The original coursework built its dataset by scraping Google Images for
backgrounds and pasting in Where's Wally character cut-outs. Neither survives
publication: the scrape is not reproducible, and the characters are
copyrighted. Everything here is drawn from a seed instead, so the dataset
regenerates byte-identically on any machine and carries no third-party assets.

Each sample is one object placed at a random position on a cluttered background,
labelled in YOLO format: `class cx cy w h`, normalized to the image size.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw

CLASS_NAMES = ["striped", "spotted", "chevron"]
NUM_CLASSES = len(CLASS_NAMES)

# Distinct palettes so the three classes are separable by colour as well as
# pattern. A model that only learned colour would still have to localise.
CLASS_PALETTES = [
    ((214, 40, 40), (255, 255, 255)),    # striped: red / white
    ((29, 53, 87), (241, 250, 238)),     # spotted: navy / off-white
    ((247, 181, 56), (61, 64, 91)),      # chevron: amber / slate
]


@dataclass(frozen=True)
class Box:
    """Axis-aligned box in normalized centre form."""

    cx: float
    cy: float
    w: float
    h: float

    def to_corners(self) -> tuple[float, float, float, float]:
        return (self.cx - self.w / 2, self.cy - self.h / 2,
                self.cx + self.w / 2, self.cy + self.h / 2)

    def as_tuple(self) -> tuple[float, float, float, float]:
        return (self.cx, self.cy, self.w, self.h)


def draw_object(rng: random.Random, class_id: int, size: int) -> Image.Image:
    """Draw one object on a transparent canvas.

    The three classes share a silhouette but differ in pattern and palette, so
    the classification head has to attend to appearance rather than shape alone.
    """
    primary, secondary = CLASS_PALETTES[class_id]
    canvas = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(canvas)

    pad = size // 10
    body = (pad, pad, size - pad, size - pad)
    draw.ellipse(body, fill=primary + (255,))

    if class_id == 0:  # horizontal stripes
        step = max(3, size // 7)
        for offset in range(pad, size - pad, step * 2):
            draw.rectangle((pad, offset, size - pad, offset + step), fill=secondary + (255,))
    elif class_id == 1:  # scattered dots
        radius = max(2, size // 12)
        for _ in range(7):
            x = rng.randint(pad + radius, size - pad - radius)
            y = rng.randint(pad + radius, size - pad - radius)
            draw.ellipse((x - radius, y - radius, x + radius, y + radius),
                         fill=secondary + (255,))
    else:  # nested chevrons
        for k in range(3):
            inset = pad + k * max(3, size // 9)
            draw.line([(inset, size * 0.62), (size / 2, inset + size * 0.12),
                       (size - inset, size * 0.62)],
                      fill=secondary + (255,), width=max(2, size // 14))

    # Re-apply the ellipse as a mask so patterns cannot spill outside the body.
    mask = Image.new("L", (size, size), 0)
    ImageDraw.Draw(mask).ellipse(body, fill=255)
    canvas.putalpha(mask)
    return canvas


def draw_background(rng: random.Random, width: int, height: int) -> Image.Image:
    """Draw cluttered doodle-style clutter for the object to hide in.

    Clutter matters: on a plain background the task collapses to finding the
    only non-uniform region, and the detector would not have to learn anything
    about the objects themselves.
    """
    background = Image.new("RGB", (width, height), (250, 249, 246))
    draw = ImageDraw.Draw(background)

    for _ in range(90):
        shape = rng.choice(("line", "circle", "rect", "arc"))
        colour = tuple(rng.randint(60, 235) for _ in range(3))
        x1, y1 = rng.randint(0, width), rng.randint(0, height)
        x2, y2 = x1 + rng.randint(-90, 90), y1 + rng.randint(-90, 90)
        box = (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))

        if shape == "line":
            draw.line((x1, y1, x2, y2), fill=colour, width=rng.randint(1, 4))
        elif shape == "circle":
            draw.ellipse(box, outline=colour, width=rng.randint(1, 4))
        elif shape == "rect":
            draw.rectangle(box, outline=colour, width=rng.randint(1, 4))
        else:
            start = rng.randint(0, 300)
            draw.arc(box, start=start, end=start + rng.randint(40, 300),
                     fill=colour, width=rng.randint(1, 4))

    return background


def load_object_sprites(directory) -> list[Image.Image] | None:
    """Load user-supplied object cut-outs, if any are present.

    Drop transparent PNGs into `assets/objects/` named `0_*.png`, `1_*.png`,
    `2_*.png` (the prefix is the class id) and they replace the drawn shapes.
    This is the hook for running the pipeline on real cut-outs, such as the
    character sprites the original coursework used. When the directory is empty
    or missing, everything falls back to the procedural objects so the dataset
    still regenerates anywhere.
    """
    from pathlib import Path

    directory = Path(directory)
    if not directory.is_dir():
        return None

    sprites: list[Image.Image | None] = [None] * NUM_CLASSES
    for path in sorted(directory.glob("*.png")):
        try:
            class_id = int(path.name.split("_")[0])
        except ValueError:
            continue
        if 0 <= class_id < NUM_CLASSES and sprites[class_id] is None:
            sprites[class_id] = Image.open(path).convert("RGBA")

    return sprites if all(s is not None for s in sprites) else None


def make_sample(seed: int, image_size: int = 224,
                min_scale: float = 0.16, max_scale: float = 0.34,
                sprites: list[Image.Image] | None = None
                ) -> tuple[Image.Image, int, Box]:
    """Build one (image, class_id, box) sample deterministically from `seed`.

    Pass `sprites` to composite real cut-outs instead of the drawn objects; the
    labelling, placement and augmentation logic is identical either way.
    """
    rng = random.Random(seed)

    background = draw_background(rng, image_size, image_size)
    class_id = rng.randrange(NUM_CLASSES)

    side = int(image_size * rng.uniform(min_scale, max_scale))
    if sprites is not None:
        sprite = sprites[class_id].copy()
        sprite.thumbnail((side, side), Image.LANCZOS)
    else:
        sprite = draw_object(rng, class_id, side)
    sprite = sprite.rotate(rng.uniform(-25, 25), resample=Image.BICUBIC, expand=True)

    # Crop back to the drawn pixels so the label matches the visible object
    # rather than the transparent canvas it was rotated inside.
    bbox = sprite.getbbox()
    sprite = sprite.crop(bbox)
    obj_w, obj_h = sprite.size

    x = rng.randint(0, max(0, image_size - obj_w))
    y = rng.randint(0, max(0, image_size - obj_h))
    background.paste(sprite, (x, y), sprite)

    box = Box(
        cx=(x + obj_w / 2) / image_size,
        cy=(y + obj_h / 2) / image_size,
        w=obj_w / image_size,
        h=obj_h / image_size,
    )
    return background, class_id, box


def horizontal_flip(image: Image.Image, box: Box) -> tuple[Image.Image, Box]:
    """Mirror an image and its box together.

    The coursework applied RandomHorizontalFlip to the image alone and left the
    label untouched, so roughly half of every epoch taught the model a box on
    the wrong side. Any geometric augmentation has to move the label with the
    pixels, which is why it lives here rather than in a torchvision Compose.
    """
    return image.transpose(Image.FLIP_LEFT_RIGHT), Box(1.0 - box.cx, box.cy, box.w, box.h)


def iou(a: Box, b: Box) -> float:
    """Intersection over union of two normalized centre-form boxes."""
    ax1, ay1, ax2, ay2 = a.to_corners()
    bx1, by1, bx2, by2 = b.to_corners()

    inter_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0.0, min(ay2, by2) - max(ay1, by1))
    intersection = inter_w * inter_h

    union = a.w * a.h + b.w * b.h - intersection
    return intersection / union if union > 0 else 0.0
