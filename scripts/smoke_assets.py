"""Create small geometric fixtures for execution checks, never benchmark data."""
from pathlib import Path
import random
from PIL import Image, ImageDraw


def create_assets(root):
    root = Path(root)
    objects, backgrounds = root / "objects", root / "backgrounds"
    objects.mkdir(parents=True, exist_ok=True)
    backgrounds.mkdir(parents=True, exist_ok=True)
    for i, name in enumerate(["Waldo", "Wenda", "Wizard Whitebeard"]):
        image = Image.new("RGBA", (14, 20))
        draw = ImageDraw.Draw(image)
        color = [(220, 50, 50, 255), (40, 170, 80, 255), (60, 90, 220, 255)][i]
        if i == 0:
            draw.rectangle((0, 0, 13, 19), fill=color)
        elif i == 1:
            draw.ellipse((0, 0, 13, 19), fill=color)
        else:
            draw.polygon([(7, 0), (13, 19), (0, 19)], fill=color)
        image.save(objects / f"{name}.png")
    rng = random.Random(91)
    for i in range(20):
        image = Image.new("RGB", (64, 64), (230, 230, 230))
        draw = ImageDraw.Draw(image)
        for _ in range(15):
            draw.line([rng.randrange(64) for _ in range(4)],
                      fill=tuple(rng.randrange(120, 220) for _ in range(3)), width=1)
        image.save(backgrounds / f"{i:03d}.png")
    return objects, backgrounds
