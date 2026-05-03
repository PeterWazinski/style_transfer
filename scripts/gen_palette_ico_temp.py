"""One-off script to generate assets/palette.ico."""
from __future__ import annotations
import math
import os
from pathlib import Path
from PIL import Image, ImageDraw


def make_palette_pil(size: int = 256) -> Image.Image:
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    m = max(2, size // 12)
    r = size // 5
    draw.rounded_rectangle([m, m * 2, size - m, size - m], radius=r, fill=(220, 185, 130, 255))
    cx, cy = size // 3, size // 3
    hole_r = max(2, size // 9)
    draw.ellipse([cx - hole_r, cy - hole_r, cx + hole_r, cy + hole_r], fill=(0, 0, 0, 0))
    blob_r = max(2, size // 10)
    radius = size * 0.3
    colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6"]
    for i, color in enumerate(colors):
        angle = math.pi * 0.05 + i * math.pi * 0.19
        bx = size * 0.65 + radius * math.cos(angle)
        by = size * 0.35 - radius * math.sin(angle)
        draw.ellipse(
            [int(bx - blob_r), int(by - blob_r), int(bx + blob_r), int(by + blob_r)],
            fill=color,
        )
    return img


if __name__ == "__main__":
    dest = Path(__file__).parent.parent / "assets" / "palette.ico"
    dest.parent.mkdir(exist_ok=True)
    base = make_palette_pil(256)
    base.save(str(dest), format="ICO", sizes=[(16, 16), (32, 32), (48, 48), (64, 64), (256, 256)])
    print(f"Generated {dest} ({os.path.getsize(dest)} bytes)")
