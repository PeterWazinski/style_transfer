"""Training utilities: COCO dataset loader, normalisation helpers."""
from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class CocoImageDataset(Dataset[torch.Tensor]):
    """Flat image dataset — walks a directory tree and loads all JPEG/PNG files.

    Compatible with the MS-COCO 2017 training folder structure:
        train2017/
            images/
                0001.jpg
                ...
    or any flat folder of JPEG/PNG images.
    """

    SUPPORTED_SUFFIXES: frozenset[str] = frozenset({".jpg", ".jpeg", ".png"})

    def __init__(self, root: Path, image_size: int = 256) -> None:
        self.paths: list[Path] = sorted(
            p for p in root.rglob("*") if p.suffix.lower() in self.SUPPORTED_SUFFIXES
        )
        if not self.paths:
            raise FileNotFoundError(f"No JPEG/PNG images found under {root}")

        self.transform: transforms.Compose = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Lambda(lambda t: t.mul(255.0)),  # keep [0, 255] range
        ])

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(img)  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Style image loader
# ---------------------------------------------------------------------------

def load_style_tensor(path: Path, size: int | None = None) -> torch.Tensor:
    """Load a style reference image as a [1, 3, H, W] float tensor in [0, 255].

    Args:
        path: Path to a JPEG or PNG file.
        size: If given, resize the style image to this dimension.

    Returns:
        Tensor of shape [1, 3, H, W].
    """
    img = Image.open(path).convert("RGB")
    ops: list[object] = []
    if size is not None:
        ops.append(transforms.Resize(size))
    ops += [transforms.ToTensor(), transforms.Lambda(lambda t: t.mul(255.0))]
    tf = transforms.Compose(ops)  # type: ignore[arg-type]
    return tf(img).unsqueeze(0)  # type: ignore[return-value]
