"""Tile splitting and Gaussian-blend merging for large image inference.

Strategy:
  - Split the source image into overlapping tiles of `tile_size × tile_size`.
  - Run TransformerNet on each tile independently.
  - Reconstruct the full image by alpha-blending tile boundaries with a
    Gaussian weight mask, eliminating visible seams.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from PIL import Image


@dataclass
class TileInfo:
    """Metadata for a single tile extracted from the source image."""

    row: int          # tile row index (0-based)
    col: int          # tile column index
    x_start: int      # pixel x start (left edge) in source image
    y_start: int      # pixel y start (top edge) in source image
    x_end: int        # pixel x end (exclusive) in source image
    y_end: int        # pixel y end (exclusive)
    # "effective" region — the non-overlapping centre area owned by this tile
    cx_start: int
    cy_start: int
    cx_end: int
    cy_end: int


def split_tiles(
    image: Image.Image,
    tile_size: int = 1024,
    overlap: int = 128,
) -> list[tuple[TileInfo, Image.Image]]:
    """Split *image* into overlapping tiles.

    Args:
        image:     Source PIL image (any size).
        tile_size: Width and height of each tile in pixels.
        overlap:   Number of pixels of overlap on each side.

    Returns:
        List of (TileInfo, tile_image) pairs.
    """
    w, h = image.size
    stride: int = tile_size - 2 * overlap

    if stride <= 0:
        raise ValueError(
            f"overlap ({overlap}) is too large for tile_size ({tile_size}). "
            f"stride = tile_size - 2*overlap must be > 0."
        )

    result: list[tuple[TileInfo, Image.Image]] = []

    def _starts(total: int) -> list[int]:
        """Tile start positions that guarantee full image coverage."""
        if total <= tile_size:
            return [0]
        starts: list[int] = []
        pos = 0
        while pos + tile_size < total:
            starts.append(pos)
            pos += stride
        starts.append(max(0, total - tile_size))  # last tile always ends at edge
        # Deduplicate while preserving order
        seen: set[int] = set()
        deduped: list[int] = []
        for s in starts:
            if s not in seen:
                seen.add(s)
                deduped.append(s)
        return deduped

    col_starts = _starts(w)
    row_starts = _starts(h)

    for row, y0_abs in enumerate(row_starts):
        for col, x0_abs in enumerate(col_starts):
            x0, y0 = x0_abs, y0_abs
            x1 = min(w, x0 + tile_size)
            y1 = min(h, y0 + tile_size)

            # Effective centre region: halfway between this start and next start
            cx0 = x0 if col == 0 else x0 + overlap
            cy0 = y0 if row == 0 else y0 + overlap
            cx1 = x1 if col == len(col_starts) - 1 else x1 - overlap
            cy1 = y1 if row == len(row_starts) - 1 else y1 - overlap
            cx0, cy0 = min(cx0, x1), min(cy0, y1)
            cx1, cy1 = max(cx1, cx0), max(cy1, cy0)

            info = TileInfo(
                row=row, col=col,
                x_start=x0, y_start=y0,
                x_end=x1, y_end=y1,
                cx_start=cx0, cy_start=cy0,
                cx_end=cx1, cy_end=cy1,
            )
            tile: Image.Image = image.crop((x0, y0, x1, y1))
            result.append((info, tile))

    return result


def _gaussian_weight_mask(h: int, w: int, sigma_ratio: float = 0.25) -> np.ndarray:
    """Generate a 2-D Gaussian weight mask of shape (h, w)."""
    cy, cx = h / 2.0, w / 2.0
    sigma_y = h * sigma_ratio
    sigma_x = w * sigma_ratio
    ys = np.arange(h, dtype=np.float32)
    xs = np.arange(w, dtype=np.float32)
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    mask = np.exp(
        -0.5 * ((yy - cy) ** 2 / sigma_y ** 2 + (xx - cx) ** 2 / sigma_x ** 2)
    )
    return mask.astype(np.float32)


def merge_tiles(
    tiles: list[tuple[TileInfo, Image.Image]],
    output_size: tuple[int, int],
) -> Image.Image:
    """Reconstruct a full image from styled tiles using Gaussian blending.

    Args:
        tiles:       List of (TileInfo, styled_tile) as returned by split_tiles
                     (with the tile replaced by its styled version).
        output_size: (width, height) of the reconstruction target.

    Returns:
        Reconstructed PIL image.
    """
    w, h = output_size
    acc = np.zeros((h, w, 3), dtype=np.float32)
    weight_acc = np.zeros((h, w), dtype=np.float32)

    for info, tile_img in tiles:
        tile_arr = np.array(tile_img.convert("RGB"), dtype=np.float32)
        th, tw = tile_arr.shape[:2]
        mask = _gaussian_weight_mask(th, tw)

        acc[info.y_start:info.y_end, info.x_start:info.x_end] += (
            tile_arr * mask[:, :, np.newaxis]
        )
        weight_acc[info.y_start:info.y_end, info.x_start:info.x_end] += mask

    # Avoid division by zero (shouldn't happen for valid tile configs)
    weight_acc = np.where(weight_acc < 1e-6, 1.0, weight_acc)
    result_arr = np.clip(acc / weight_acc[:, :, np.newaxis], 0, 255).astype(np.uint8)
    return Image.fromarray(result_arr)
