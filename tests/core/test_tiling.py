"""Unit tests for the tiling module (split_tiles / merge_tiles)."""
from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from src.core.tiling import TileInfo, merge_tiles, split_tiles


# ---------------------------------------------------------------------------
# split_tiles
# ---------------------------------------------------------------------------

def test_split_produces_tiles(sample_256: Image.Image) -> None:
    tiles = split_tiles(sample_256, tile_size=128, overlap=16)
    assert len(tiles) > 0


def test_each_tile_is_pil_image(sample_256: Image.Image) -> None:
    for _info, tile in split_tiles(sample_256, tile_size=128, overlap=16):
        assert isinstance(tile, Image.Image)


def test_tile_size_does_not_exceed_request(sample_256: Image.Image) -> None:
    tile_size = 128
    for _info, tile in split_tiles(sample_256, tile_size=tile_size, overlap=16):
        w, h = tile.size
        assert w <= tile_size
        assert h <= tile_size


def test_overlap_too_large_raises() -> None:
    img = Image.new("RGB", (256, 256))
    with pytest.raises(ValueError, match="stride"):
        split_tiles(img, tile_size=64, overlap=32)  # stride = 64-64 = 0


def test_single_tile_for_small_image() -> None:
    img = Image.new("RGB", (64, 64))
    tiles = split_tiles(img, tile_size=512, overlap=64)
    assert len(tiles) == 1


# ---------------------------------------------------------------------------
# merge_tiles — round-trip size preservation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("w,h,tile_size,overlap", [
    (256, 256, 128, 16),
    (512, 384, 256, 32),
    (300, 200, 128, 16),   # non-multiple dimensions
])
def test_round_trip_preserves_size(w: int, h: int, tile_size: int, overlap: int) -> None:
    img = Image.fromarray(
        np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
    )
    tiles = split_tiles(img, tile_size=tile_size, overlap=overlap)
    result = merge_tiles(tiles, (w, h))
    assert result.size == (w, h), f"Expected ({w},{h}), got {result.size}"


def test_round_trip_output_is_rgb() -> None:
    img = Image.new("RGB", (128, 128), color=(100, 150, 200))
    tiles = split_tiles(img, tile_size=64, overlap=8)
    result = merge_tiles(tiles, (128, 128))
    assert result.mode == "RGB"


# ---------------------------------------------------------------------------
# merge_tiles — seam smoothness (no hard edges)
# ---------------------------------------------------------------------------

def test_seam_gradient_is_smooth() -> None:
    """A uniformly-coloured image round-tripped through tiling must stay uniform."""
    colour = (128, 64, 192)
    img = Image.new("RGB", (256, 256), color=colour)
    tiles = split_tiles(img, tile_size=128, overlap=32)
    result = merge_tiles(tiles, (256, 256))
    arr = np.array(result, dtype=float)
    # Mean deviation from expected solid colour must be very small
    expected = np.array(colour, dtype=float)
    deviation = np.abs(arr - expected).mean()
    assert deviation < 2.0, f"Seam artefact detected: mean deviation = {deviation:.2f}"


def test_no_hard_edge_at_tile_boundary() -> None:
    """Max pixel-to-pixel gradient at a tile seam must stay below threshold."""
    arr_in = np.ones((256, 256, 3), dtype=np.uint8) * 128
    img = Image.fromarray(arr_in)
    tiles = split_tiles(img, tile_size=128, overlap=32)
    result = merge_tiles(tiles, (256, 256))
    arr_out = np.array(result, dtype=float)
    # Check horizontal seam at tile boundary column = 128
    seam_col = 128
    grad = np.abs(arr_out[:, seam_col, :] - arr_out[:, seam_col - 1, :]).max()
    assert grad < 10.0, f"Hard edge at seam: max gradient = {grad:.1f}"
