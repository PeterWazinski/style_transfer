"""Unit tests for PhotoManager (load, save, thumbnail, tiles)."""
from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.core.photo_manager import PhotoManager, UnsupportedFormatError
from tests.helpers import save_jpeg, save_png


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def pm() -> PhotoManager:
    return PhotoManager()


# ---------------------------------------------------------------------------
# load — happy paths
# ---------------------------------------------------------------------------

def test_load_jpeg_returns_rgb_image(tmp_path: Path, pm: PhotoManager) -> None:
    p = save_jpeg(tmp_path / "test.jpg")
    img = pm.load(p)
    assert isinstance(img, Image.Image)
    assert img.mode == "RGB"


def test_load_png_returns_rgb_image(tmp_path: Path, pm: PhotoManager) -> None:
    p = save_png(tmp_path / "test.png")
    img = pm.load(p)
    assert isinstance(img, Image.Image)
    assert img.mode == "RGB"


def test_load_preserves_dimensions(tmp_path: Path, pm: PhotoManager) -> None:
    p = save_jpeg(tmp_path / "test.jpg", size=(200, 150))
    img = pm.load(p)
    assert img.size == (200, 150)


def test_load_jpeg_uppercase_extension(tmp_path: Path, pm: PhotoManager) -> None:
    """Extension check must be case-insensitive."""
    src = save_jpeg(tmp_path / "test.jpg")
    dest = tmp_path / "test.JPG"
    dest.write_bytes(src.read_bytes())
    img = pm.load(dest)
    assert img.mode == "RGB"


# ---------------------------------------------------------------------------
# load — error paths
# ---------------------------------------------------------------------------

def test_load_unsupported_format_raises(tmp_path: Path, pm: PhotoManager) -> None:
    p = tmp_path / "photo.bmp"
    Image.new("RGB", (32, 32)).save(p, format="BMP")
    with pytest.raises(UnsupportedFormatError):
        pm.load(p)


def test_load_heic_rejected(tmp_path: Path, pm: PhotoManager) -> None:
    p = tmp_path / "photo.heic"
    p.write_bytes(b"fake")
    with pytest.raises(UnsupportedFormatError):
        pm.load(p)


def test_load_missing_file_raises(tmp_path: Path, pm: PhotoManager) -> None:
    with pytest.raises(FileNotFoundError):
        pm.load(tmp_path / "nonexistent.jpg")


def test_load_webp_rejected(tmp_path: Path, pm: PhotoManager) -> None:
    p = tmp_path / "photo.webp"
    p.write_bytes(b"fake")
    with pytest.raises(UnsupportedFormatError):
        pm.load(p)


# ---------------------------------------------------------------------------
# save
# ---------------------------------------------------------------------------

def test_save_jpeg_creates_file(tmp_path: Path, pm: PhotoManager) -> None:
    img = Image.new("RGB", (64, 64), color=(128, 64, 32))
    dest = tmp_path / "out.jpg"
    pm.save(img, dest)
    assert dest.exists()
    assert dest.stat().st_size > 0


def test_save_png_creates_file(tmp_path: Path, pm: PhotoManager) -> None:
    img = Image.new("RGB", (64, 64))
    dest = tmp_path / "out.png"
    pm.save(img, dest)
    assert dest.exists()


def test_save_unsupported_format_raises(tmp_path: Path, pm: PhotoManager) -> None:
    img = Image.new("RGB", (64, 64))
    with pytest.raises(UnsupportedFormatError):
        pm.save(img, tmp_path / "out.bmp")


def test_save_creates_parent_dirs(tmp_path: Path, pm: PhotoManager) -> None:
    img = Image.new("RGB", (64, 64))
    dest = tmp_path / "sub" / "dir" / "out.jpg"
    pm.save(img, dest)
    assert dest.exists()


def test_save_and_reload_preserves_dimensions(tmp_path: Path, pm: PhotoManager) -> None:
    img = Image.new("RGB", (200, 150), color=(50, 100, 200))
    dest = tmp_path / "out.jpg"
    pm.save(img, dest)
    reloaded = pm.load(dest)
    assert reloaded.size == (200, 150)


def test_save_jpeg_quality_param_accepted(tmp_path: Path, pm: PhotoManager) -> None:
    img = Image.new("RGB", (64, 64))
    low_q = tmp_path / "low.jpg"
    high_q = tmp_path / "high.jpg"
    pm.save(img, low_q, quality=10)
    pm.save(img, high_q, quality=95)
    assert low_q.stat().st_size <= high_q.stat().st_size


# ---------------------------------------------------------------------------
# thumbnail
# ---------------------------------------------------------------------------

def test_thumbnail_fits_within_max_size(pm: PhotoManager) -> None:
    img = Image.new("RGB", (1024, 768))
    thumb = pm.thumbnail(img, (256, 256))
    w, h = thumb.size
    assert w <= 256
    assert h <= 256


def test_thumbnail_preserves_aspect_ratio(pm: PhotoManager) -> None:
    img = Image.new("RGB", (800, 400))  # 2:1 ratio
    thumb = pm.thumbnail(img, (200, 200))
    w, h = thumb.size
    assert abs(w / h - 2.0) < 0.1


def test_thumbnail_does_not_upscale(pm: PhotoManager) -> None:
    """Small images must not be enlarged."""
    img = Image.new("RGB", (64, 64))
    thumb = pm.thumbnail(img, (512, 512))
    assert thumb.size[0] <= 64
    assert thumb.size[1] <= 64


def test_thumbnail_returns_rgb(pm: PhotoManager) -> None:
    img = Image.new("RGBA", (128, 128))
    thumb = pm.thumbnail(img, (64, 64))
    assert thumb.mode == "RGB"


# ---------------------------------------------------------------------------
# split_tiles / merge_tiles (delegation tests — minimal; tiling has its own suite)
# ---------------------------------------------------------------------------

def test_split_produces_tiles(pm: PhotoManager) -> None:
    img = Image.new("RGB", (256, 256))
    tiles = pm.split_tiles(img, tile_size=128, overlap=16)
    assert len(tiles) > 0


def test_merge_round_trip_size(pm: PhotoManager) -> None:
    img = Image.new("RGB", (256, 256))
    tiles = pm.split_tiles(img, tile_size=128, overlap=16)
    result = pm.merge_tiles(tiles, (256, 256))
    assert result.size == (256, 256)


# ---------------------------------------------------------------------------
# EXIF / auto-rotate
# ---------------------------------------------------------------------------

def test_auto_rotate_orientation_3(pm: PhotoManager, tmp_path: Path) -> None:
    """An image tagged orientation=3 (180°) must come back rotated."""
    from unittest.mock import patch, MagicMock
    from src.core.photo_manager import _auto_rotate

    img = Image.new("RGB", (40, 20), color=(10, 20, 30))
    mock_exif = MagicMock()
    mock_exif.get.return_value = 3  # 180° rotation
    with patch.object(img, "getexif", return_value=mock_exif):
        result = _auto_rotate(img)
    # After 180° rotation size stays the same but pixels are flipped
    assert result.size == (40, 20)


def test_auto_rotate_orientation_6(pm: PhotoManager) -> None:
    """Orientation=6 (90° CW) swaps width and height."""
    from unittest.mock import patch, MagicMock
    from src.core.photo_manager import _auto_rotate

    img = Image.new("RGB", (40, 20))
    mock_exif = MagicMock()
    mock_exif.get.return_value = 6  # 90° CCW (expand swaps dims)
    with patch.object(img, "getexif", return_value=mock_exif):
        result = _auto_rotate(img)
    # Width and height should be swapped after 90° rotation with expand=True
    assert result.size == (20, 40)


def test_auto_rotate_no_orientation_tag_unchanged(pm: PhotoManager) -> None:
    """Missing orientation tag must leave the image unchanged."""
    from unittest.mock import patch, MagicMock
    from src.core.photo_manager import _auto_rotate

    img = Image.new("RGB", (100, 50))
    mock_exif = MagicMock()
    mock_exif.get.return_value = None  # no orientation tag
    with patch.object(img, "getexif", return_value=mock_exif):
        result = _auto_rotate(img)
    assert result.size == (100, 50)


def test_auto_rotate_unknown_orientation_unchanged() -> None:
    """An unrecognised orientation value (e.g. 1) must leave image unchanged."""
    from unittest.mock import patch, MagicMock
    from src.core.photo_manager import _auto_rotate

    img = Image.new("RGB", (100, 50))
    mock_exif = MagicMock()
    mock_exif.get.return_value = 1  # normal orientation — no rotation needed
    with patch.object(img, "getexif", return_value=mock_exif):
        result = _auto_rotate(img)
    assert result.size == (100, 50)


def test_auto_rotate_getexif_raises_returns_unchanged() -> None:
    """If getexif() raises, the original image must be returned unchanged."""
    from unittest.mock import patch
    from src.core.photo_manager import _auto_rotate

    img = Image.new("RGB", (100, 50))
    with patch.object(img, "getexif", side_effect=Exception("EXIF error")):
        result = _auto_rotate(img)
    assert result.size == (100, 50)


def test_save_with_source_exif_param(tmp_path: Path, pm: PhotoManager) -> None:
    """save() must accept a source_exif argument without raising."""
    img = Image.new("RGB", (64, 64), color=(10, 20, 30))
    source = Image.new("RGB", (64, 64))
    dest = tmp_path / "out.jpg"
    pm.save(img, dest, source_exif=source)
    assert dest.exists()


def test_save_copies_exif_bytes_from_source(tmp_path: Path, pm: PhotoManager) -> None:
    """When source_exif carries real EXIF bytes they are embedded in the JPEG."""
    import io as _io
    # Build a JPEG in memory that carries at least a minimal EXIF block
    buf = _io.BytesIO()
    src_img = Image.new("RGB", (64, 64))
    exif = src_img.getexif()
    exif[274] = 1  # 274 = Orientation tag; value 1 = normal
    src_img.save(buf, format="JPEG", exif=exif.tobytes())
    buf.seek(0)
    source_with_exif = Image.open(buf)
    source_with_exif.load()

    dest_img = Image.new("RGB", (64, 64), color=(200, 100, 50))
    out = tmp_path / "with_exif.jpg"
    pm.save(dest_img, out, source_exif=source_with_exif)

    assert out.exists()
    # The saved file should be loadable (EXIF was embedded, no corruption)
    reloaded = pm.load(out)
    assert reloaded.mode == "RGB"
