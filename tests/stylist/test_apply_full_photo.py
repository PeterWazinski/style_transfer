"""Integration tests — full photo apply flow.

Tests the pipeline:
  PhotoManager.load()  →  StyleTransferEngine.apply()  →  PhotoManager.save()

The ONNX model is mocked so no real model file is required.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

from src.core.engine import StyleTransferEngine
from src.core.models import StyleModel
from src.core.photo_manager import PhotoManager
from src.core.registry import StyleRegistry


# ---------------------------------------------------------------------------
# Helpers — mock ONNX session (same pattern as test_engine.py)
# ---------------------------------------------------------------------------

def _make_mock_session(
    output_colour: tuple[int, int, int] = (100, 150, 200)
) -> MagicMock:
    session = MagicMock()
    inp = MagicMock()
    inp.name = "input"
    session.get_inputs.return_value = [inp]

    def _run(
        output_names: list[str], feed: dict[str, np.ndarray]
    ) -> list[np.ndarray]:
        tensor = feed["input"]
        h, w = tensor.shape[2], tensor.shape[3]
        rgb = np.full((1, 3, h, w), 0.0, dtype=np.float32)
        rgb[0, 0, :, :] = float(output_colour[0])
        rgb[0, 1, :, :] = float(output_colour[1])
        rgb[0, 2, :, :] = float(output_colour[2])
        return [rgb]

    session.run.side_effect = _run
    return session


def _engine_with_candy() -> StyleTransferEngine:
    engine = StyleTransferEngine()
    with (
        patch("src.core.engine._ORT_AVAILABLE", True),
        patch("src.core.engine.ort") as mock_ort,
        patch.object(Path, "exists", return_value=True),
    ):
        mock_ort.InferenceSession.return_value = _make_mock_session()
        engine.load_model("candy", Path("styles/candy/model.onnx"))
    return engine


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def pm() -> PhotoManager:
    return PhotoManager()


@pytest.fixture()
def engine() -> StyleTransferEngine:
    return _engine_with_candy()


def _write_jpeg(path: Path, w: int, h: int) -> None:
    arr = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
    Image.fromarray(arr).save(path, format="JPEG", quality=90)


# ---------------------------------------------------------------------------
# Full pipeline tests
# ---------------------------------------------------------------------------

def test_load_apply_save_reload_dimensions(
    tmp_path: Path, pm: PhotoManager, engine: StyleTransferEngine
) -> None:
    """Load → apply → save → reload must preserve image dimensions."""
    src = tmp_path / "photo.jpg"
    _write_jpeg(src, 256, 192)

    img = pm.load(src)
    styled = engine.apply(img, "candy", strength=1.0, tile_size=128, overlap=16)
    out = tmp_path / "output.jpg"
    pm.save(styled, out)

    reloaded = pm.load(out)
    assert reloaded.size == img.size, (
        f"Expected {img.size}, got {reloaded.size}"
    )


def test_save_output_is_valid_jpeg(
    tmp_path: Path, pm: PhotoManager, engine: StyleTransferEngine
) -> None:
    """Output file must be readable as a valid JPEG by Pillow."""
    src = tmp_path / "photo.jpg"
    _write_jpeg(src, 128, 128)

    img = pm.load(src)
    styled = engine.apply(img, "candy", strength=0.5, tile_size=128, overlap=16)
    out = tmp_path / "output.jpg"
    pm.save(styled, out)

    with Image.open(out) as reloaded:
        reloaded.verify()  # raises if file is corrupt


def test_pipeline_with_strength_zero_produces_near_original(
    tmp_path: Path, pm: PhotoManager, engine: StyleTransferEngine
) -> None:
    """At strength=0 the output should be very close to the original."""
    src = tmp_path / "photo.jpg"
    _write_jpeg(src, 128, 128)

    img = pm.load(src)
    result = engine.apply(img, "candy", strength=0.0, tile_size=128, overlap=16)

    orig_arr = np.array(img, dtype=float)
    result_arr = np.array(result, dtype=float)
    assert orig_arr.shape == result_arr.shape
    assert np.abs(orig_arr - result_arr).mean() < 5.0


def test_thumbnail_pipeline(
    tmp_path: Path, pm: PhotoManager
) -> None:
    """load → thumbnail must produce correct max-bounded size."""
    src = tmp_path / "large.jpg"
    _write_jpeg(src, 1024, 768)
    img = pm.load(src)
    thumb = pm.thumbnail(img, (256, 256))
    w, h = thumb.size
    assert w <= 256
    assert h <= 256


def test_registry_integration(tmp_path: Path) -> None:
    """StyleRegistry loaded from disk must list the same styles that were saved."""
    catalog = tmp_path / "catalog.json"
    r = StyleRegistry(catalog)
    r.add(StyleModel(
        id="candy",
        name="Candy",
        model_path="styles/candy/model.onnx",
        preview_path="styles/candy/preview.jpg",
        is_builtin=True,
    ))
    r.add(StyleModel(
        id="mosaic",
        name="Mosaic",
        model_path="styles/mosaic/model.onnx",
        preview_path="styles/mosaic/preview.jpg",
        is_builtin=True,
    ))

    r2 = StyleRegistry(catalog)
    styles = r2.list_styles()
    assert len(styles) == 2
    assert {s.id for s in styles} == {"candy", "mosaic"}


@pytest.mark.parametrize("w,h", [(200, 150), (500, 400)])
def test_apply_various_resolutions(
    tmp_path: Path,
    pm: PhotoManager,
    engine: StyleTransferEngine,
    w: int,
    h: int,
) -> None:
    """Apply must return the correct size for several input resolutions."""
    src = tmp_path / "photo.jpg"
    _write_jpeg(src, w, h)
    img = pm.load(src)
    result = engine.apply(img, "candy", strength=1.0, tile_size=128, overlap=16)
    assert result.size == (w, h)
