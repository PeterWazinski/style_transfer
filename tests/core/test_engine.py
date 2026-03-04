"""Unit tests for StyleTransferEngine (ONNX inference mocked)."""
from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

from src.core.engine import StyleModelNotFoundError, StyleTransferEngine


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_session(output_colour: tuple[int, int, int] = (128, 64, 192)) -> MagicMock:
    """Return a mock ort.InferenceSession that echoes a solid-colour image."""
    session = MagicMock()
    inp = MagicMock()
    inp.name = "input"
    session.get_inputs.return_value = [inp]

    def _run(output_names: list[str], feed: dict[str, np.ndarray]) -> list[np.ndarray]:
        tensor = feed["input"]  # [1, 3, H, W]
        h, w = tensor.shape[2], tensor.shape[3]
        rgb = np.full((1, 3, h, w), 0.0, dtype=np.float32)
        rgb[0, 0, :, :] = float(output_colour[0])
        rgb[0, 1, :, :] = float(output_colour[1])
        rgb[0, 2, :, :] = float(output_colour[2])
        return [rgb]

    session.run.side_effect = _run
    return session


def _engine_with_style(
    style_id: str = "candy",
    output_colour: tuple[int, int, int] = (128, 64, 192),
    model_path: Path = Path("dummy/candy.onnx"),
) -> StyleTransferEngine:
    """Return a StyleTransferEngine with a mocked ONNX session already loaded."""
    engine = StyleTransferEngine()
    with (
        patch("src.core.engine._ORT_AVAILABLE", True),
        patch("src.core.engine.ort") as mock_ort,
        patch.object(Path, "exists", return_value=True),
    ):
        mock_ort.InferenceSession.return_value = _make_mock_session(output_colour)
        engine.load_model(style_id, model_path)
    return engine


# ---------------------------------------------------------------------------
# load_model
# ---------------------------------------------------------------------------

def test_load_model_missing_file_raises() -> None:
    engine = StyleTransferEngine()
    with (
        patch("src.core.engine._ORT_AVAILABLE", True),
        patch("src.core.engine.ort"),
    ):
        with pytest.raises(StyleModelNotFoundError):
            engine.load_model("candy", Path("/nonexistent/model.onnx"))


def test_load_model_ort_not_installed_raises() -> None:
    engine = StyleTransferEngine()
    with patch("src.core.engine._ORT_AVAILABLE", False):
        with pytest.raises(ImportError):
            engine.load_model("candy", Path("any.onnx"))


def test_is_loaded_false_before_load() -> None:
    engine = StyleTransferEngine()
    assert not engine.is_loaded("candy")


def test_is_loaded_true_after_load() -> None:
    engine = _engine_with_style("candy")
    assert engine.is_loaded("candy")


# ---------------------------------------------------------------------------
# apply — validation
# ---------------------------------------------------------------------------

def test_apply_unknown_style_raises() -> None:
    engine = StyleTransferEngine()
    img = Image.new("RGB", (64, 64))
    with pytest.raises(KeyError):
        engine.apply(img, "unknown_style")


def test_apply_invalid_strength_raises() -> None:
    engine = _engine_with_style()
    img = Image.new("RGB", (64, 64))
    with pytest.raises(ValueError, match="strength"):
        engine.apply(img, "candy", strength=1.5)
    with pytest.raises(ValueError, match="strength"):
        engine.apply(img, "candy", strength=-0.1)


# ---------------------------------------------------------------------------
# apply — output properties
# ---------------------------------------------------------------------------

def test_apply_output_size_matches_input() -> None:
    engine = _engine_with_style()
    img = Image.new("RGB", (200, 150))
    result = engine.apply(img, "candy", strength=1.0, tile_size=128, overlap=16)
    assert result.size == (200, 150)


def test_apply_strength_zero_returns_original_pixels() -> None:
    engine = _engine_with_style()
    img = Image.new("RGB", (64, 64), color=(255, 0, 0))
    result = engine.apply(img, "candy", strength=0.0, tile_size=128, overlap=16)
    arr = np.array(result)
    # Should be essentially the original red image
    assert arr[:, :, 0].mean() > 200


def test_apply_strength_one_returns_styled_pixels() -> None:
    engine = _engine_with_style(output_colour=(0, 255, 0))
    img = Image.new("RGB", (64, 64), color=(255, 0, 0))
    result = engine.apply(img, "candy", strength=1.0, tile_size=128, overlap=16)
    arr = np.array(result)
    # Styled is solid green — green channel should dominate
    assert arr[:, :, 1].mean() > 200


def test_apply_output_is_pil_image() -> None:
    engine = _engine_with_style()
    img = Image.new("RGB", (64, 64))
    result = engine.apply(img, "candy", strength=0.5, tile_size=128, overlap=16)
    assert isinstance(result, Image.Image)


def test_apply_progress_callback_called() -> None:
    engine = _engine_with_style()
    img = Image.new("RGB", (64, 64))
    calls: list[tuple[int, int]] = []
    engine.apply(
        img,
        "candy",
        strength=1.0,
        tile_size=128,
        overlap=16,
        progress_callback=lambda done, total: calls.append((done, total)),
    )
    assert len(calls) >= 1
    # Last call: done == total
    assert calls[-1][0] == calls[-1][1]


# ---------------------------------------------------------------------------
# preview
# ---------------------------------------------------------------------------

def test_preview_output_max_dim_respected() -> None:
    engine = _engine_with_style()
    img = Image.new("RGB", (1024, 512))
    result = engine.preview(img, "candy", strength=1.0, max_dim=256)
    w, h = result.size
    assert max(w, h) <= 256


def test_preview_small_image_not_upscaled() -> None:
    engine = _engine_with_style()
    img = Image.new("RGB", (64, 64))
    result = engine.preview(img, "candy", strength=1.0, max_dim=512)
    # Image is smaller than max_dim — should not be upscaled
    assert result.size[0] <= 64
    assert result.size[1] <= 64
