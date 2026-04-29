"""Unit tests for the nchw_tanh tensor layout (CycleGAN-style models)."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from PIL import Image

from src.core.engine import StyleTransferEngine
from tests.helpers import make_mock_session_nchw_tanh


def _engine_with_nchw_tanh_style(
    style_id: str = "monet",
    output_colour: tuple[int, int, int] = (100, 150, 200),
    model_path: Path = Path("dummy/monet.onnx"),
) -> StyleTransferEngine:
    """Return an engine with a mocked CycleGAN-style NCHW tanh session loaded."""
    engine = StyleTransferEngine()
    with (
        patch("src.core.engine._ORT_AVAILABLE", True),
        patch("src.core.engine.ort") as mock_ort,
        patch.object(Path, "exists", return_value=True),
    ):
        mock_ort.InferenceSession.return_value = make_mock_session_nchw_tanh(output_colour)
        engine.load_model(style_id, model_path, tensor_layout="nchw_tanh")
    return engine


# ---------------------------------------------------------------------------
# Layout stored in _model_meta
# ---------------------------------------------------------------------------

def test_nchw_tanh_tensor_layout_stored() -> None:
    engine = StyleTransferEngine()
    with (
        patch("src.core.engine._ORT_AVAILABLE", True),
        patch("src.core.engine.ort") as mock_ort,
        patch.object(Path, "exists", return_value=True),
    ):
        mock_ort.InferenceSession.return_value = make_mock_session_nchw_tanh()
        engine.load_model("monet", Path("dummy.onnx"), tensor_layout="nchw_tanh")
    assert engine._model_meta["monet"] == "nchw_tanh"


# ---------------------------------------------------------------------------
# apply — output size and type
# ---------------------------------------------------------------------------

def test_nchw_tanh_apply_output_size_matches_input() -> None:
    engine = _engine_with_nchw_tanh_style()
    img = Image.new("RGB", (200, 160))
    result = engine.apply(img, "monet", strength=1.0, tile_size=256, overlap=32)
    assert result.size == (200, 160)


def test_nchw_tanh_apply_output_is_pil_image() -> None:
    engine = _engine_with_nchw_tanh_style()
    img = Image.new("RGB", (64, 64))
    result = engine.apply(img, "monet", strength=1.0, tile_size=256, overlap=32)
    assert isinstance(result, Image.Image)


# ---------------------------------------------------------------------------
# apply — colour correctness (de-normalisation)
# ---------------------------------------------------------------------------

def test_nchw_tanh_apply_output_colour_correct() -> None:
    """De-normalisation must map [-1,1] tanh output to correct [0,255] colour."""
    engine = _engine_with_nchw_tanh_style(output_colour=(100, 150, 200))
    img = Image.new("RGB", (64, 64))
    result = engine.apply(img, "monet", strength=1.0, tile_size=256, overlap=32)
    arr = np.array(result)
    assert abs(int(arr[:, :, 0].mean()) - 100) <= 5
    assert abs(int(arr[:, :, 1].mean()) - 150) <= 5
    assert abs(int(arr[:, :, 2].mean()) - 200) <= 5


def test_nchw_tanh_apply_pixel_range_is_0_255() -> None:
    """Output pixels must be in [0, 255] after de-normalisation."""
    engine = _engine_with_nchw_tanh_style(output_colour=(100, 150, 200))
    img = Image.new("RGB", (64, 64))
    result = engine.apply(img, "monet", strength=1.0, tile_size=256, overlap=32)
    arr = np.array(result)
    assert arr.min() >= 0
    assert arr.max() <= 255


# ---------------------------------------------------------------------------
# apply — strength blending
# ---------------------------------------------------------------------------

def test_nchw_tanh_apply_strength_zero_returns_original() -> None:
    engine = _engine_with_nchw_tanh_style(output_colour=(0, 255, 0))
    img = Image.new("RGB", (64, 64), color=(255, 0, 0))
    result = engine.apply(img, "monet", strength=0.0, tile_size=256, overlap=32)
    arr = np.array(result)
    assert arr[:, :, 0].mean() > 200


# ---------------------------------------------------------------------------
# dispatch: _infer_tile routes to _infer_tile_nchw_tanh
# ---------------------------------------------------------------------------

def test_nchw_tanh_dispatch_calls_correct_method() -> None:
    """_infer_tile must call _infer_tile_nchw_tanh for nchw_tanh layout."""
    engine = _engine_with_nchw_tanh_style()
    img = Image.new("RGB", (64, 64))
    with patch.object(
        engine, "_infer_tile_nchw_tanh", wraps=engine._infer_tile_nchw_tanh
    ) as spy:
        engine.apply(img, "monet", strength=1.0, tile_size=256, overlap=32)
    assert spy.call_count >= 1
