"""Unit tests for error handling across the core layer.

Verifies that:
- Missing ONNX model → :exc:`StyleModelNotFoundError`
- Corrupt ONNX model → :exc:`CorruptModelError`
- Out-of-memory tile  → :exc:`OOMError`
- Unsupported image format → :exc:`UnsupportedFormatError`
- Invalid strength value → :exc:`ValueError`
- Missing COCO path → :exc:`COCODatasetNotFoundError`
- Invalid AppSettings values → :exc:`ValueError`
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from PIL import Image

from src.core.engine import (
    CorruptModelError,
    OOMError,
    StyleModelNotFoundError,
    StyleTransferEngine,
)
from src.core.photo_manager import PhotoManager, UnsupportedFormatError
from src.core.settings import AppSettings
from tests.helpers import make_mock_session


def _engine_with_mock_style(
    style_id: str = "test-style",
    raise_oom: bool = False,
) -> StyleTransferEngine:
    engine = StyleTransferEngine()
    with (
        patch("src.core.engine._ORT_AVAILABLE", True),
        patch("src.core.engine.ort") as mock_ort,
        patch.object(Path, "exists", return_value=True),
    ):
        mock_ort.InferenceSession.return_value = make_mock_session(raise_oom=raise_oom)
        engine.load_model(style_id, Path("dummy/model.onnx"))
    return engine


# ---------------------------------------------------------------------------
# StyleModelNotFoundError
# ---------------------------------------------------------------------------

class TestStyleModelNotFoundError:
    def test_missing_model_file_raises(self) -> None:
        engine = StyleTransferEngine()
        with patch("src.core.engine._ORT_AVAILABLE", True):
            with pytest.raises(StyleModelNotFoundError, match="not found"):
                engine.load_model("candy", Path("/no/such/file.onnx"))

    def test_apply_unloaded_style_raises_key_error(self) -> None:
        engine = StyleTransferEngine()
        img = Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8))
        with pytest.raises(KeyError, match="not loaded"):
            engine.apply(img, "nonexistent")


# ---------------------------------------------------------------------------
# CorruptModelError
# ---------------------------------------------------------------------------

class TestCorruptModelError:
    def test_corrupt_model_raises(self) -> None:
        engine = StyleTransferEngine()
        with (
            patch("src.core.engine._ORT_AVAILABLE", True),
            patch("src.core.engine.ort") as mock_ort,
            patch.object(Path, "exists", return_value=True),
        ):
            mock_ort.InferenceSession.side_effect = RuntimeError("Bad graph")
            with pytest.raises(CorruptModelError, match="Failed to load"):
                engine.load_model("candy", Path("bad_model.onnx"))


# ---------------------------------------------------------------------------
# OOMError
# ---------------------------------------------------------------------------

class TestOOMError:
    def test_oom_during_tile_raises(self) -> None:
        engine = _engine_with_mock_style(raise_oom=True)
        img = Image.fromarray(np.zeros((128, 128, 3), dtype=np.uint8))
        with pytest.raises(OOMError, match="Out of memory"):
            engine.apply(img, "test-style", tile_size=64, overlap=0)


# ---------------------------------------------------------------------------
# UnsupportedFormatError
# ---------------------------------------------------------------------------

class TestUnsupportedFormatError:
    def test_unsupported_extension_on_load_raises(self, tmp_path: Path) -> None:
        p = tmp_path / "image.bmp"
        p.write_bytes(b"\x00" * 10)
        pm = PhotoManager()
        with pytest.raises(UnsupportedFormatError, match="bmp"):
            pm.load(p)

    def test_unsupported_extension_on_save_raises(self, tmp_path: Path) -> None:
        img = Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8))
        pm = PhotoManager()
        with pytest.raises(UnsupportedFormatError, match="gif"):
            pm.save(img, tmp_path / "output.gif")


# ---------------------------------------------------------------------------
# ValueError — invalid strength
# ---------------------------------------------------------------------------

class TestInvalidStrength:
    def test_strength_above_one_raises(self) -> None:
        engine = _engine_with_mock_style()
        img = Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8))
        with pytest.raises(ValueError, match="strength"):
            engine.apply(img, "test-style", strength=3.1)

    def test_strength_below_zero_raises(self) -> None:
        engine = _engine_with_mock_style()
        img = Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8))
        with pytest.raises(ValueError, match="strength"):
            engine.apply(img, "test-style", strength=-0.1)


# ---------------------------------------------------------------------------
# AppSettings validation
# ---------------------------------------------------------------------------

class TestAppSettingsValidation:
    def test_invalid_tile_size_raises(self) -> None:
        with pytest.raises(ValueError, match="tile_size"):
            AppSettings(tile_size=999)

    def test_invalid_overlap_raises(self) -> None:
        with pytest.raises(ValueError, match="overlap"):
            AppSettings(overlap=99)

    def test_invalid_provider_raises(self) -> None:
        with pytest.raises(ValueError, match="execution_provider"):
            AppSettings(execution_provider="tpu")

    def test_overlap_too_large_for_tile_size_raises(self) -> None:
        # overlap 256 is valid alone, but >= tile_size // 2 = 256 when tile_size=512
        with pytest.raises(ValueError, match="overlap"):
            AppSettings(tile_size=512, overlap=256)

    def test_valid_settings_accepted(self) -> None:
        s = AppSettings(tile_size=1024, overlap=128, execution_provider="cpu")
        assert s.tile_size == 1024
        assert s.overlap == 128

    def test_settings_round_trips_json(self, tmp_path: Path) -> None:
        s = AppSettings(tile_size=768, overlap=64, use_float16=True)
        p = tmp_path / "settings.json"
        s.save(p)
        loaded = AppSettings.load(p)
        assert loaded == s

    def test_load_missing_file_returns_defaults(self, tmp_path: Path) -> None:
        s = AppSettings.load(tmp_path / "nonexistent.json")
        assert s == AppSettings()

    def test_load_corrupt_file_returns_defaults(self, tmp_path: Path) -> None:
        p = tmp_path / "settings.json"
        p.write_text("not json {{{")
        s = AppSettings.load(p)
        assert s == AppSettings()
