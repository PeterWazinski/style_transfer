"""Unit tests for StyleModel and StyleStore (src/core/models.py)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.core.models import StyleModel, StyleStore


# ---------------------------------------------------------------------------
# StyleModel — defaults
# ---------------------------------------------------------------------------

def test_style_model_default_tensor_layout() -> None:
    m = StyleModel(id="candy", name="Candy", model_path="p.onnx", preview_path="p.jpg")
    assert m.tensor_layout == "nchw"


def test_style_model_default_is_builtin() -> None:
    m = StyleModel(id="x", name="X", model_path="p", preview_path="q")
    assert m.is_builtin is True


# ---------------------------------------------------------------------------
# StyleModel — from_dict round-trips
# ---------------------------------------------------------------------------

def test_from_dict_tensor_layout_nhwc_tanh() -> None:
    m = StyleModel.from_dict({
        "id": "anime", "name": "Anime",
        "model_path": "styles/anime/model.onnx",
        "preview_path": "styles/anime/preview.jpg",
        "tensor_layout": "nhwc_tanh",
    })
    assert m.tensor_layout == "nhwc_tanh"


def test_from_dict_tensor_layout_defaults_to_nchw_when_absent() -> None:
    m = StyleModel.from_dict({
        "id": "candy", "name": "Candy",
        "model_path": "p", "preview_path": "q",
    })
    assert m.tensor_layout == "nchw"


def test_from_dict_unknown_keys_ignored() -> None:
    """Extra catalog keys from future versions must not raise."""
    m = StyleModel.from_dict({
        "id": "x", "name": "X", "model_path": "p", "preview_path": "q",
        "future_unknown_key": "some_value",
    })
    assert m.id == "x"


def test_to_dict_round_trips_tensor_layout() -> None:
    m = StyleModel(
        id="anime", name="Anime",
        model_path="styles/anime/model.onnx",
        preview_path="styles/anime/preview.jpg",
        tensor_layout="nhwc_tanh",
    )
    d = m.to_dict()
    assert d["tensor_layout"] == "nhwc_tanh"
    restored = StyleModel.from_dict(d)
    assert restored.tensor_layout == "nhwc_tanh"


# ---------------------------------------------------------------------------
# StyleStore — JSON persistence
# ---------------------------------------------------------------------------

def test_store_load_save_preserves_tensor_layout(tmp_path: Path) -> None:
    catalog_path = tmp_path / "catalog.json"
    catalog_path.write_text(
        json.dumps({
            "styles": [
                {
                    "id": "anime",
                    "name": "Anime",
                    "model_path": "styles/anime/model.onnx",
                    "preview_path": "styles/anime/preview.jpg",
                    "tensor_layout": "nhwc_tanh",
                }
            ]
        }),
        encoding="utf-8",
    )
    store = StyleStore(catalog_path)
    styles = store.load()
    assert styles[0].tensor_layout == "nhwc_tanh"
