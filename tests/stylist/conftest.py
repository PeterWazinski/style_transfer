"""Shared pytest-qt fixtures for all UI tests."""
from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image
from PySide6.QtGui import QPixmap

from src.core.engine import StyleTransferEngine
from src.core.models import StyleModel
from src.core.photo_manager import PhotoManager
from src.core.registry import StyleRegistry
from src.stylist.main_window import MainWindow
from src.stylist.photo_canvas import PhotoCanvasView
from src.stylist.style_gallery import StyleGalleryView


# ---------------------------------------------------------------------------
# Registry / catalog
# ---------------------------------------------------------------------------

@pytest.fixture()
def catalog_path(tmp_path: Path) -> Path:
    """Empty catalog file in a temp directory."""
    return tmp_path / "catalog.json"


@pytest.fixture()
def empty_registry(catalog_path: Path) -> StyleRegistry:
    return StyleRegistry(catalog_path=catalog_path)


@pytest.fixture()
def user_style(tmp_path: Path) -> StyleModel:
    """A non-builtin StyleModel usable in tests (no actual files required)."""
    return StyleModel(
        id="test-paint",
        name="Test Paint",
        model_path=str(tmp_path / "model.onnx"),
        preview_path=str(tmp_path / "preview.jpg"),
        description="A test style",
        author="Tester",
        is_builtin=False,
    )


@pytest.fixture()
def registry(empty_registry: StyleRegistry, user_style: StyleModel) -> StyleRegistry:
    """Registry pre-populated with one user style."""
    empty_registry.add(user_style)
    return empty_registry


# ---------------------------------------------------------------------------
# UI component fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def gallery(qtbot, registry: StyleRegistry) -> StyleGalleryView:
    w = StyleGalleryView(registry=registry)
    qtbot.addWidget(w)
    return w


@pytest.fixture()
def canvas(qtbot) -> PhotoCanvasView:
    w = PhotoCanvasView()
    qtbot.addWidget(w)
    return w


@pytest.fixture()
def main_window(qtbot, registry: StyleRegistry) -> MainWindow:
    window = MainWindow(
        registry=registry,
        engine=StyleTransferEngine(),
        photo_manager=PhotoManager(),
    )
    qtbot.addWidget(window)
    return window
