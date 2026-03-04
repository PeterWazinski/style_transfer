"""Tests for StyleGalleryView — read-only thumbnail browser."""
from __future__ import annotations

from pathlib import Path

import pytest
from PySide6.QtCore import Qt

from src.core.models import StyleModel
from src.stylist.style_gallery import StyleGalleryView


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _item_count(gallery: StyleGalleryView) -> int:
    return gallery.model().rowCount()


# ---------------------------------------------------------------------------
# Thumbnail click
# ---------------------------------------------------------------------------

class TestThumbnailClick:
    def test_click_thumbnail_emits_style_selected(
        self, qtbot, gallery: StyleGalleryView
    ) -> None:
        received: list[StyleModel] = []
        gallery.style_selected.connect(received.append)

        first_index = gallery.model().index(0, 0)
        gallery._list_view.clicked.emit(first_index)

        assert len(received) == 1
        assert received[0].id == "test-paint"

    def test_click_style_updates_canvas_active_style(
        self, qtbot, main_window
    ) -> None:
        """Clicking a gallery item sets the active style on the canvas."""
        first_index = main_window.gallery.model().index(0, 0)
        main_window.gallery._list_view.clicked.emit(first_index)
        assert main_window.canvas._current_style_id == "test-paint"


# ---------------------------------------------------------------------------
# Refresh
# ---------------------------------------------------------------------------

class TestRefresh:
    def test_refresh_repopulates_model(
        self, qtbot, gallery: StyleGalleryView, registry, tmp_path: Path
    ) -> None:
        """Adding a style to the registry and calling refresh() shows the new entry."""
        extra = StyleModel(
            id="extra-style",
            name="Extra",
            model_path=str(tmp_path / "m.onnx"),
            preview_path=str(tmp_path / "p.jpg"),
        )
        registry.add(extra)
        before = _item_count(gallery)
        gallery.refresh()
        assert _item_count(gallery) == before + 1
