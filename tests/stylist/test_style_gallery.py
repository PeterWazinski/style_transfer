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

# ---------------------------------------------------------------------------
# Tooltip
# ---------------------------------------------------------------------------

class TestTooltip:
    def test_tooltip_set_when_description_present(
        self, qtbot, registry: StyleRegistry, tmp_path
    ) -> None:
        """Items whose style has a non-empty description get a tooltip."""
        # The fixture's user_style already has description="A test style"
        gallery = StyleGalleryView(registry=registry)
        qtbot.addWidget(gallery)
        item = gallery.model().item(0)
        assert item is not None
        assert item.toolTip() == "A test style"

    def test_no_tooltip_when_description_empty(
        self, qtbot, empty_registry: StyleRegistry, tmp_path
    ) -> None:
        """Items with empty description should have no tooltip."""
        from src.core.models import StyleModel
        style = StyleModel(
            id="no-desc", name="No Desc",
            model_path=str(tmp_path / "model.onnx"),
            description="",
        )
        empty_registry.add(style)
        gallery = StyleGalleryView(registry=empty_registry)
        qtbot.addWidget(gallery)
        item = gallery.model().item(0)
        assert item is not None
        assert item.toolTip() == ""


# ---------------------------------------------------------------------------
# Right-click context menu signals
# ---------------------------------------------------------------------------

class TestContextMenu:
    def test_context_menu_apply_emits_style_apply_requested(
        self, qtbot, gallery: StyleGalleryView
    ) -> None:
        received: list = []
        gallery.style_apply_requested.connect(received.append)
        first_index = gallery.model().index(0, 0)
        style = first_index.data(Qt.UserRole)

        # Simulate the context-menu handler directly (avoids Qt mouse event machinery)
        gallery._on_context_menu_requested.__func__  # ensure it exists  # noqa: B018
        # Build a QPoint that maps to the first item
        rect = gallery._list_view.visualRect(first_index)
        pos = rect.center()
        # Patch menu.exec to return apply_action without showing UI
        import unittest.mock as mock
        with mock.patch("src.stylist.style_gallery.QMenu") as MockMenu:
            instance = MockMenu.return_value
            apply_action = object()
            reapply_action = object()
            instance.addAction.side_effect = [apply_action, reapply_action]
            instance.exec.return_value = apply_action
            gallery._on_context_menu_requested(pos)

        assert len(received) == 1
        assert received[0].id == style.id

    def test_context_menu_reapply_emits_style_reapply_requested(
        self, qtbot, gallery: StyleGalleryView
    ) -> None:
        received: list = []
        gallery.style_reapply_requested.connect(received.append)
        first_index = gallery.model().index(0, 0)
        style = first_index.data(Qt.UserRole)

        rect = gallery._list_view.visualRect(first_index)
        pos = rect.center()
        import unittest.mock as mock
        with mock.patch("src.stylist.style_gallery.QMenu") as MockMenu:
            instance = MockMenu.return_value
            apply_action = object()
            reapply_action = object()
            instance.addAction.side_effect = [apply_action, reapply_action]
            instance.exec.return_value = reapply_action
            gallery._on_context_menu_requested(pos)

        assert len(received) == 1
        assert received[0].id == style.id