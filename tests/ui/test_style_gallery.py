"""Tests for StyleGalleryView — thumbnail browser and CRUD controls."""
from __future__ import annotations

from pathlib import Path

import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QDialog, QMenu, QMessageBox

from src.core.models import StyleModel
from src.ui.style_editor import StyleEditorDialog
from src.ui.style_gallery import StyleGalleryView


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _item_count(gallery: StyleGalleryView) -> int:
    return gallery.model().rowCount()


# ---------------------------------------------------------------------------
# Add button
# ---------------------------------------------------------------------------

class TestAddButton:
    def test_add_button_emits_add_requested(self, qtbot, gallery: StyleGalleryView) -> None:
        with qtbot.waitSignal(gallery.add_requested, timeout=300):
            gallery.add_button.click()

    def test_add_button_opens_editor_dialog(
        self, qtbot, main_window, mocker
    ) -> None:
        """MainWindow wires add_requested → StyleEditorDialog; the class must be
        instantiated with parent=main_window and exec() called."""
        mock_dlg = mocker.MagicMock()
        mock_dlg.style_saved = mocker.MagicMock()
        mock_cls = mocker.patch(
            "src.ui.main_window.StyleEditorDialog",
            return_value=mock_dlg,
        )
        main_window.gallery.add_button.click()
        mock_cls.assert_called_once_with(parent=main_window)
        mock_dlg.exec.assert_called_once()


# ---------------------------------------------------------------------------
# Delete button
# ---------------------------------------------------------------------------

class TestDeleteButton:
    def test_delete_button_emits_delete_requested_for_selected(
        self, qtbot, gallery: StyleGalleryView
    ) -> None:
        # Select the first (and only) item
        first_index = gallery.model().index(0, 0)
        gallery._list_view.setCurrentIndex(first_index)

        received: list[str] = []
        gallery.delete_requested.connect(received.append)

        gallery.delete_button.click()

        assert received == ["test-paint"]

    def test_delete_button_does_nothing_when_nothing_selected(
        self, qtbot, gallery: StyleGalleryView
    ) -> None:
        received: list[str] = []
        gallery.delete_requested.connect(received.append)
        gallery._list_view.clearSelection()
        gallery.delete_button.click()
        assert received == []

    def test_delete_removes_item_from_model(
        self, qtbot, main_window, user_style, mocker
    ) -> None:
        """Calling registry.delete + gallery.refresh reduces the row count by one."""
        mocker.patch.object(
            QMessageBox, "question",
            return_value=QMessageBox.StandardButton.Yes,
        )
        n_before = _item_count(main_window.gallery)
        main_window._delete_style(user_style.id)
        assert _item_count(main_window.gallery) == n_before - 1


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
# Context menu
# ---------------------------------------------------------------------------

class TestContextMenu:
    def test_context_menu_contains_edit_and_delete(
        self, qtbot, gallery: StyleGalleryView, mocker
    ) -> None:
        """Right-click context menu must contain both 'Edit…' and 'Delete' actions."""
        from PySide6.QtCore import QPoint
        from src.ui import style_gallery as _sg

        captured: list[QMenu] = []

        # Subclass QMenu so Python's exec() override is reliably called.
        class _CapturingMenu(QMenu):
            def exec(self, *args, **kwargs) -> None:  # type: ignore[override]
                captured.append(self)
                return None  # dismiss without selection

        mocker.patch.object(_sg, "QMenu", _CapturingMenu)

        # Stub indexAt so the handler finds the first item regardless of layout.
        first_index = gallery.model().index(0, 0)
        mocker.patch.object(gallery._list_view, "indexAt", return_value=first_index)

        gallery._on_context_menu(QPoint(5, 5))

        assert len(captured) == 1
        action_texts = [a.text() for a in captured[0].actions() if not a.isSeparator()]
        assert "Edit…" in action_texts
        assert "Delete" in action_texts


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
