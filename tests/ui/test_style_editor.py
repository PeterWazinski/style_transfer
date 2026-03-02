"""Tests for StyleEditorDialog — add / edit a custom style."""
from __future__ import annotations

from pathlib import Path

import pytest
from PySide6.QtWidgets import QDialog, QFileDialog

from src.core.models import StyleModel
from src.ui.style_editor import StyleEditorDialog


# ---------------------------------------------------------------------------
# Name validation
# ---------------------------------------------------------------------------

class TestNameValidation:
    def test_empty_name_blocks_acceptance(
        self, qtbot, editor_dialog: StyleEditorDialog
    ) -> None:
        """Clicking OK with an empty name must NOT close the dialog."""
        editor_dialog.name_edit.clear()
        # Trigger the accept logic directly (avoids exec() blocking)
        editor_dialog._on_accept()
        # isHidden() is reliable even when parent dialog is not shown.
        assert not editor_dialog._name_error.isHidden()

    def test_non_empty_name_hides_error_label(
        self, qtbot, editor_dialog: StyleEditorDialog
    ) -> None:
        editor_dialog.name_edit.setText("Van Gogh")
        editor_dialog._on_accept()
        assert not editor_dialog._name_error.isVisible()

    def test_whitespace_only_name_is_blocked(
        self, qtbot, editor_dialog: StyleEditorDialog
    ) -> None:
        editor_dialog.name_edit.setText("   ")
        editor_dialog._on_accept()
        assert not editor_dialog._name_error.isHidden()


# ---------------------------------------------------------------------------
# Browse button (file dialog)
# ---------------------------------------------------------------------------

class TestBrowseButton:
    def test_browse_button_calls_file_dialog(
        self, qtbot, editor_dialog: StyleEditorDialog, mocker
    ) -> None:
        """Clicking Browse must invoke QFileDialog.getOpenFileNames."""
        mocker.patch.object(
            QFileDialog,
            "getOpenFileNames",
            return_value=([], ""),
        )
        editor_dialog.browse_button.click()
        QFileDialog.getOpenFileNames.assert_called_once()  # type: ignore[attr-defined]

    def test_selected_files_appear_in_ref_list(
        self, qtbot, editor_dialog: StyleEditorDialog, tmp_path: Path, mocker
    ) -> None:
        fake_files = [str(tmp_path / "style1.jpg"), str(tmp_path / "style2.jpg")]
        mocker.patch.object(
            QFileDialog,
            "getOpenFileNames",
            return_value=(fake_files, ""),
        )
        editor_dialog.browse_button.click()
        assert editor_dialog.ref_list.count() == 2

    def test_remove_button_deletes_selected_reference(
        self, qtbot, editor_dialog: StyleEditorDialog, tmp_path: Path, mocker
    ) -> None:
        mocker.patch.object(
            QFileDialog,
            "getOpenFileNames",
            return_value=([str(tmp_path / "a.jpg")], ""),
        )
        editor_dialog.browse_button.click()
        assert editor_dialog.ref_list.count() == 1
        editor_dialog.ref_list.setCurrentRow(0)
        editor_dialog.remove_ref_button.click()
        assert editor_dialog.ref_list.count() == 0


# ---------------------------------------------------------------------------
# style_saved signal
# ---------------------------------------------------------------------------

class TestStyleSavedSignal:
    def test_save_emits_style_saved_with_correct_name(
        self, qtbot, editor_dialog: StyleEditorDialog
    ) -> None:
        editor_dialog.name_edit.setText("Starry Night")
        received: list[StyleModel] = []
        editor_dialog.style_saved.connect(received.append)

        editor_dialog._on_accept()

        assert len(received) == 1
        assert received[0].name == "Starry Night"
        assert received[0].is_builtin is False

    def test_save_emits_style_saved_with_description(
        self, qtbot, editor_dialog: StyleEditorDialog
    ) -> None:
        editor_dialog.name_edit.setText("Monet")
        editor_dialog.description_edit.setPlainText("Impressionist water lilies")
        received: list[StyleModel] = []
        editor_dialog.style_saved.connect(received.append)
        editor_dialog._on_accept()
        assert received[0].description == "Impressionist water lilies"

    def test_dialog_accepts_after_valid_name(
        self, qtbot, editor_dialog: StyleEditorDialog
    ) -> None:
        editor_dialog.name_edit.setText("Candy")
        # Use waitSignal to verify style_saved without blocking
        with qtbot.waitSignal(editor_dialog.style_saved, timeout=300):
            editor_dialog._on_accept()


# ---------------------------------------------------------------------------
# Edit mode
# ---------------------------------------------------------------------------

class TestEditMode:
    def test_populated_with_existing_style(
        self, qtbot, tmp_path: Path
    ) -> None:
        existing = StyleModel(
            id="candy",
            name="Candy",
            model_path=str(tmp_path / "model.onnx"),
            preview_path=str(tmp_path / "preview.jpg"),
            description="Bright cartoon",
            author="Yakhyo",
        )
        dlg = StyleEditorDialog(style=existing)
        qtbot.addWidget(dlg)

        assert dlg.name_edit.text() == "Candy"
        assert dlg.description_edit.toPlainText() == "Bright cartoon"
        assert dlg.author_edit.text() == "Yakhyo"

    def test_editing_preserves_style_id(
        self, qtbot, tmp_path: Path
    ) -> None:
        existing = StyleModel(
            id="candy",
            name="Candy",
            model_path=str(tmp_path / "model.onnx"),
            preview_path=str(tmp_path / "preview.jpg"),
        )
        dlg = StyleEditorDialog(style=existing)
        qtbot.addWidget(dlg)
        dlg.name_edit.setText("Candy Updated")
        received: list[StyleModel] = []
        dlg.style_saved.connect(received.append)
        dlg._on_accept()
        assert received[0].id == "candy"   # ID must not change on edit
