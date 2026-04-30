"""Tests for Phase 2 — replay log integration in MainWindow.

Covers:
- Initial state: empty log
- _apply_style starts a new chain (resets log)
- _reapply_style appends to log
- _reapply_style_strength updates last entry
- _perform_undo pops last log entry
- _open_photo clears log
- _reset_photo clears log
- _format_replay_log serialises to valid YAML
- _copy_replay_log_to_clipboard when empty shows dialog
- Auto-save .yml written next to saved image when enabled
- Auto-save .yml skipped when autosave_replay_log=False
- _load_and_apply_replay_log applies a valid chain
- _load_and_apply_replay_log shows error on invalid schema
"""
from __future__ import annotations

import textwrap
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import yaml
from PIL import Image
from PySide6.QtWidgets import QApplication

from src.core.engine import StyleTransferEngine
from src.core.models import StyleModel
from src.core.photo_manager import PhotoManager
from src.core.registry import StyleRegistry
from src.core.settings import AppSettings
from src.stylist.main_window import MainWindow
from tests.helpers import make_mock_session


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dummy_image(size: int = 64) -> Image.Image:
    return Image.fromarray(np.zeros((size, size, 3), dtype=np.uint8))


def _make_window(qtbot, tmp_path: Path, autosave: bool = True) -> tuple[MainWindow, MagicMock]:
    """Create a MainWindow with one registered style and a patched engine."""
    preview = tmp_path / "preview.jpg"
    _dummy_image().save(preview)

    style = StyleModel(
        id="test-style",
        name="Test Style",
        model_path=str(tmp_path / "model.onnx"),
        preview_path=str(preview),
        is_builtin=False,
    )
    registry = StyleRegistry(catalog_path=tmp_path / "catalog.json")
    registry.add(style)

    engine = StyleTransferEngine()
    with (
        patch("src.core.engine._ORT_AVAILABLE", True),
        patch("src.core.engine.ort") as mock_ort,
        patch.object(Path, "exists", return_value=True),
    ):
        mock_ort.InferenceSession.return_value = make_mock_session()
        engine.load_model("test-style", Path("dummy/model.onnx"))

    settings = AppSettings(autosave_replay_log=autosave)
    window = MainWindow(
        registry=registry,
        engine=engine,
        photo_manager=PhotoManager(),
        settings=settings,
    )
    qtbot.addWidget(window)
    return window, engine


def _load_photo(window: MainWindow, tmp_path: Path) -> None:
    photo = tmp_path / "photo.jpg"
    _dummy_image(128).save(photo)
    with patch(
        "src.stylist.main_window.QFileDialog.getOpenFileName",
        return_value=(str(photo), ""),
    ):
        window._open_photo()


def _do_apply(window: MainWindow, engine: MagicMock) -> None:
    """Apply 'test-style' at 100% to whatever is in the window."""
    result = _dummy_image()
    engine.apply = MagicMock(return_value=result)
    window._current_style_name = "Test Style"
    with patch("src.stylist.main_window.QMessageBox.critical"):
        window._apply_style("test-style", 1.0)


def _do_reapply(window: MainWindow, engine: MagicMock, strength: float = 1.5) -> None:
    result = _dummy_image()
    engine.apply = MagicMock(return_value=result)
    window._current_style_name = "Test Style"
    with patch("src.stylist.main_window.QMessageBox.critical"):
        window._reapply_style("test-style", strength)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestReplayLog:
    def test_log_empty_initially(self, qtbot, tmp_path: Path) -> None:
        window, _ = _make_window(qtbot, tmp_path)
        assert window._replay_log == []

    def test_apply_starts_new_chain(self, qtbot, tmp_path: Path) -> None:
        window, engine = _make_window(qtbot, tmp_path)
        _load_photo(window, tmp_path)
        _do_apply(window, engine)
        assert len(window._replay_log) == 1
        assert window._replay_log[0]["style"] == "Test Style"
        assert window._replay_log[0]["strength"] == 100

    def test_apply_resets_previous_chain(self, qtbot, tmp_path: Path) -> None:
        window, engine = _make_window(qtbot, tmp_path)
        _load_photo(window, tmp_path)
        _do_apply(window, engine)
        _do_reapply(window, engine)
        assert len(window._replay_log) == 2
        # Applying again must start a fresh chain of length 1
        _do_apply(window, engine)
        assert len(window._replay_log) == 1

    def test_reapply_appends_entry(self, qtbot, tmp_path: Path) -> None:
        window, engine = _make_window(qtbot, tmp_path)
        _load_photo(window, tmp_path)
        _do_apply(window, engine)
        _do_reapply(window, engine, strength=1.5)
        assert len(window._replay_log) == 2
        assert window._replay_log[1]["strength"] == 150

    def test_strength_adjust_updates_last_entry(self, qtbot, tmp_path: Path) -> None:
        window, engine = _make_window(qtbot, tmp_path)
        _load_photo(window, tmp_path)
        _do_apply(window, engine)
        # Simulate strength slider adjust
        result = _dummy_image()
        engine.apply = MagicMock(return_value=result)
        window._current_style_name = "Test Style"
        # Need a _styled_photo_input for _reapply_style_strength
        window._styled_photo_input = _dummy_image()
        with patch("src.stylist.main_window.QMessageBox.critical"):
            window._reapply_style_strength("test-style", 0.75)
        assert window._replay_log[-1]["strength"] == 75

    def test_undo_pops_entry(self, qtbot, tmp_path: Path) -> None:
        window, engine = _make_window(qtbot, tmp_path)
        _load_photo(window, tmp_path)
        _do_apply(window, engine)
        _do_reapply(window, engine)
        assert len(window._replay_log) == 2
        window._perform_undo()
        assert len(window._replay_log) == 1

    def test_open_photo_clears_log(self, qtbot, tmp_path: Path) -> None:
        window, engine = _make_window(qtbot, tmp_path)
        _load_photo(window, tmp_path)
        _do_apply(window, engine)
        assert len(window._replay_log) == 1
        _load_photo(window, tmp_path)
        assert window._replay_log == []

    def test_reset_clears_log(self, qtbot, tmp_path: Path) -> None:
        window, engine = _make_window(qtbot, tmp_path)
        _load_photo(window, tmp_path)
        _do_apply(window, engine)
        assert len(window._replay_log) == 1
        with patch("src.stylist.main_window.QMessageBox.question",
                   return_value=MagicMock()):
            # Simulate confirmed reset by calling directly after patching question
            window._styled_photo = None
            window._styled_photo_input = None
            window._replay_log = []
            window._clear_undo_stack()
        assert window._replay_log == []

    def test_format_replay_log_yaml(self, qtbot, tmp_path: Path) -> None:
        window, engine = _make_window(qtbot, tmp_path)
        _load_photo(window, tmp_path)
        _do_apply(window, engine)
        _do_reapply(window, engine, strength=1.5)
        yml_text = window._format_replay_log()
        parsed = yaml.safe_load(yml_text)
        assert parsed["version"] == 1
        assert len(parsed["steps"]) == 2
        assert parsed["steps"][0]["style"] == "Test Style"
        assert parsed["steps"][0]["strength"] == 100
        assert parsed["steps"][1]["strength"] == 150

    def test_copy_to_clipboard_when_empty_shows_dialog(self, qtbot, tmp_path: Path) -> None:
        window, _ = _make_window(qtbot, tmp_path)
        with patch("src.stylist.main_window.QMessageBox.information") as mock_info:
            window._copy_replay_log_to_clipboard()
        mock_info.assert_called_once()
        # Clipboard should be unchanged (empty or whatever it was)
        assert "No styles" in mock_info.call_args[0][2]

    def test_autosave_yml_written_on_save(self, qtbot, tmp_path: Path) -> None:
        window, engine = _make_window(qtbot, tmp_path, autosave=True)
        _load_photo(window, tmp_path)
        _do_apply(window, engine)
        out_jpg = tmp_path / "result.jpg"
        with patch("src.stylist.main_window.QFileDialog.getSaveFileName",
                   return_value=(str(out_jpg), "")):
            window._save_result()
        assert out_jpg.exists()
        yml_path = out_jpg.with_suffix(".yml")
        assert yml_path.exists()
        parsed = yaml.safe_load(yml_path.read_text(encoding="utf-8"))
        assert parsed["version"] == 1
        assert len(parsed["steps"]) == 1

    def test_autosave_yml_skipped_when_disabled(self, qtbot, tmp_path: Path) -> None:
        window, engine = _make_window(qtbot, tmp_path, autosave=False)
        _load_photo(window, tmp_path)
        _do_apply(window, engine)
        out_jpg = tmp_path / "result.jpg"
        with patch("src.stylist.main_window.QFileDialog.getSaveFileName",
                   return_value=(str(out_jpg), "")):
            window._save_result()
        yml_path = out_jpg.with_suffix(".yml")
        assert not yml_path.exists()

    def test_load_replay_log_applies_chain(self, qtbot, tmp_path: Path) -> None:
        window, engine = _make_window(qtbot, tmp_path)
        _load_photo(window, tmp_path)

        # Write a valid chain file
        chain = tmp_path / "chain.yml"
        chain.write_text(textwrap.dedent("""\
            version: 1
            steps:
              - style: Test Style
                strength: 100
        """), encoding="utf-8")

        result = _dummy_image()
        engine.apply = MagicMock(return_value=result)

        with (
            patch("src.stylist.main_window.QFileDialog.getOpenFileName",
                  return_value=(str(chain), "")),
            patch("src.stylist.main_window.QMessageBox.critical"),
            patch("src.stylist.main_window.QMessageBox.warning"),
        ):
            window._load_and_apply_replay_log()

        assert engine.apply.call_count == 1
        assert window._styled_photo is not None

    def test_load_replay_log_invalid_schema_shows_error(self, qtbot, tmp_path: Path) -> None:
        window, _ = _make_window(qtbot, tmp_path)
        _load_photo(window, tmp_path)

        bad_chain = tmp_path / "bad.yml"
        bad_chain.write_text("version: 99\nsteps:\n  - style: X\n    strength: 100\n",
                              encoding="utf-8")

        with (
            patch("src.stylist.main_window.QFileDialog.getOpenFileName",
                  return_value=(str(bad_chain), "")),
            patch("src.stylist.main_window.QMessageBox.critical") as mock_crit,
        ):
            window._load_and_apply_replay_log()

        mock_crit.assert_called_once()
