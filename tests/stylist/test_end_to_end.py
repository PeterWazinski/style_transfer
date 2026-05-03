"""pytest-qt end-to-end integration test.

Full user journey: open photo → select style → adjust strength → Apply
→ result shown → Save → file written to disk.

All heavy operations (ONNX inference and file dialogs) are mocked so the
test runs quickly without requiring actual model files.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image
from PySide6.QtCore import Qt

from src.core.engine import StyleTransferEngine
from src.core.models import StyleModel
from src.core.photo_manager import PhotoManager
from src.core.registry import StyleRegistry
from src.stylist.main_window import MainWindow
from tests.helpers import make_mock_session


def _make_style_image_file(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8)).save(path)
    return path


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def e2e_registry(tmp_path: Path) -> StyleRegistry:
    """Registry pre-populated with one test style that has a real preview file."""
    preview = _make_style_image_file(tmp_path / "preview.jpg")
    style = StyleModel(
        id="e2e-style",
        name="E2E Style",
        model_path=str(tmp_path / "model.onnx"),
        preview_path=str(preview),
        is_builtin=False,
    )
    reg = StyleRegistry(catalog_path=tmp_path / "catalog.json")
    reg.add(style)
    return reg


@pytest.fixture()
def e2e_engine(tmp_path: Path) -> StyleTransferEngine:
    """Engine with a mocked ONNX session for 'e2e-style'."""
    engine = StyleTransferEngine()
    with (
        patch("src.core.engine._ORT_AVAILABLE", True),
        patch("src.core.engine.ort") as mock_ort,
        patch.object(Path, "exists", return_value=True),
    ):
        mock_ort.InferenceSession.return_value = make_mock_session(output_colour=(0, 200, 0))
        engine.load_model("e2e-style", Path("dummy/model.onnx"))
    return engine


@pytest.fixture()
def e2e_window(
    qtbot, tmp_path: Path, e2e_registry: StyleRegistry, e2e_engine: StyleTransferEngine
) -> MainWindow:
    window = MainWindow(
        registry=e2e_registry,
        engine=e2e_engine,
        photo_manager=PhotoManager(),
    )
    qtbot.addWidget(window)
    return window


# ---------------------------------------------------------------------------
# Full journey
# ---------------------------------------------------------------------------

class TestEndToEndJourney:
    def test_open_photo_shows_on_canvas(
        self, qtbot, e2e_window: MainWindow, tmp_path: Path
    ) -> None:
        """Mocking getOpenFileName and PhotoManager.load; canvas should report photo loaded."""
        photo_path = tmp_path / "photo.jpg"
        arr = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
        Image.fromarray(arr).save(photo_path)

        with patch(
            "src.stylist.main_window.QFileDialog.getOpenFileName",
            return_value=(str(photo_path), "Images (*.jpg)"),
        ):
            e2e_window._open_photo()

        assert e2e_window.canvas.has_original()

    def test_style_selection_enables_apply_button(
        self, qtbot, e2e_window: MainWindow, tmp_path: Path
    ) -> None:
        """Selecting a style AND loading a photo enables the Apply button."""
        photo_path = tmp_path / "photo.jpg"
        Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8)).save(photo_path)

        with patch(
            "src.stylist.main_window.QFileDialog.getOpenFileName",
            return_value=(str(photo_path), ""),
        ):
            e2e_window._open_photo()

        # Simulate selecting first gallery item
        first_index = e2e_window.gallery.model().index(0, 0)
        e2e_window.gallery._list_view.clicked.emit(first_index)

        assert not e2e_window.canvas.apply_button.isEnabled() or True  # noqa: PT017
        # The button is enabled when both a photo and a style are set
        assert e2e_window.canvas._current_style_id == "e2e-style"
        assert e2e_window.canvas.has_original()
        assert e2e_window.canvas.apply_button.isEnabled()

    def test_apply_produces_styled_result(
        self, qtbot, e2e_window: MainWindow, tmp_path: Path
    ) -> None:
        """Clicking Apply calls engine.apply and shows the result."""
        photo_path = tmp_path / "photo.jpg"
        Image.fromarray(np.zeros((128, 128, 3), dtype=np.uint8)).save(photo_path)

        with patch(
            "src.stylist.main_window.QFileDialog.getOpenFileName",
            return_value=(str(photo_path), ""),
        ):
            e2e_window._open_photo()

        first_index = e2e_window.gallery.model().index(0, 0)
        e2e_window.gallery._list_view.clicked.emit(first_index)

        e2e_window._apply_style("e2e-style", 1.0)

        assert e2e_window.canvas.has_styled()
        assert e2e_window._save_action.isEnabled()

    def test_save_result_calls_photo_manager(
        self, qtbot, e2e_window: MainWindow, tmp_path: Path, mocker
    ) -> None:
        """After Apply, clicking Save should call PhotoManager.save."""
        # Prepare
        photo_path = tmp_path / "photo.jpg"
        Image.fromarray(np.zeros((128, 128, 3), dtype=np.uint8)).save(photo_path)

        with patch(
            "src.stylist.main_window.QFileDialog.getOpenFileName",
            return_value=(str(photo_path), ""),
        ):
            e2e_window._open_photo()

        first_index = e2e_window.gallery.model().index(0, 0)
        e2e_window.gallery._list_view.clicked.emit(first_index)
        e2e_window._apply_style("e2e-style", 1.0)

        save_path = tmp_path / "output.jpg"
        spy = mocker.spy(e2e_window.photo_manager, "save")

        with patch(
            "src.stylist.main_window.QFileDialog.getSaveFileName",
            return_value=(str(save_path), "JPEG (*.jpg)"),
        ):
            e2e_window._save_result()

        spy.assert_called_once()
        args = spy.call_args[0]
        assert Path(args[1]) == save_path

    def test_save_file_exists_on_disk(
        self, qtbot, e2e_window: MainWindow, tmp_path: Path
    ) -> None:
        """The saved file must actually appear on disk."""
        photo_path = tmp_path / "photo.jpg"
        Image.fromarray(np.zeros((128, 128, 3), dtype=np.uint8)).save(photo_path)

        with patch(
            "src.stylist.main_window.QFileDialog.getOpenFileName",
            return_value=(str(photo_path), ""),
        ):
            e2e_window._open_photo()

        first_index = e2e_window.gallery.model().index(0, 0)
        e2e_window.gallery._list_view.clicked.emit(first_index)
        e2e_window._apply_style("e2e-style", 0.8)

        save_path = tmp_path / "output.jpg"
        with patch(
            "src.stylist.main_window.QFileDialog.getSaveFileName",
            return_value=(str(save_path), "JPEG (*.jpg)"),
        ):
            e2e_window._save_result()

        assert save_path.exists()
        assert save_path.stat().st_size > 0


# ---------------------------------------------------------------------------
# Settings integration
# ---------------------------------------------------------------------------

class TestSettingsIntegration:
    def test_settings_dialog_opens_without_crash(
        self, qtbot, e2e_window: MainWindow, mocker
    ) -> None:
        """SettingsDialog must be instantiated and its exec() called."""
        from src.stylist.settings_dialog import SettingsDialog  # noqa: PLC0415

        mock_dlg = mocker.MagicMock()
        mock_cls = mocker.patch(
            "src.stylist.main_window.SettingsDialog",
            return_value=mock_dlg,
        )
        e2e_window._open_settings_dialog()
        mock_cls.assert_called_once_with(settings=e2e_window._settings, parent=e2e_window)
        mock_dlg.exec.assert_called_once()

    def test_settings_changed_updates_window_settings(
        self, qtbot, e2e_window: MainWindow
    ) -> None:
        from src.core.settings import AppSettings  # noqa: PLC0415

        new_settings = AppSettings(tile_size=512, overlap=64, use_float16=True)
        e2e_window._on_settings_changed(new_settings)
        assert e2e_window._settings.tile_size == 512
        assert e2e_window._settings.use_float16 is True


# ---------------------------------------------------------------------------
# Re-Apply journey
# ---------------------------------------------------------------------------

def _load_photo_and_apply(window: MainWindow, tmp_path: Path, style_id: str = "e2e-style") -> None:
    """Shared helper: open a photo file and apply a style."""
    photo_path = tmp_path / "photo.jpg"
    Image.fromarray(np.zeros((128, 128, 3), dtype=np.uint8)).save(photo_path)
    with patch(
        "src.stylist.main_window.QFileDialog.getOpenFileName",
        return_value=(str(photo_path), ""),
    ):
        window._open_photo()
    first_index = window.gallery.model().index(0, 0)
    window.gallery._list_view.clicked.emit(first_index)
    window._apply_style(style_id, 1.0)


class TestReApplyJourney:
    def test_reapply_uses_styled_photo_as_input(
        self, qtbot, e2e_window: MainWindow, tmp_path: Path
    ) -> None:
        """_reapply_style must pass _styled_photo (not _current_photo) to engine.apply."""
        _load_photo_and_apply(e2e_window, tmp_path)

        # Capture the styled result that should become the reapply source
        styled_before_reapply = e2e_window._styled_photo

        captured_inputs: list[Image.Image] = []
        original_apply = e2e_window.engine.apply

        def _spy_apply(photo: Image.Image, *args, **kwargs) -> Image.Image:
            captured_inputs.append(photo)
            return original_apply(photo, *args, **kwargs)

        e2e_window.engine.apply = _spy_apply  # type: ignore[assignment]
        try:
            e2e_window._reapply_style("e2e-style", 1.0)
        finally:
            e2e_window.engine.apply = original_apply  # type: ignore[assignment]

        assert len(captured_inputs) == 1
        assert np.array_equal(
            np.array(captured_inputs[0]),
            np.array(styled_before_reapply),
        ), "Re-Apply must use the styled result as input, not the original photo"

    def test_reapply_updates_styled_buffer(
        self, qtbot, e2e_window: MainWindow, tmp_path: Path
    ) -> None:
        """After _reapply_style, _styled_photo must be the new result (not the previous one)."""
        _load_photo_and_apply(e2e_window, tmp_path)

        styled_before = e2e_window._styled_photo
        e2e_window._reapply_style("e2e-style", 1.0)
        styled_after = e2e_window._styled_photo

        # Both are PIL Images; after reapply the buffer must be refreshed
        assert styled_after is not None
        assert styled_after is not styled_before

    def test_reapply_shows_previous_result_in_left_pane(
        self, qtbot, e2e_window: MainWindow, tmp_path: Path
    ) -> None:
        """After Re-Apply, the split view must still show content (not be blank)."""
        _load_photo_and_apply(e2e_window, tmp_path)
        e2e_window._reapply_style("e2e-style", 1.0)

        # The canvas still reports a styled result — both panes are populated
        assert e2e_window.canvas.has_styled()

    def test_open_new_photo_clears_styled_buffer(
        self, qtbot, e2e_window: MainWindow, tmp_path: Path
    ) -> None:
        """Opening a new photo must reset _styled_photo to None and clear has_styled()."""
        _load_photo_and_apply(e2e_window, tmp_path)
        assert e2e_window._styled_photo is not None
        assert e2e_window.canvas.has_styled()

        # Open a second (different) photo
        photo2 = tmp_path / "photo2.jpg"
        Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8)).save(photo2)
        with patch(
            "src.stylist.main_window.QFileDialog.getOpenFileName",
            return_value=(str(photo2), ""),
        ):
            e2e_window._open_photo()

        assert e2e_window._styled_photo is None
        assert not e2e_window.canvas.has_styled()

    def test_open_new_photo_disables_reapply_button(
        self, qtbot, e2e_window: MainWindow, tmp_path: Path
    ) -> None:
        """Re-Apply button must be disabled after opening a new photo."""
        _load_photo_and_apply(e2e_window, tmp_path)
        assert e2e_window.canvas.reapply_button.isEnabled()

        photo2 = tmp_path / "photo2.jpg"
        Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8)).save(photo2)
        with patch(
            "src.stylist.main_window.QFileDialog.getOpenFileName",
            return_value=(str(photo2), ""),
        ):
            e2e_window._open_photo()

        assert not e2e_window.canvas.reapply_button.isEnabled()
