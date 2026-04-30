"""Unit and integration tests for ApplyWorker.

Covers:
- Successful inference → ``finished`` signal with PIL Image
- Progress callback → one ``progress(done, total)`` signal per tile
- Engine exception → ``error(str)`` signal; no ``finished``
- Cancellation before first tile → ``cancelled`` signal; no ``finished``
- Cursor restored after error and cancellation (integration via MainWindow)
- All three apply paths (_apply_style, _reapply_style, _reapply_style_strength)
  handle errors and cancellations without leaving the UI in a broken state
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image
from PySide6.QtCore import QCoreApplication, Qt
from PySide6.QtWidgets import QApplication

from src.core.engine import StyleTransferEngine
from src.core.models import StyleModel
from src.core.photo_manager import PhotoManager
from src.core.registry import StyleRegistry
from src.stylist.apply_worker import ApplyWorker
from src.stylist.main_window import MainWindow
from tests.helpers import make_mock_session


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dummy_image(size: int = 64) -> Image.Image:
    return Image.fromarray(np.zeros((size, size, 3), dtype=np.uint8))


def _make_mock_engine(
    *,
    raises: Exception | None = None,
    n_tiles: int = 1,
    result_colour: tuple[int, int, int] = (42, 42, 42),
) -> MagicMock:
    """Return a mock :class:`StyleTransferEngine` for *ApplyWorker* tests.

    The mock ``apply()`` method calls *progress_callback* once per tile and
    then returns a solid-colour PIL Image unless *raises* is set.
    """
    result_img = Image.fromarray(
        np.full((64, 64, 3), result_colour, dtype=np.uint8)
    )
    engine = MagicMock(spec=StyleTransferEngine)

    def _apply(
        source: Image.Image,
        style_id: str,
        *,
        strength: float,
        tile_size: int,
        overlap: int,
        use_float16: bool,
        progress_callback=None,
    ) -> Image.Image:
        if raises is not None:
            raise raises
        for i in range(n_tiles):
            if progress_callback is not None:
                progress_callback(i + 1, n_tiles)
        return result_img

    engine.apply.side_effect = _apply
    return engine


def _make_worker(engine: MagicMock) -> ApplyWorker:
    return ApplyWorker(
        engine=engine,
        source=_dummy_image(),
        style_id="test-style",
        strength=1.0,
        tile_size=64,
        overlap=0,
        use_float16=False,
    )


# ---------------------------------------------------------------------------
# ApplyWorker unit tests
# ---------------------------------------------------------------------------

class TestApplyWorkerSignals:
    def test_finished_emitted_on_success(self, qtbot) -> None:
        """``finished`` is emitted with a PIL Image when inference succeeds."""
        worker = _make_worker(_make_mock_engine(n_tiles=1))
        with qtbot.waitSignal(worker.finished, timeout=2000) as blocker:
            worker.start()
        worker.wait()
        result = blocker.args[0]
        assert isinstance(result, Image.Image)

    def test_progress_emitted_once_per_tile(self, qtbot) -> None:
        """One ``progress`` emission per tile, values strictly increasing."""
        n = 4
        worker = _make_worker(_make_mock_engine(n_tiles=n))
        calls: list[tuple[int, int]] = []
        worker.progress.connect(lambda done, total: calls.append((done, total)))

        with qtbot.waitSignal(worker.finished, timeout=2000):
            worker.start()
        worker.wait()

        assert len(calls) == n
        assert [d for d, _ in calls] == list(range(1, n + 1))
        assert all(total == n for _, total in calls)

    def test_error_emitted_on_engine_exception(self, qtbot) -> None:
        """``error`` is emitted with the exception message; ``finished`` is NOT."""
        worker = _make_worker(_make_mock_engine(raises=RuntimeError("onnx exploded")))
        finished_calls: list[object] = []
        worker.finished.connect(finished_calls.append)

        with qtbot.waitSignal(worker.error, timeout=2000) as blocker:
            worker.start()
        worker.wait()

        assert "onnx exploded" in blocker.args[0]
        assert finished_calls == [], "finished must NOT be emitted on error"

    def test_cancelled_emitted_when_interrupted_mid_run(self, qtbot) -> None:
        """Calling requestInterruption() while the engine is running triggers ``cancelled``.

        Qt resets the interruption flag on ``start()``, so we must request
        interruption *after* the thread has started.  A threading event pair
        (``running`` / ``proceed``) lets us inject the interruption
        deterministically before the engine's progress callback fires.
        """
        import threading

        running = threading.Event()  # worker signals it has entered apply()
        proceed = threading.Event()  # main thread lets the worker continue

        def _apply_gated(
            source: Image.Image,
            style_id: str,
            *,
            progress_callback=None,
            **kw: object,
        ) -> Image.Image:
            running.set()
            proceed.wait(timeout=2.0)
            if progress_callback is not None:
                progress_callback(1, 3)
            return _dummy_image()

        engine = MagicMock(spec=StyleTransferEngine)
        engine.apply.side_effect = _apply_gated

        worker = _make_worker(engine)
        finished_calls: list[object] = []
        error_calls: list[str] = []
        worker.finished.connect(finished_calls.append)
        worker.error.connect(error_calls.append)

        with qtbot.waitSignal(worker.cancelled, timeout=3000):
            worker.start()
            running.wait(timeout=2.0)     # wait until worker is inside apply()
            worker.requestInterruption()  # set flag before progress_callback fires
            proceed.set()                 # unblock the worker

        worker.wait()
        assert finished_calls == [], "finished must NOT be emitted on cancel"
        assert error_calls == [], "error must NOT be emitted on cancel"

    def test_finished_result_is_pil_image(self, qtbot) -> None:
        """The object emitted via ``finished`` is a PIL Image instance."""
        worker = _make_worker(_make_mock_engine(n_tiles=2))
        with qtbot.waitSignal(worker.finished, timeout=2000) as blocker:
            worker.start()
        worker.wait()
        assert isinstance(blocker.args[0], Image.Image)

    def test_progress_total_matches_n_tiles(self, qtbot) -> None:
        """The ``total`` field in every progress emission equals n_tiles."""
        n = 6
        worker = _make_worker(_make_mock_engine(n_tiles=n))
        totals: list[int] = []
        worker.progress.connect(lambda done, total: totals.append(total))

        with qtbot.waitSignal(worker.finished, timeout=2000):
            worker.start()
        worker.wait()

        assert all(t == n for t in totals)


# ---------------------------------------------------------------------------
# MainWindow integration: error handling
# ---------------------------------------------------------------------------

def _make_e2e_window(qtbot, tmp_path: Path) -> tuple[MainWindow, MagicMock]:
    """Return a (window, raw_engine) pair for integration tests."""
    preview = tmp_path / "preview.jpg"
    preview.parent.mkdir(parents=True, exist_ok=True)
    _dummy_image().save(preview)

    style = StyleModel(
        id="w-style",
        name="Window Style",
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
        engine.load_model("w-style", Path("dummy/model.onnx"))

    window = MainWindow(
        registry=registry,
        engine=engine,
        photo_manager=PhotoManager(),
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


class TestMainWindowErrorHandling:
    def test_cursor_restored_after_engine_error(
        self, qtbot, tmp_path: Path
    ) -> None:
        """WaitCursor must be cleaned up even when the engine raises."""
        window, engine = _make_e2e_window(qtbot, tmp_path)
        _load_photo(window, tmp_path)

        engine.apply = MagicMock(side_effect=RuntimeError("boom"))

        with patch("src.stylist.main_window.QMessageBox.critical"):
            window._apply_style("w-style", 1.0)

        # If the cursor stack is clean, overrideShape() returns None
        assert QApplication.overrideCursor() is None

    def test_apply_button_reenabled_after_engine_error(
        self, qtbot, tmp_path: Path
    ) -> None:
        """Apply button must be re-enabled when the engine raises."""
        window, engine = _make_e2e_window(qtbot, tmp_path)
        _load_photo(window, tmp_path)

        engine.apply = MagicMock(side_effect=RuntimeError("boom"))

        with patch("src.stylist.main_window.QMessageBox.critical"):
            window._apply_style("w-style", 1.0)

        assert window.canvas.apply_button.isEnabled()

    def test_styled_photo_not_set_after_error(
        self, qtbot, tmp_path: Path
    ) -> None:
        """_styled_photo must remain None when the engine fails."""
        window, engine = _make_e2e_window(qtbot, tmp_path)
        _load_photo(window, tmp_path)

        engine.apply = MagicMock(side_effect=RuntimeError("boom"))

        with patch("src.stylist.main_window.QMessageBox.critical"):
            window._apply_style("w-style", 1.0)

        assert window._styled_photo is None


# ---------------------------------------------------------------------------
# MainWindow integration: cancellation
# ---------------------------------------------------------------------------

class TestMainWindowCancellation:
    def test_cursor_restored_after_cancel(
        self, qtbot, tmp_path: Path
    ) -> None:
        """WaitCursor must be cleaned up when the user cancels."""
        window, engine = _make_e2e_window(qtbot, tmp_path)
        _load_photo(window, tmp_path)

        # Replace engine.apply with a version that triggers cancellation
        # by requesting interruption on the worker from the first tile's callback.
        original_apply = engine.apply

        def _cancelling_apply(source, style_id, *, progress_callback=None, **kw):
            # Reach into the window to request interruption on the active worker
            # by simulating dialog.cancel() via the progress_callback check path:
            # we just set an interruption flag on all living ApplyWorker instances.
            if progress_callback is not None:
                # Import private name to raise _CancelledError directly
                from src.stylist.apply_worker import _CancelledError  # noqa: PLC0415
                raise _CancelledError
            return original_apply(source, style_id, **kw)

        engine.apply = _cancelling_apply

        window._apply_style("w-style", 1.0)

        assert QApplication.overrideCursor() is None

    def test_styled_photo_not_set_after_cancel(
        self, qtbot, tmp_path: Path
    ) -> None:
        """_styled_photo must remain None when the user cancels."""
        window, engine = _make_e2e_window(qtbot, tmp_path)
        _load_photo(window, tmp_path)

        from src.stylist.apply_worker import _CancelledError  # noqa: PLC0415

        engine.apply = MagicMock(side_effect=_CancelledError)

        window._apply_style("w-style", 1.0)

        assert window._styled_photo is None

    def test_apply_button_reenabled_after_cancel(
        self, qtbot, tmp_path: Path
    ) -> None:
        """Apply button must be re-enabled after a cancellation."""
        window, engine = _make_e2e_window(qtbot, tmp_path)
        _load_photo(window, tmp_path)

        from src.stylist.apply_worker import _CancelledError  # noqa: PLC0415

        engine.apply = MagicMock(side_effect=_CancelledError)

        window._apply_style("w-style", 1.0)

        assert window.canvas.apply_button.isEnabled()


# ---------------------------------------------------------------------------
# All three apply paths succeed
# ---------------------------------------------------------------------------

class TestAllThreeApplyPaths:
    def test_apply_style_stores_result(self, qtbot, tmp_path: Path) -> None:
        window, _ = _make_e2e_window(qtbot, tmp_path)
        _load_photo(window, tmp_path)
        window._apply_style("w-style", 1.0)
        assert window._styled_photo is not None
        assert window.canvas.has_styled()

    def test_reapply_style_uses_styled_photo(self, qtbot, tmp_path: Path) -> None:
        window, engine = _make_e2e_window(qtbot, tmp_path)
        _load_photo(window, tmp_path)
        window._apply_style("w-style", 1.0)

        captured: list[Image.Image] = []
        original_apply = engine.apply

        def _spy(photo, *a, **kw):
            captured.append(photo)
            return original_apply(photo, *a, **kw)

        engine.apply = _spy
        before = window._styled_photo
        window._reapply_style("w-style", 1.0)

        assert len(captured) == 1
        assert np.array_equal(np.array(captured[0]), np.array(before))

    def test_reapply_strength_uses_styled_photo_input(
        self, qtbot, tmp_path: Path
    ) -> None:
        window, engine = _make_e2e_window(qtbot, tmp_path)
        _load_photo(window, tmp_path)
        window._apply_style("w-style", 1.0)

        captured: list[Image.Image] = []
        original_apply = engine.apply

        def _spy(photo, *a, **kw):
            captured.append(photo)
            return original_apply(photo, *a, **kw)

        engine.apply = _spy
        expected_input = window._styled_photo_input
        window._reapply_style_strength("w-style", 0.5)

        assert len(captured) == 1
        assert np.array_equal(np.array(captured[0]), np.array(expected_input))


# ---------------------------------------------------------------------------
# Undo stack
# ---------------------------------------------------------------------------

class TestUndoStack:
    def test_undo_button_disabled_initially(
        self, qtbot, tmp_path: Path
    ) -> None:
        window, _ = _make_e2e_window(qtbot, tmp_path)
        _load_photo(window, tmp_path)
        assert not window.canvas.undo_button.isEnabled()

    def test_undo_button_enabled_after_apply(
        self, qtbot, tmp_path: Path
    ) -> None:
        window, _ = _make_e2e_window(qtbot, tmp_path)
        _load_photo(window, tmp_path)
        window._apply_style("w-style", 1.0)
        assert window.canvas.undo_button.isEnabled()

    def test_undo_after_apply_clears_styled_state(
        self, qtbot, tmp_path: Path
    ) -> None:
        window, _ = _make_e2e_window(qtbot, tmp_path)
        _load_photo(window, tmp_path)
        window._apply_style("w-style", 1.0)
        assert window._styled_photo is not None
        window._perform_undo()
        assert window._styled_photo is None
        assert not window.canvas.has_styled()

    def test_undo_after_reapply_restores_first_result(
        self, qtbot, tmp_path: Path
    ) -> None:
        window, _ = _make_e2e_window(qtbot, tmp_path)
        _load_photo(window, tmp_path)
        window._apply_style("w-style", 1.0)
        first_result = window._styled_photo
        window._reapply_style("w-style", 1.0)
        window._perform_undo()
        assert window._styled_photo is first_result

    def test_undo_stack_capped_at_three(
        self, qtbot, tmp_path: Path
    ) -> None:
        """After 4 applies only 3 undos are available; the 4th undo is a no-op."""
        window, _ = _make_e2e_window(qtbot, tmp_path)
        _load_photo(window, tmp_path)
        window._apply_style("w-style", 1.0)
        window._reapply_style("w-style", 1.0)
        window._reapply_style("w-style", 1.0)
        window._reapply_style("w-style", 1.0)  # 4th: oldest slot drops off
        window._perform_undo()
        window._perform_undo()
        window._perform_undo()
        assert not window.canvas.undo_button.isEnabled()

    def test_undo_button_disabled_after_open_photo(
        self, qtbot, tmp_path: Path
    ) -> None:
        window, _ = _make_e2e_window(qtbot, tmp_path)
        _load_photo(window, tmp_path)
        window._apply_style("w-style", 1.0)
        assert window.canvas.undo_button.isEnabled()
        _load_photo(window, tmp_path)
        assert not window.canvas.undo_button.isEnabled()

    def test_undo_does_not_reenable_apply_after_gpu_crash(
        self, qtbot, tmp_path: Path
    ) -> None:
        """Undo must not re-enable Apply if the GPU crash handler disabled it."""
        window, engine = _make_e2e_window(qtbot, tmp_path)
        _load_photo(window, tmp_path)
        window._apply_style("w-style", 1.0)
        # Simulate GPU crash disabling the buttons
        window.canvas.apply_button.setEnabled(False)
        window.canvas.reapply_button.setEnabled(False)
        window._perform_undo()
        # Undo restores image state but must NOT re-enable the crashed buttons
        assert not window.canvas.apply_button.isEnabled()
        assert not window.canvas.reapply_button.isEnabled()
