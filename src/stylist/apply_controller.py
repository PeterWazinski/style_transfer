"""ApplyController mixin — style-apply operations for MainWindow.

Provides ``_create_progress_dialog``, ``_run_apply_worker``,
``_apply_style``, ``_reapply_style``, and ``_reapply_style_strength``.
Mixed into :class:`src.stylist.main_window.MainWindow`.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from PIL.Image import Image as PILImage
from PySide6.QtCore import Qt, QEventLoop
from PySide6.QtWidgets import QApplication, QMessageBox, QProgressDialog

from src.stylist.apply_worker import ApplyWorker, is_gpu_crash as _is_gpu_crash

if TYPE_CHECKING:
    from src.stylist.main_window import MainWindow

logger: logging.Logger = logging.getLogger(__name__)


class ApplyController:
    """Mixin that adds apply/re-apply operations to MainWindow.

    All attributes accessed via ``self`` (e.g. ``self.engine``,
    ``self._current_photo``) are expected to be present on the
    concrete :class:`~src.stylist.main_window.MainWindow` instance at
    runtime.
    """

    # ------------------------------------------------------------------
    # Progress dialog helpers
    # ------------------------------------------------------------------

    def _create_progress_dialog(  # type: ignore[misc]
        self: "MainWindow",
        label: str = "Processing tiles\u2026",
    ) -> QProgressDialog:
        """Return a modal :class:`QProgressDialog` centred over this window."""
        dlg = QProgressDialog(label, "Cancel", 0, 100, self)  # type: ignore[call-arg]
        dlg.setWindowTitle("Applying Style")
        dlg.setWindowModality(Qt.WindowModality.WindowModal)
        dlg.setMinimumDuration(400)
        dlg.setAutoClose(False)
        dlg.setAutoReset(False)
        dlg.setValue(0)
        return dlg

    def _run_apply_worker(  # type: ignore[misc]
        self: "MainWindow",
        source: PILImage,
        style_id: str,
        strength: float,
        dlg: QProgressDialog,
    ) -> PILImage | None:
        """Run :class:`ApplyWorker` in a background thread and wait for it."""
        worker = ApplyWorker(
            engine=self.engine,
            source=source,
            style_id=style_id,
            strength=strength,
            tile_size=self._settings.tile_size,
            overlap=self._settings.overlap,
            use_float16=self._settings.use_float16,
        )

        result_holder: list[PILImage | None] = [None]
        error_holder: list[str | None] = [None]
        cancelled_holder: list[bool] = [False]
        loop = QEventLoop()

        def _on_progress(done: int, total: int) -> None:
            if total > 0:
                dlg.setValue(int(done / total * 100))

        def _on_finished(img: PILImage) -> None:
            result_holder[0] = img
            loop.quit()

        def _on_error(msg: str) -> None:
            error_holder[0] = msg
            loop.quit()

        def _on_cancelled() -> None:
            cancelled_holder[0] = True
            loop.quit()

        worker.progress.connect(_on_progress)
        worker.finished.connect(_on_finished)
        worker.error.connect(_on_error)
        worker.cancelled.connect(_on_cancelled)
        dlg.canceled.connect(worker.requestInterruption)

        worker.start()
        loop.exec()
        worker.wait()

        if cancelled_holder[0]:
            return None
        if error_holder[0]:
            msg = error_holder[0]
            logger.error("Style transfer error: %s", msg)
            if _is_gpu_crash(msg):
                QMessageBox.critical(self, "GPU Driver Error", msg)  # type: ignore[call-arg]
                _restart_tip = "GPU driver crashed — please restart the application."
                self.canvas.apply_button.setEnabled(False)
                self.canvas.apply_button.setToolTip(_restart_tip)
                self.canvas.reapply_button.setEnabled(False)
                self.canvas.reapply_button.setToolTip(_restart_tip)
            else:
                QMessageBox.critical(self, "Apply Error", msg)  # type: ignore[call-arg]
            return None
        return result_holder[0]

    # ------------------------------------------------------------------
    # Apply / Re-Apply
    # ------------------------------------------------------------------

    def _reapply_style(  # type: ignore[misc]
        self: "MainWindow",
        style_id: str,
        strength: float,
    ) -> None:
        """Apply *style_id* to the already-styled photo (chain styles)."""
        source_photo = self._styled_photo
        if source_photo is None:
            return
        self._status.showMessage("Re-applying style\u2026")
        self.canvas.apply_button.setEnabled(False)
        self.canvas.reapply_button.setEnabled(False)
        QApplication.setOverrideCursor(Qt.WaitCursor)  # type: ignore[attr-defined]
        dlg = self._create_progress_dialog("Re-applying style\u2026")
        try:
            result = self._run_apply_worker(source_photo, style_id, strength, dlg)
        finally:
            dlg.close()
            QApplication.restoreOverrideCursor()
            self.canvas.apply_button.setEnabled(True)
            self.canvas.reapply_button.setEnabled(True)
        if result is None:
            self._status.showMessage("Re-apply cancelled." if dlg.wasCanceled() else "Error during style transfer.")
            return
        self._push_undo_snapshot()
        self._replay_log.append({"style": self._current_style_name, "strength": int(strength * 100)})
        self.canvas.split_view.set_original_pixmap(self._pil_to_pixmap(source_photo))
        self._left_pane_pil = source_photo
        self._styled_photo_input = source_photo
        self._styled_photo = result
        self.canvas.set_styled(self._pil_to_pixmap(result))
        self._save_action.setEnabled(True)
        self._status.showMessage("Style re-applied.")

    def _reapply_style_strength(  # type: ignore[misc]
        self: "MainWindow",
        style_id: str,
        strength: float,
    ) -> None:
        """Re-run the current chain step with a new strength (no chain advance)."""
        source = self._styled_photo_input
        if source is None:
            self._apply_style(style_id, strength)
            return
        self._status.showMessage("Adjusting strength\u2026")
        self.canvas.apply_button.setEnabled(False)
        self.canvas.reapply_button.setEnabled(False)
        QApplication.setOverrideCursor(Qt.WaitCursor)  # type: ignore[attr-defined]
        dlg = self._create_progress_dialog("Adjusting strength\u2026")
        try:
            result = self._run_apply_worker(source, style_id, strength, dlg)
        finally:
            dlg.close()
            QApplication.restoreOverrideCursor()
            self.canvas.apply_button.setEnabled(True)
            self.canvas.reapply_button.setEnabled(True)
        if result is None:
            self._status.showMessage("Adjustment cancelled." if dlg.wasCanceled() else "Error during style transfer.")
            return
        self._styled_photo = result
        if self._replay_log:
            self._replay_log[-1]["strength"] = int(strength * 100)
        self.canvas.set_styled(self._pil_to_pixmap(result))
        self._save_action.setEnabled(True)
        self._status.showMessage("Strength adjusted.")

    def _apply_style(  # type: ignore[misc]
        self: "MainWindow",
        style_id: str,
        strength: float,
    ) -> None:
        if self._current_photo is None:
            return
        self._status.showMessage("Applying style\u2026")
        self.canvas.apply_button.setEnabled(False)
        QApplication.setOverrideCursor(Qt.WaitCursor)  # type: ignore[attr-defined]
        dlg = self._create_progress_dialog()
        try:
            result = self._run_apply_worker(self._current_photo, style_id, strength, dlg)
        finally:
            dlg.close()
            QApplication.restoreOverrideCursor()
            self.canvas.apply_button.setEnabled(True)
        if result is None:
            self._status.showMessage("Apply cancelled." if dlg.wasCanceled() else "Error during style transfer.")
            return
        self._push_undo_snapshot()
        self._replay_log = [{"style": self._current_style_name, "strength": int(strength * 100)}]
        self._styled_photo_input = self._current_photo
        self._styled_photo = result
        self.canvas.set_styled(self._pil_to_pixmap(result))
        self._save_action.setEnabled(True)
        self._status.showMessage("Style applied.")
