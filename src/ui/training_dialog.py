"""TrainingProgressDialog — trains a new custom style in a background QThread.

Flow
----
1. Dialog opens → **pre-flight section** visible: warning text + "Start Training".
2. User clicks "Start Training" → :attr:`user_confirmed` emitted → worker launched.
3. Worker emits :attr:`TrainingWorker.progress` → progress bar + ETA update.
4. Worker finishes → :attr:`training_completed` emitted → dialog auto-closes.
5. User clicks "Cancel" at any time → worker interrupted → :attr:`training_cancelled`.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Optional, Protocol, runtime_checkable

from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import (
    QDialog,
    QGroupBox,
    QLabel,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

logger: logging.Logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Protocol (duck-typed trainer interface so tests can inject mocks easily)
# ---------------------------------------------------------------------------

@runtime_checkable
class _TrainerProtocol(Protocol):
    def train(
        self,
        *,
        progress_callback: Optional[Callable[[int, int, int], None]],
        **kwargs: object,
    ) -> Path: ...


# ---------------------------------------------------------------------------
# Background worker
# ---------------------------------------------------------------------------

class TrainingWorker(QThread):
    """Runs :meth:`_TrainerProtocol.train` in a background thread.

    Signals:
        progress(int, int, int): Fired periodically: ``(batch, total, eta_seconds)``.
        finished_ok(str):        Training succeeded; payload is the saved ``.pth`` path.
        error_occurred(str):     An exception was raised during training.
    """

    progress: Signal = Signal(int, int, int)
    finished_ok: Signal = Signal(str)
    error_occurred: Signal = Signal(str)

    def __init__(
        self,
        trainer: _TrainerProtocol,
        train_kwargs: dict[str, object],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._trainer = trainer
        self._train_kwargs = train_kwargs

    def run(self) -> None:  # called in worker thread
        def _cb(batch: int, total: int, eta_seconds: int) -> None:
            if self.isInterruptionRequested():
                raise InterruptedError("Training cancelled by user.")
            self.progress.emit(batch, total, eta_seconds)

        try:
            result: Path = self._trainer.train(
                progress_callback=_cb,
                **self._train_kwargs,
            )
            self.finished_ok.emit(str(result))
        except InterruptedError:
            logger.info("Training was cancelled.")
        except Exception as exc:  # noqa: BLE001
            logger.exception("Training error: %s", exc)
            self.error_occurred.emit(str(exc))


# ---------------------------------------------------------------------------
# Dialog
# ---------------------------------------------------------------------------

class TrainingProgressDialog(QDialog):
    """Two-phase dialog: pre-flight warning then live progress.

    Args:
        trainer:      An object conforming to :class:`_TrainerProtocol`.
                      Pass *None* to create the dialog without actual training
                      capability (useful in tests).
        train_kwargs: Keyword arguments forwarded verbatim to ``trainer.train()``.
        parent:       Optional parent widget.

    Signals:
        user_confirmed():        "Start Training" button was clicked.
        training_completed(str): Training succeeded; ``str`` = saved model path.
        training_cancelled():    Cancel was clicked or the dialog was closed.
    """

    user_confirmed: Signal = Signal()
    training_completed: Signal = Signal(str)
    training_cancelled: Signal = Signal()

    def __init__(
        self,
        trainer: Optional[_TrainerProtocol] = None,
        train_kwargs: Optional[dict[str, object]] = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._trainer = trainer
        self._train_kwargs: dict[str, object] = train_kwargs or {}
        self._worker: TrainingWorker | None = None
        self.setWindowTitle("Train Custom Style")
        self.setMinimumWidth(420)
        self._build_ui()

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)

        # --- Pre-flight section ---
        preflight_box = QGroupBox("Before you start", self)
        pf_layout = QVBoxLayout(preflight_box)
        self.warning_label = QLabel(
            "⚠  Training a new style can take several hours even on a capable GPU.\n\n"
            "• Estimated time on a mid-range GPU (RTX 3060): 2–4 h per epoch.\n"
            "• On CPU only this can take 40+ hours — not recommended.\n"
            "• Make sure your COCO dataset path is configured in Settings.\n\n"
            "Checkpoints are saved every 1 000 images so you can resume after\n"
            "an interruption.",
            self,
        )
        self.warning_label.setWordWrap(True)
        pf_layout.addWidget(self.warning_label)
        root.addWidget(preflight_box)

        self.start_button = QPushButton("Start Training", self)
        root.addWidget(self.start_button)

        # --- Progress section (hidden until training starts) ---
        self._progress_section = QWidget(self)
        prog_layout = QVBoxLayout(self._progress_section)
        prog_layout.setContentsMargins(0, 0, 0, 0)

        self.progress_bar = QProgressBar(self)
        self.progress_bar.setMinimum(0)
        self.progress_bar.setValue(0)
        prog_layout.addWidget(self.progress_bar)

        self.eta_label = QLabel("ETA: —", self)
        prog_layout.addWidget(self.eta_label)

        self._progress_section.setVisible(False)
        root.addWidget(self._progress_section)

        # Cancel is always visible so tests can click it regardless of phase.
        self.cancel_button = QPushButton("Cancel", self)
        root.addWidget(self.cancel_button)

        # --- Connections ---
        self.start_button.clicked.connect(self._on_start_clicked)
        self.cancel_button.clicked.connect(self._on_cancel_clicked)

    # ------------------------------------------------------------------
    # Public slots
    # ------------------------------------------------------------------

    def update_progress(self, batch: int, total: int, eta_seconds: int) -> None:
        """Update the progress bar and ETA label.

        Can be called directly from tests or wired to a worker signal.

        Args:
            batch:       Current batch / image index.
            total:       Total batches / images.
            eta_seconds: Estimated seconds remaining.
        """
        self.progress_bar.setMaximum(max(total, 1))
        self.progress_bar.setValue(batch)

        hours = eta_seconds // 3600
        minutes = (eta_seconds % 3600) // 60
        if hours > 0:
            self.eta_label.setText(f"ETA: {hours}h {minutes}m")
        else:
            self.eta_label.setText(f"ETA: {minutes}m")

    def is_training(self) -> bool:
        """Return *True* if the worker thread is currently running."""
        return self._worker is not None and self._worker.isRunning()

    # ------------------------------------------------------------------
    # Private slots
    # ------------------------------------------------------------------

    def _on_start_clicked(self) -> None:
        self.user_confirmed.emit()
        self.start_button.setEnabled(False)
        self._progress_section.setVisible(True)

        if self._trainer is not None:
            self._worker = TrainingWorker(self._trainer, self._train_kwargs, self)
            self._worker.progress.connect(
                lambda b, t, e: self.update_progress(b, t, e)
            )
            self._worker.finished_ok.connect(self._on_worker_finished)
            self._worker.error_occurred.connect(self._on_worker_error)
            self._worker.start()

    def _on_cancel_clicked(self) -> None:
        if self._worker and self._worker.isRunning():
            self._worker.requestInterruption()
            self._worker.quit()
            self._worker.wait(2000)
        self.training_cancelled.emit()
        self.reject()

    def _on_worker_finished(self, model_path: str) -> None:
        self.training_completed.emit(model_path)
        self.accept()

    def _on_worker_error(self, message: str) -> None:
        self.eta_label.setText(f"Error: {message}")
        self.start_button.setEnabled(True)

    def closeEvent(self, event) -> None:  # type: ignore[override]
        if self.is_training():
            self._on_cancel_clicked()
        super().closeEvent(event)
