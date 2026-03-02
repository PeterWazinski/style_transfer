"""Tests for TrainingProgressDialog — pre-flight + live training progress."""
from __future__ import annotations

from pathlib import Path

import pytest

from src.ui.training_dialog import TrainingProgressDialog
from tests.ui.conftest import MockTrainer


# ---------------------------------------------------------------------------
# Pre-flight phase
# ---------------------------------------------------------------------------

class TestPreflightPhase:
    def test_warning_label_is_visible_on_open(
        self, training_dialog: TrainingProgressDialog
    ) -> None:
        """The pre-flight warning must be shown as soon as the dialog is created."""
        # isHidden() reflects the widget's own show/hide state independently
        # of whether the top-level window is displayed (test environment).
        assert not training_dialog.warning_label.isHidden()

    def test_progress_section_hidden_until_start(
        self, training_dialog: TrainingProgressDialog
    ) -> None:
        assert not training_dialog._progress_section.isVisible()

    def test_start_button_emits_user_confirmed(
        self, qtbot, training_dialog: TrainingProgressDialog
    ) -> None:
        with qtbot.waitSignal(training_dialog.user_confirmed, timeout=300):
            training_dialog.start_button.click()

    def test_clicking_start_reveals_progress_section(
        self, qtbot, training_dialog: TrainingProgressDialog
    ) -> None:
        training_dialog.start_button.click()
        assert not training_dialog._progress_section.isHidden()


# ---------------------------------------------------------------------------
# Progress updates
# ---------------------------------------------------------------------------

class TestProgressUpdates:
    def test_update_progress_sets_progress_bar_value(
        self, training_dialog: TrainingProgressDialog, mock_trainer: MockTrainer
    ) -> None:
        mock_trainer.emit_progress(batch=5, total=100, eta_seconds=7200)
        assert training_dialog.progress_bar.value() == 5

    def test_update_progress_sets_progress_bar_maximum(
        self, training_dialog: TrainingProgressDialog, mock_trainer: MockTrainer
    ) -> None:
        mock_trainer.emit_progress(batch=5, total=100, eta_seconds=7200)
        assert training_dialog.progress_bar.maximum() == 100

    def test_update_progress_shows_hours_in_eta(
        self, training_dialog: TrainingProgressDialog, mock_trainer: MockTrainer
    ) -> None:
        """7200 s = 2 h 0 m → ETA label contains '2h'."""
        mock_trainer.emit_progress(batch=5, total=100, eta_seconds=7200)
        assert "2h" in training_dialog.eta_label.text()

    def test_update_progress_shows_minutes_when_under_one_hour(
        self, training_dialog: TrainingProgressDialog, mock_trainer: MockTrainer
    ) -> None:
        mock_trainer.emit_progress(batch=10, total=100, eta_seconds=540)  # 9 minutes
        assert "9m" in training_dialog.eta_label.text()
        assert "h" not in training_dialog.eta_label.text()

    def test_update_progress_combined_hours_and_minutes(
        self, training_dialog: TrainingProgressDialog, mock_trainer: MockTrainer
    ) -> None:
        mock_trainer.emit_progress(batch=20, total=100, eta_seconds=5400)  # 1h 30m
        label = training_dialog.eta_label.text()
        assert "1h" in label
        assert "30m" in label


# ---------------------------------------------------------------------------
# Cancel
# ---------------------------------------------------------------------------

class TestCancel:
    def test_cancel_emits_training_cancelled(
        self, qtbot, training_dialog: TrainingProgressDialog
    ) -> None:
        with qtbot.waitSignal(training_dialog.training_cancelled, timeout=300):
            training_dialog.cancel_button.click()

    def test_cancel_without_worker_does_not_crash(
        self, training_dialog: TrainingProgressDialog
    ) -> None:
        """Cancel must be safe to call even before a worker was ever started."""
        training_dialog._on_cancel_clicked()   # must not raise

    def test_cancel_stops_running_worker(self, qtbot) -> None:
        """Worker started with a slow-but-interruptible trainer stops when cancelled."""

        class _InterruptibleTrainer:
            def train(self, *, progress_callback=None, **kwargs) -> Path:
                for i in range(500):
                    # Each call to progress_callback checks interruption
                    if progress_callback:
                        progress_callback(i, 500, 0)
                return Path("result.pth")

        dlg = TrainingProgressDialog(trainer=_InterruptibleTrainer())
        qtbot.addWidget(dlg)
        dlg.start_button.click()
        # Worker should have started
        assert dlg.is_training() or dlg._worker is not None

        with qtbot.waitSignal(dlg.training_cancelled, timeout=2000):
            dlg.cancel_button.click()

        assert not dlg.is_training()


# ---------------------------------------------------------------------------
# Successful completion
# ---------------------------------------------------------------------------

class TestCompletion:
    def test_instant_trainer_fires_training_completed(self, qtbot) -> None:
        class _InstantTrainer:
            def train(self, *, progress_callback=None, **kwargs) -> Path:
                return Path("output.pth")

        dlg = TrainingProgressDialog(trainer=_InstantTrainer())
        qtbot.addWidget(dlg)

        with qtbot.waitSignal(dlg.training_completed, timeout=500) as blocker:
            dlg.start_button.click()

        assert blocker.args[0] == str(Path("output.pth"))

    def test_is_training_false_after_completion(self, qtbot) -> None:
        class _InstantTrainer:
            def train(self, *, progress_callback=None, **kwargs) -> Path:
                return Path("output.pth")

        dlg = TrainingProgressDialog(trainer=_InstantTrainer())
        qtbot.addWidget(dlg)

        with qtbot.waitSignal(dlg.training_completed, timeout=500):
            dlg.start_button.click()

        assert not dlg.is_training()
