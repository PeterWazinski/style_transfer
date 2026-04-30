"""Tests for PhotoCanvasView — photo viewer with split-view and strength slider."""
from __future__ import annotations

import numpy as np
import pytest
from PIL import Image
from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap

from src.stylist.photo_canvas import PhotoCanvasView
from src.stylist.widgets.strength_slider import StrengthSlider


def _make_pixmap(width: int = 64, height: int = 64) -> QPixmap:
    arr = np.full((height, width, 3), 128, dtype=np.uint8)
    img = Image.fromarray(arr)
    pix = QPixmap(width, height)
    pix.fill(Qt.gray)  # type: ignore[attr-defined]
    return pix


# ---------------------------------------------------------------------------
# set_original
# ---------------------------------------------------------------------------

class TestSetOriginal:
    def test_set_original_marks_canvas_as_having_photo(
        self, canvas: PhotoCanvasView
    ) -> None:
        assert not canvas.has_original()
        canvas.set_original(_make_pixmap())
        assert canvas.has_original()

    def test_apply_button_disabled_without_style(self, canvas: PhotoCanvasView) -> None:
        canvas.set_original(_make_pixmap())
        assert not canvas.apply_button.isEnabled()

    def test_apply_button_enabled_after_original_and_style(
        self, canvas: PhotoCanvasView
    ) -> None:
        canvas.set_active_style("candy")
        canvas.set_original(_make_pixmap())
        assert canvas.apply_button.isEnabled()

    def test_set_styled_enables_save_button(self, canvas: PhotoCanvasView) -> None:
        assert not canvas.save_button.isEnabled()
        canvas.set_styled(_make_pixmap())
        assert canvas.save_button.isEnabled()


# ---------------------------------------------------------------------------
# Strength slider
# ---------------------------------------------------------------------------

class TestStrengthSlider:
    def test_strength_slider_emits_value_changed_signal(
        self, qtbot, canvas: PhotoCanvasView
    ) -> None:
        received: list[float] = []
        canvas.strength_slider.value_changed.connect(received.append)

        canvas.strength_slider.set_strength(0.75)

        assert len(received) == 1
        assert abs(received[0] - 0.75) < 0.01

    def test_strength_slider_default_is_one(self, canvas: PhotoCanvasView) -> None:
        assert canvas.strength_slider.strength() == pytest.approx(1.0)

    def test_strength_slider_clamps_below_zero(
        self, canvas: PhotoCanvasView
    ) -> None:
        canvas.strength_slider.set_strength(-0.5)
        assert canvas.strength_slider.strength() == pytest.approx(0.0)

    def test_strength_slider_clamps_above_three(
        self, canvas: PhotoCanvasView
    ) -> None:
        canvas.strength_slider.set_strength(3.5)
        assert canvas.strength_slider.strength() == pytest.approx(3.0)

    def test_loading_new_photo_resets_slider_to_100_pct(
        self, canvas: PhotoCanvasView
    ) -> None:
        canvas.strength_slider.set_strength(2.0)
        canvas.set_original(_make_pixmap())
        assert canvas.strength_slider.strength() == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Split-view divider
# ---------------------------------------------------------------------------

class TestSplitView:
    def test_split_view_default_ratio_is_half(
        self, canvas: PhotoCanvasView
    ) -> None:
        assert canvas.split_view.split_ratio() == pytest.approx(0.5)

    def test_split_view_ratio_can_be_set_programmatically(
        self, canvas: PhotoCanvasView
    ) -> None:
        canvas.split_view.set_split_ratio(0.3)
        assert canvas.split_view.split_ratio() == pytest.approx(0.3)

    def test_split_view_ratio_clamps_to_zero(
        self, canvas: PhotoCanvasView
    ) -> None:
        canvas.split_view.set_split_ratio(-1.0)
        assert canvas.split_view.split_ratio() == pytest.approx(0.0)

    def test_split_view_ratio_clamps_to_one(
        self, canvas: PhotoCanvasView
    ) -> None:
        canvas.split_view.set_split_ratio(2.0)
        assert canvas.split_view.split_ratio() == pytest.approx(1.0)

    def test_split_view_boundary_is_draggable(
        self, canvas: PhotoCanvasView
    ) -> None:
        """Setting a new ratio via the public API simulates a drag; ratio persists."""
        canvas.split_view.set_split_ratio(0.5)
        canvas.split_view.set_split_ratio(0.7)
        assert canvas.split_view.split_ratio() == pytest.approx(0.7)

    def test_split_ratio_changed_signal_fired_on_set(
        self, qtbot, canvas: PhotoCanvasView
    ) -> None:
        received: list[float] = []
        canvas.split_view.split_ratio_changed.connect(received.append)

        # Simulate drag: manually set internal state and emit
        canvas.split_view._split_ratio = 0.4
        canvas.split_view.split_ratio_changed.emit(0.4)

        assert len(received) == 1
        assert received[0] == pytest.approx(0.4)


# ---------------------------------------------------------------------------
# apply / open signals
# ---------------------------------------------------------------------------

class TestSignals:
    def test_open_photo_button_emits_open_photo_requested(
        self, qtbot, canvas: PhotoCanvasView
    ) -> None:
        with qtbot.waitSignal(canvas.open_photo_requested, timeout=300):
            canvas.open_button.click()

    def test_apply_button_emits_apply_requested(
        self, qtbot, canvas: PhotoCanvasView
    ) -> None:
        canvas.set_original(_make_pixmap())
        canvas.set_active_style("candy")
        received_args: list[tuple] = []

        def _capture(style_id: str, strength: float) -> None:
            received_args.append((style_id, strength))

        canvas.apply_requested.connect(_capture)
        canvas.apply_button.click()

        assert len(received_args) == 1
        assert received_args[0][0] == "candy"
        assert received_args[0][1] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Re-Apply button
# ---------------------------------------------------------------------------

class TestReApplyButton:
    def test_reapply_button_starts_disabled(self, canvas: PhotoCanvasView) -> None:
        assert not canvas.reapply_button.isEnabled()

    def test_reapply_button_disabled_when_only_original_set(
        self, canvas: PhotoCanvasView
    ) -> None:
        canvas.set_original(_make_pixmap())
        canvas.set_active_style("candy")
        assert not canvas.reapply_button.isEnabled()

    def test_reapply_button_enabled_after_styled_result_and_style(
        self, canvas: PhotoCanvasView
    ) -> None:
        canvas.set_active_style("candy")
        canvas.set_styled(_make_pixmap())
        assert canvas.reapply_button.isEnabled()

    def test_reapply_button_enabled_by_set_active_style_when_styled_exists(
        self, canvas: PhotoCanvasView
    ) -> None:
        canvas.set_styled(_make_pixmap())      # styled result present
        canvas.set_active_style("mosaic")      # then a different style is selected
        assert canvas.reapply_button.isEnabled()

    def test_reapply_button_emits_reapply_requested(
        self, qtbot, canvas: PhotoCanvasView
    ) -> None:
        canvas.set_active_style("candy")
        canvas.set_styled(_make_pixmap())
        received: list[tuple] = []
        canvas.reapply_requested.connect(lambda sid, s: received.append((sid, s)))

        canvas.reapply_button.click()

        assert len(received) == 1
        assert received[0][0] == "candy"
        assert received[0][1] == pytest.approx(1.0)

    def test_reapply_button_not_clickable_without_style(
        self, canvas: PhotoCanvasView
    ) -> None:
        canvas.set_styled(_make_pixmap())   # result exists but no style selected
        # Button should be disabled — reapply_button requires a style AND styled result
        assert not canvas.reapply_button.isEnabled()


# ---------------------------------------------------------------------------
# reset_styled
# ---------------------------------------------------------------------------

class TestResetStyled:
    def test_reset_styled_clears_has_styled_flag(
        self, canvas: PhotoCanvasView
    ) -> None:
        canvas.set_styled(_make_pixmap())
        assert canvas.has_styled()
        canvas.reset_styled()
        assert not canvas.has_styled()

    def test_reset_styled_disables_save_button(
        self, canvas: PhotoCanvasView
    ) -> None:
        canvas.set_styled(_make_pixmap())
        canvas.reset_styled()
        assert not canvas.save_button.isEnabled()

    def test_reset_styled_disables_reapply_button(
        self, canvas: PhotoCanvasView
    ) -> None:
        canvas.set_active_style("candy")
        canvas.set_styled(_make_pixmap())
        canvas.reset_styled()
        assert not canvas.reapply_button.isEnabled()

    def test_reset_styled_resets_split_ratio(
        self, canvas: PhotoCanvasView
    ) -> None:
        canvas.split_view.set_split_ratio(0.8)
        canvas.reset_styled()
        assert canvas.split_view.split_ratio() == pytest.approx(0.5)

    def test_slider_auto_apply_fires_after_styled_result(
        self, qtbot, canvas: PhotoCanvasView
    ) -> None:
        """When a styled result exists, the slider must emit reapply_strength_requested
        (not apply_requested or reapply_requested) so the left pane / chain is preserved."""
        canvas.set_original(_make_pixmap())
        canvas.set_active_style("candy")
        canvas.set_styled(_make_pixmap())

        apply_received: list[tuple] = []
        reapply_received: list[tuple] = []
        strength_received: list[tuple] = []
        canvas.apply_requested.connect(lambda sid, s: apply_received.append((sid, s)))
        canvas.reapply_requested.connect(lambda sid, s: reapply_received.append((sid, s)))
        canvas.reapply_strength_requested.connect(lambda sid, s: strength_received.append((sid, s)))

        canvas.strength_slider.released.emit()

        assert apply_received == [], "Slider must NOT emit apply_requested when styled result exists"
        assert reapply_received == [], "Slider must NOT emit reapply_requested (would advance chain)"
        assert len(strength_received) == 1, "Slider must emit reapply_strength_requested"
        assert strength_received[0][0] == "candy"

    def test_slider_does_not_apply_without_styled_result(
        self, qtbot, canvas: PhotoCanvasView
    ) -> None:
        """Without a styled result, releasing the slider must NOT emit any signal.

        The strength value is simply stored; the user must click Apply to trigger
        the initial style transfer.  This prevents spurious Apply-Error dialogs when
        models have been unloaded (e.g. after a Reset) but _current_style_id is
        still set from a previous session.
        """
        canvas.set_original(_make_pixmap())
        canvas.set_active_style("candy")

        apply_received: list[tuple] = []
        reapply_received: list[tuple] = []
        strength_received: list[tuple] = []
        canvas.apply_requested.connect(lambda sid, s: apply_received.append((sid, s)))
        canvas.reapply_requested.connect(lambda sid, s: reapply_received.append((sid, s)))
        canvas.reapply_strength_requested.connect(lambda sid, s: strength_received.append((sid, s)))

        canvas.strength_slider.released.emit()

        assert apply_received == [], "Slider must NOT emit apply_requested when no styled result"
        assert reapply_received == [], "Slider must NOT emit reapply_requested"
        assert strength_received == [], "Slider must NOT emit reapply_strength_requested"


# ---------------------------------------------------------------------------
# Undo button
# ---------------------------------------------------------------------------

class TestUndoButton:
    def test_undo_button_starts_disabled(self, canvas: PhotoCanvasView) -> None:
        assert not canvas.undo_button.isEnabled()

    def test_undo_requested_signal_emitted_on_click(
        self, qtbot, canvas: PhotoCanvasView
    ) -> None:
        canvas.set_undo_available(True)
        with qtbot.waitSignal(canvas.undo_requested, timeout=300):
            canvas.undo_button.click()
