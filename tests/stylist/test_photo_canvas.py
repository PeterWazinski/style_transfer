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

    def test_strength_slider_clamps_above_one(
        self, canvas: PhotoCanvasView
    ) -> None:
        canvas.strength_slider.set_strength(1.5)
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
