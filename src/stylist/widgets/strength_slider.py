"""StrengthSlider — labelled horizontal slider (0–300%) with tick-mark labels."""
from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QFont, QPainter
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QSlider,
    QStyle,
    QStyleOptionSlider,
    QVBoxLayout,
    QWidget,
)

# Tick positions drawn below the slider track (% values)
_TICKS: list[int] = [0, 50, 100, 150, 200, 250, 300]
_LABEL_TICKS: set[int] = {0, 100, 200, 300}   # labelled; others show tick only
_NATURAL: int = 100  # "natural" reference — full model output, no extrapolation


class _TickLabels(QWidget):
    """Paints percentage labels aligned to the groove of a companion QSlider."""

    def __init__(self, slider: QSlider, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._slider = slider
        self.setFixedHeight(16)

    def paintEvent(self, _event) -> None:  # noqa: N802
        opt = QStyleOptionSlider()
        self._slider.initStyleOption(opt)
        groove = self._slider.style().subControlRect(
            QStyle.ComplexControl.CC_Slider,  # type: ignore[attr-defined]
            opt,
            QStyle.SubControl.SC_SliderGroove,  # type: ignore[attr-defined]
            self._slider,
        )

        groove_left = groove.left()
        groove_width = groove.width()
        value_range = self._slider.maximum() - self._slider.minimum()

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing)  # type: ignore[attr-defined]

        base_font = painter.font()
        tick_font = QFont(base_font)
        tick_font.setPointSize(max(6, base_font.pointSize() - 2))
        bold_font = QFont(tick_font)
        bold_font.setBold(True)

        for tick in _TICKS:
            ratio = (tick - self._slider.minimum()) / value_range
            x = groove_left + int(ratio * groove_width)

            if tick not in _LABEL_TICKS:
                # Draw a short tick mark only — no label
                painter.setPen(self.palette().windowText().color())
                painter.setFont(tick_font)
                fm = painter.fontMetrics()
                painter.drawText(x, fm.ascent(), "·")
                continue

            if tick == _NATURAL:
                painter.setFont(bold_font)
                painter.setPen(QColor("#b06000"))  # amber — highlights natural point
            else:
                painter.setFont(tick_font)
                painter.setPen(self.palette().windowText().color())

            label = f"{tick}%"
            fm = painter.fontMetrics()
            text_w = fm.horizontalAdvance(label)
            # Clamp so edge labels (0% / 300%) never clip outside the widget
            draw_x = max(0, min(x - text_w // 2, self.width() - text_w))
            painter.drawText(draw_x, fm.ascent(), label)

        painter.end()


class StrengthSlider(QWidget):
    """Composite widget: ``Strength:``  [━━━●━━━]  ``150 %``

    Tick-mark labels are drawn below the slider track at 0 / 50 / 100 / 150 / 200 /
    250 / 300 %.  The **100 %** mark is highlighted in amber as the "natural" reference
    (model output at full strength, no extrapolation).

    Emits :attr:`value_changed` (float in ``[0.0, 3.0]``) on every slider move.
    Emits :attr:`released` (no payload) when the user releases the slider handle.
    Values above 100 % extrapolate the style effect beyond the model's native output.
    """

    value_changed: Signal = Signal(float)
    released: Signal = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._build_ui()

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setAlignment(Qt.AlignVCenter)  # type: ignore[attr-defined]

        layout.addWidget(QLabel("Strength:", self))

        # ── slider + tick-labels stacked vertically ───────────────────
        slider_container = QWidget(self)
        vbox = QVBoxLayout(slider_container)
        vbox.setContentsMargins(0, 0, 0, 0)
        vbox.setSpacing(0)

        self._slider = QSlider(Qt.Horizontal, slider_container)  # type: ignore[attr-defined]
        self._slider.setMinimum(0)
        self._slider.setMaximum(300)
        self._slider.setValue(100)
        self._slider.setTickPosition(QSlider.TicksBelow)  # type: ignore[attr-defined]
        self._slider.setTickInterval(50)
        self._slider.setMinimumWidth(180)
        vbox.addWidget(self._slider)

        self._tick_labels = _TickLabels(self._slider, slider_container)
        vbox.addWidget(self._tick_labels)

        layout.addWidget(slider_container)

        self._pct_label = QLabel("100 %", self)
        self._pct_label.setMinimumWidth(52)
        layout.addWidget(self._pct_label)

        self._slider.valueChanged.connect(self._on_value_changed)
        self._slider.sliderReleased.connect(self.released)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _on_value_changed(self, raw: int) -> None:
        self._pct_label.setText(f"{raw} %")
        self.value_changed.emit(raw / 100.0)  # [0.0, 3.0]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def strength(self) -> float:
        """Return the current strength as a float in ``[0.0, 3.0]``."""
        return self._slider.value() / 100.0

    def set_strength(self, value: float) -> None:
        """Set strength programmatically; clamps to ``[0.0, 3.0]``."""
        clamped = max(0, min(300, int(round(value * 100))))
        self._slider.setValue(clamped)
