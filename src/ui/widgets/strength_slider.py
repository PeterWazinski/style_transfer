"""StrengthSlider — labelled horizontal slider (0 – 100 %) for style blend strength."""
from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QHBoxLayout, QLabel, QSlider, QWidget


class StrengthSlider(QWidget):
    """Composite widget: ``Strength:``  [━━━●━━━]  ``75 %``

    Emits :attr:`value_changed` (float in ``[0.0, 1.0]``) on every slider move.
    Emits :attr:`released` (no payload) when the user releases the slider handle.
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

        layout.addWidget(QLabel("Strength:", self))

        self._slider = QSlider(Qt.Horizontal, self)  # type: ignore[attr-defined]
        self._slider.setMinimum(0)
        self._slider.setMaximum(100)
        self._slider.setValue(100)
        self._slider.setTickInterval(10)
        self._slider.setMinimumWidth(120)
        layout.addWidget(self._slider)

        self._pct_label = QLabel("100 %", self)
        self._pct_label.setMinimumWidth(44)
        layout.addWidget(self._pct_label)

        self._slider.valueChanged.connect(self._on_value_changed)
        self._slider.sliderReleased.connect(self.released)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _on_value_changed(self, raw: int) -> None:
        self._pct_label.setText(f"{raw} %")
        self.value_changed.emit(raw / 100.0)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def strength(self) -> float:
        """Return the current strength as a float in ``[0.0, 1.0]``."""
        return self._slider.value() / 100.0

    def set_strength(self, value: float) -> None:
        """Set strength programmatically; clamps to ``[0.0, 1.0]``."""
        clamped = max(0, min(100, int(round(value * 100))))
        self._slider.setValue(clamped)
