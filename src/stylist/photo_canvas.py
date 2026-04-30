"""PhotoCanvasView — central widget for the photo viewer with before/after split.

Layout
------
::

    ┌─────────────────────────────────────────────────────┐
    │  PhotoSplitView  (QGraphicsView with draggable bar) │
    ├─────────────────────────────────────────────────────┤
    │  Strength: ──●──  [Open Photo]  [Apply]  [Save]     │
    └─────────────────────────────────────────────────────┘

The split divider can be dragged left/right to compare the original photo
with the styled result.  Use :meth:`set_original` / :meth:`set_styled` to
update the images, and connect to the emitted signals for actions.
"""
from __future__ import annotations

import logging
from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import (
    QColor, QCursor, QPainter, QPen, QPixmap,
)
from PySide6.QtWidgets import (
    QHBoxLayout,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from src.stylist.widgets.strength_slider import StrengthSlider

logger: logging.Logger = logging.getLogger(__name__)


class PhotoSplitView(QWidget):
    """Widget that composites original (left) and styled (right) images.

    The styled image is clipped to the right half of the widget so that
    dragging the divider reveals more or less of each version — a classic
    before/after comparison view.

    Signals:
        split_ratio_changed(float): Emitted when the divider moves (0.0 – 1.0).
    """

    split_ratio_changed: Signal = Signal(float)

    _GRAB_TOLERANCE: int = 8  # px around the divider that counts as a grab
    _BG_COLOR: QColor = QColor("#1e1e1e")

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._split_ratio: float = 0.5
        self._dragging: bool = False
        self._original_pixmap: QPixmap = QPixmap()
        self._styled_pixmap: QPixmap = QPixmap()
        self.setMouseTracking(True)

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def split_ratio(self) -> float:
        """Return current divider position in ``[0.0, 1.0]``."""
        return self._split_ratio

    def set_split_ratio(self, ratio: float) -> None:
        """Set divider position; clamps to ``[0.0, 1.0]``."""
        self._split_ratio = max(0.0, min(1.0, ratio))
        self.update()

    def set_original_pixmap(self, pixmap: QPixmap) -> None:
        """Set the original (left) image."""
        self._original_pixmap = pixmap
        self.update()

    def set_styled_pixmap(self, pixmap: QPixmap) -> None:
        """Set the styled (right) image."""
        self._styled_pixmap = pixmap
        self.update()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _divider_x(self) -> int:
        return int(self.width() * self._split_ratio)

    def _draw_pixmap_scaled(self, painter: QPainter, pixmap: QPixmap) -> None:
        """Draw *pixmap* scaled-to-fit and centred inside the widget."""
        scaled = pixmap.scaled(
            self.width(), self.height(),
            Qt.KeepAspectRatio,  # type: ignore[attr-defined]
            Qt.SmoothTransformation,  # type: ignore[attr-defined]
        )
        ox = (self.width() - scaled.width()) // 2
        oy = (self.height() - scaled.height()) // 2
        painter.drawPixmap(ox, oy, scaled)

    # ------------------------------------------------------------------
    # Mouse handling
    # ------------------------------------------------------------------

    def mousePressEvent(self, event) -> None:  # type: ignore[override]
        if abs(event.position().x() - self._divider_x()) <= self._GRAB_TOLERANCE:
            self._dragging = True
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:  # type: ignore[override]
        near_divider = abs(event.position().x() - self._divider_x()) <= self._GRAB_TOLERANCE
        self.setCursor(QCursor(Qt.SplitHCursor if near_divider else Qt.ArrowCursor))  # type: ignore[attr-defined]
        if self._dragging and self.width() > 0:
            self.set_split_ratio(event.position().x() / self.width())
            self.split_ratio_changed.emit(self._split_ratio)
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # type: ignore[override]
        self._dragging = False
        super().mouseReleaseEvent(event)

    # ------------------------------------------------------------------
    # Painting
    # ------------------------------------------------------------------

    def paintEvent(self, event) -> None:  # type: ignore[override]
        painter = QPainter(self)
        painter.setRenderHint(QPainter.SmoothPixmapTransform)  # type: ignore[attr-defined]
        painter.fillRect(self.rect(), self._BG_COLOR)

        # Draw original image across full width
        if not self._original_pixmap.isNull():
            self._draw_pixmap_scaled(painter, self._original_pixmap)

        # Draw styled image clipped to the right of the divider
        if not self._styled_pixmap.isNull():
            x = self._divider_x()
            painter.save()
            painter.setClipRect(x, 0, self.width() - x, self.height())
            self._draw_pixmap_scaled(painter, self._styled_pixmap)
            painter.restore()

        # Draw divider line
        pen = QPen(QColor("#ffffff"), 2, Qt.DashLine)  # type: ignore[attr-defined]
        painter.setPen(pen)
        x = self._divider_x()
        painter.drawLine(x, 0, x, self.height())
        painter.end()


class PhotoCanvasView(QWidget):
    """Central photo + controls widget.

    Signals:
        open_photo_requested():           "Open Photo" button clicked.
        apply_requested(str, float):      "Apply" button clicked; ``(style_id, strength)``.
        save_requested():                 "Save Result" button clicked.
    """

    open_photo_requested: Signal = Signal()
    reset_requested: Signal = Signal()
    apply_requested: Signal = Signal(str, float)
    reapply_requested: Signal = Signal(str, float)
    reapply_strength_requested: Signal = Signal(str, float)  # slider: re-run same step, no chain advance
    save_requested: Signal = Signal()
    undo_requested: Signal = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._current_style_id: str | None = None
        self._has_original: bool = False
        self._has_styled: bool = False
        self._build_ui()

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)

        self.split_view = PhotoSplitView(self)
        self.split_view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)  # type: ignore[attr-defined]
        root.addWidget(self.split_view)

        # Controls row
        ctrl = QHBoxLayout()
        self.strength_slider = StrengthSlider(self)
        ctrl.addWidget(self.strength_slider)
        ctrl.addStretch()

        # Left group: 📂 | ↺ | 💾
        self.open_button = QPushButton("\U0001f4c2", self)
        self.open_button.setToolTip("Open Photo (Ctrl+O)")
        self.reset_button = QPushButton("\u21ba", self)
        self.save_button = QPushButton("\U0001f4be", self)
        self.reset_button.setEnabled(False)
        self.reset_button.setToolTip("Reset \u2014 discard all style filters")
        self.save_button.setEnabled(False)
        self.save_button.setToolTip("Save Result (Ctrl+S)")
        ctrl.addWidget(self.open_button)
        ctrl.addWidget(self.reset_button)
        ctrl.addWidget(self.save_button)

        ctrl.addSpacing(24)  # visual gap between groups

        # Right group: ▶ | ⏩ | ↩
        self.apply_button = QPushButton("\u25b6", self)
        self.apply_button.setToolTip("Apply selected style to photo")
        self.reapply_button = QPushButton("\u23e9", self)
        self.reapply_button.setToolTip("Stack style on top of current result")
        self.undo_button = QPushButton("\u21a9", self)
        self.undo_button.setToolTip("Undo last apply step")
        self.apply_button.setEnabled(False)
        self.reapply_button.setEnabled(False)
        self.undo_button.setEnabled(False)
        ctrl.addWidget(self.apply_button)
        ctrl.addWidget(self.reapply_button)
        ctrl.addWidget(self.undo_button)
        root.addLayout(ctrl)

        # Connections
        self.open_button.clicked.connect(self.open_photo_requested)
        self.reset_button.clicked.connect(self.reset_requested)
        self.apply_button.clicked.connect(self._on_apply_clicked)
        self.reapply_button.clicked.connect(self._on_reapply_clicked)
        self.undo_button.clicked.connect(self.undo_requested)
        self.save_button.clicked.connect(self.save_requested)
        self.strength_slider.released.connect(self._on_strength_released)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_original(self, pixmap: QPixmap) -> None:
        """Display *pixmap* as the original (left) layer."""
        self.split_view.set_original_pixmap(pixmap)
        self._has_original = True
        self.strength_slider.set_strength(1.0)
        self.reset_button.setEnabled(True)
        self.apply_button.setEnabled(self._current_style_id is not None)

    def set_styled(self, pixmap: QPixmap) -> None:
        """Display *pixmap* as the styled (right) layer."""
        self.split_view.set_styled_pixmap(pixmap)
        self._has_styled = True
        self.save_button.setEnabled(True)
        self.reapply_button.setEnabled(self._current_style_id is not None)

    def reset_styled(self) -> None:
        """Clear the styled result and reset the canvas to show only the original."""
        self.split_view.set_styled_pixmap(QPixmap())
        self._has_styled = False
        self.reapply_button.setEnabled(False)
        self.save_button.setEnabled(False)
        self.split_view.set_split_ratio(0.5)

    def reset_all(self) -> None:
        """Full reset: clear both panes and disable all action buttons."""
        self.split_view.set_original_pixmap(QPixmap())
        self.split_view.set_styled_pixmap(QPixmap())
        self._has_original = False
        self._has_styled = False
        self.apply_button.setEnabled(False)
        self.reapply_button.setEnabled(False)
        self.save_button.setEnabled(False)
        self.reset_button.setEnabled(False)
        self.split_view.set_split_ratio(0.5)

    def set_active_style(self, style_id: str) -> None:
        """Record which style is currently selected; enables Apply when a photo is loaded."""
        self._current_style_id = style_id
        self.apply_button.setEnabled(self._has_original)
        self.reapply_button.setEnabled(self._has_styled)

    def has_original(self) -> bool:
        """Return *True* if an original photo has been set."""
        return self._has_original

    def has_styled(self) -> bool:
        """Return *True* if a styled result has been set."""
        return self._has_styled

    def set_undo_available(self, available: bool) -> None:
        """Enable or disable the Undo button."""
        self.undo_button.setEnabled(available)

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_apply_clicked(self) -> None:
        if self._current_style_id:
            self.apply_requested.emit(
                self._current_style_id,
                self.strength_slider.strength(),
            )

    def _on_reapply_clicked(self) -> None:
        if self._current_style_id and self._has_styled:
            self.reapply_requested.emit(
                self._current_style_id,
                self.strength_slider.strength(),
            )

    def _on_strength_released(self) -> None:
        """Re-apply the current style with the new strength when the slider is released.

        Only fires when a styled result already exists (``_has_styled`` is True).
        Emits ``reapply_strength_requested`` so MainWindow re-runs the *same input
        step* with the new strength without advancing the chain (left pane is
        preserved).  If no styled result exists yet the slider value is simply
        stored for use when the user clicks Apply.
        """
        if not self._has_original or not self._current_style_id:
            return
        if self._has_styled:
            self.reapply_strength_requested.emit(
                self._current_style_id,
                self.strength_slider.strength(),
            )
