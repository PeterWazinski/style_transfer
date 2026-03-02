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

from PySide6.QtCore import Qt, QRectF, Signal
from PySide6.QtGui import (
    QColor, QCursor, QPainter, QPen, QPixmap,
)
from PySide6.QtWidgets import (
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsView,
    QHBoxLayout,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from src.ui.widgets.strength_slider import StrengthSlider

logger: logging.Logger = logging.getLogger(__name__)


class PhotoSplitView(QGraphicsView):
    """QGraphicsView subclass that shows original (left) and styled (right) image.

    A draggable vertical white dashed line separates the two halves.

    Signals:
        split_ratio_changed(float): Emitted when the divider moves (0.0 – 1.0).
    """

    split_ratio_changed: Signal = Signal(float)

    _GRAB_TOLERANCE: int = 8  # px around the divider that counts as a grab

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._split_ratio: float = 0.5
        self._dragging: bool = False

        self.setMouseTracking(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)  # type: ignore[attr-defined]
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)  # type: ignore[attr-defined]
        self.setRenderHint(QPainter.SmoothPixmapTransform)  # type: ignore[attr-defined]

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def split_ratio(self) -> float:
        """Return current divider position in ``[0.0, 1.0]``."""
        return self._split_ratio

    def set_split_ratio(self, ratio: float) -> None:
        """Set divider position; clamps to ``[0.0, 1.0]``."""
        self._split_ratio = max(0.0, min(1.0, ratio))
        self.viewport().update()

    # ------------------------------------------------------------------
    # Mouse handling
    # ------------------------------------------------------------------

    def _divider_x(self) -> int:
        return int(self.viewport().width() * self._split_ratio)

    def mousePressEvent(self, event) -> None:  # type: ignore[override]
        if abs(event.position().x() - self._divider_x()) <= self._GRAB_TOLERANCE:
            self._dragging = True
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:  # type: ignore[override]
        w = self.viewport().width()
        near_divider = abs(event.position().x() - self._divider_x()) <= self._GRAB_TOLERANCE
        self.setCursor(QCursor(Qt.SplitHCursor if near_divider else Qt.ArrowCursor))  # type: ignore[attr-defined]
        if self._dragging and w > 0:
            self.set_split_ratio(event.position().x() / w)
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
        super().paintEvent(event)
        if not self.scene():
            return
        painter = QPainter(self.viewport())
        pen = QPen(QColor("#ffffff"), 2, Qt.DashLine)  # type: ignore[attr-defined]
        painter.setPen(pen)
        x = self._divider_x()
        painter.drawLine(x, 0, x, self.viewport().height())
        painter.end()


class PhotoCanvasView(QWidget):
    """Central photo + controls widget.

    Signals:
        open_photo_requested():           "Open Photo" button clicked.
        apply_requested(str, float):      "Apply" button clicked; ``(style_id, strength)``.
        save_requested():                 "Save Result" button clicked.
    """

    open_photo_requested: Signal = Signal()
    apply_requested: Signal = Signal(str, float)
    save_requested: Signal = Signal()

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

        # Scene + split view
        self._scene = QGraphicsScene(self)
        self.split_view = PhotoSplitView(self)
        self.split_view.setScene(self._scene)
        self.split_view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        root.addWidget(self.split_view)

        # Pixmap graphics items
        self._original_item = QGraphicsPixmapItem()
        self._styled_item = QGraphicsPixmapItem()
        self._scene.addItem(self._original_item)
        self._scene.addItem(self._styled_item)

        # Controls row
        ctrl = QHBoxLayout()
        self.strength_slider = StrengthSlider(self)
        ctrl.addWidget(self.strength_slider)
        ctrl.addStretch()

        self.open_button = QPushButton("Open Photo", self)
        self.apply_button = QPushButton("Apply", self)
        self.save_button = QPushButton("Save Result", self)
        self.apply_button.setEnabled(False)
        self.save_button.setEnabled(False)
        ctrl.addWidget(self.open_button)
        ctrl.addWidget(self.apply_button)
        ctrl.addWidget(self.save_button)
        root.addLayout(ctrl)

        # Connections
        self.open_button.clicked.connect(self.open_photo_requested)
        self.apply_button.clicked.connect(self._on_apply_clicked)
        self.save_button.clicked.connect(self.save_requested)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_original(self, pixmap: QPixmap) -> None:
        """Display *pixmap* as the original (left) layer."""
        self._original_item.setPixmap(pixmap)
        self._scene.setSceneRect(self._original_item.boundingRect())
        self.split_view.fitInView(self._scene.sceneRect(), Qt.KeepAspectRatio)  # type: ignore[attr-defined]
        self._has_original = True
        self.apply_button.setEnabled(self._current_style_id is not None)

    def set_styled(self, pixmap: QPixmap) -> None:
        """Display *pixmap* as the styled (right) layer."""
        self._styled_item.setPixmap(pixmap)
        self._has_styled = True
        self.save_button.setEnabled(True)

    def set_active_style(self, style_id: str) -> None:
        """Record which style is currently selected; enables Apply when a photo is loaded."""
        self._current_style_id = style_id
        self.apply_button.setEnabled(self._has_original)

    def has_original(self) -> bool:
        """Return *True* if an original photo has been set."""
        return self._has_original

    def has_styled(self) -> bool:
        """Return *True* if a styled result has been set."""
        return self._has_styled

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_apply_clicked(self) -> None:
        if self._current_style_id:
            self.apply_requested.emit(
                self._current_style_id,
                self.strength_slider.strength(),
            )
