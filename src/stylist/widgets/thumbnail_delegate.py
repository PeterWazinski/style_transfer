"""ThumbnailDelegate — custom QStyledItemDelegate for the style gallery.

Each gallery item is rendered as a square thumbnail card with the style
name centred below the image.
"""
from __future__ import annotations

from PySide6.QtCore import QModelIndex, QRect, QSize, Qt
from PySide6.QtGui import QColor, QFont, QIcon, QPainter, QPixmap
from PySide6.QtWidgets import QStyle, QStyledItemDelegate, QStyleOptionViewItem

THUMB_SIZE: int = 110
ITEM_WIDTH: int = 130
ITEM_HEIGHT: int = 148
PADDING: int = 6
FONT_SIZE: int = 9


class ThumbnailDelegate(QStyledItemDelegate):
    """Renders each gallery item as a fixed-size thumbnail card with a name label."""

    def paint(
        self,
        painter: QPainter,
        option: QStyleOptionViewItem,
        index: QModelIndex,
    ) -> None:
        painter.save()
        rect = option.rect

        # Background
        if option.state & QStyle.State_Selected:  # type: ignore[attr-defined]
            painter.fillRect(rect, option.palette.highlight())
        else:
            painter.fillRect(rect, QColor("#3c3f41"))

        # Thumbnail — DecorationRole can return QPixmap or QIcon depending on how
        # the item was constructed; normalise to QPixmap.
        raw = index.data(Qt.DecorationRole)  # type: ignore[attr-defined]
        if isinstance(raw, QIcon):
            pixmap: QPixmap | None = raw.pixmap(THUMB_SIZE, THUMB_SIZE)
        else:
            pixmap = raw  # already QPixmap or None
        if pixmap and not pixmap.isNull():
            scaled = pixmap.scaled(
                THUMB_SIZE, THUMB_SIZE,
                Qt.KeepAspectRatio,  # type: ignore[attr-defined]
                Qt.SmoothTransformation,  # type: ignore[attr-defined]
            )
            px = rect.x() + (ITEM_WIDTH - scaled.width()) // 2
            py = rect.y() + PADDING
            painter.drawPixmap(px, py, scaled)

        # Name label
        name: str | None = index.data(Qt.DisplayRole)  # type: ignore[attr-defined]
        if name:
            if option.state & QStyle.State_Selected:  # type: ignore[attr-defined]
                painter.setPen(option.palette.highlightedText().color())
            else:
                painter.setPen(QColor("#dddddd"))
            font = QFont()
            font.setPointSize(FONT_SIZE)
            painter.setFont(font)
            text_top = rect.y() + THUMB_SIZE + PADDING * 2
            text_rect = QRect(rect.x(), text_top, ITEM_WIDTH, ITEM_HEIGHT - THUMB_SIZE - PADDING * 2)
            painter.drawText(
                text_rect,
                Qt.AlignHCenter | Qt.AlignTop | Qt.TextWordWrap,  # type: ignore[attr-defined]
                name,
            )

        painter.restore()

    def sizeHint(self, option: QStyleOptionViewItem, index: QModelIndex) -> QSize:
        return QSize(ITEM_WIDTH, ITEM_HEIGHT)
