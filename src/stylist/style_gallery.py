"""StyleGalleryView — QListView-based thumbnail browser for the style catalog.

The view is populated from a :class:`~src.core.registry.StyleRegistry` and
emits signals when the user interacts with it.  The gallery never mutates
the registry directly; it only fires signals that the owner (MainWindow)
connects to the appropriate service calls.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

from PySide6.QtCore import QModelIndex, Qt, Signal
from PySide6.QtGui import QPixmap, QStandardItem, QStandardItemModel
from PySide6.QtWidgets import (
    QAbstractItemView,
    QListView,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from src.core.models import StyleModel
from src.core.registry import StyleRegistry
from src.stylist.widgets.thumbnail_delegate import ThumbnailDelegate

logger: logging.Logger = logging.getLogger(__name__)

_PLACEHOLDER_SIZE: int = 128
_PROJECT_ROOT: Path = (
    Path(sys.executable).parent
    if getattr(sys, "frozen", False)
    else Path(__file__).resolve().parent.parent.parent
)


def _load_pixmap(path: Path | str | None) -> QPixmap:
    """Return a QPixmap from *path*, or a grey placeholder on failure.

    Relative paths are resolved against the project root so the gallery
    works regardless of the process working directory.
    """
    if path:
        resolved = Path(path) if isinstance(path, str) else path
        if not resolved.is_absolute():
            resolved = _PROJECT_ROOT / resolved
        if resolved.exists():
            pix = QPixmap(str(resolved))
            if not pix.isNull():
                return pix
    pix = QPixmap(_PLACEHOLDER_SIZE, _PLACEHOLDER_SIZE)
    pix.fill(Qt.gray)  # type: ignore[attr-defined]
    return pix


class StyleGalleryView(QWidget):
    """Visual browser for the style catalog.

    Signals:
        style_selected(StyleModel):       Emitted when the user clicks a thumbnail.
        style_apply_requested(StyleModel): Emitted on double-click; owner should apply.
        add_requested():                  "Add Style" button pressed.
        edit_requested(str):              Right-click → Edit; carries ``style_id``.
        delete_requested(str):            "Delete" button or right-click → Delete.
    """

    style_selected: Signal = Signal(object)        # payload: StyleModel
    style_apply_requested: Signal = Signal(object) # payload: StyleModel

    def __init__(
        self,
        registry: StyleRegistry,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._registry: StyleRegistry = registry
        self._item_model: QStandardItemModel = QStandardItemModel(self)
        self._build_ui()
        self.refresh()

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)

        # --- List view ---
        self._list_view = QListView(self)
        self._list_view.setModel(self._item_model)
        self._list_view.setItemDelegate(ThumbnailDelegate(self._list_view))
        self._list_view.setViewMode(QListView.IconMode)  # type: ignore[attr-defined]
        self._list_view.setResizeMode(QListView.Adjust)  # type: ignore[attr-defined]
        self._list_view.setMovement(QListView.Static)  # type: ignore[attr-defined]
        self._list_view.setSelectionMode(QAbstractItemView.SingleSelection)  # type: ignore[attr-defined]
        self._list_view.setSpacing(4)
        self._list_view.setContextMenuPolicy(Qt.CustomContextMenu)  # type: ignore[attr-defined]
        self._list_view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        root.addWidget(self._list_view)

        # --- Connections ---
        self._list_view.clicked.connect(self._on_item_clicked)
        self._list_view.doubleClicked.connect(self._on_item_double_clicked)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def refresh(self) -> None:
        """Reload all styles from the registry and repopulate the list."""
        self._item_model.clear()
        for style in self._registry.list_styles():
            pix = _load_pixmap(style.preview_path)
            item = QStandardItem(style.name)
            item.setData(pix, Qt.DecorationRole)  # type: ignore[attr-defined]
            item.setData(style, Qt.UserRole)  # type: ignore[attr-defined]
            item.setTextAlignment(Qt.AlignHCenter)  # type: ignore[attr-defined]
            item.setEditable(False)
            self._item_model.appendRow(item)
        logger.debug("Gallery refreshed: %d styles", self._item_model.rowCount())

    def model(self) -> QStandardItemModel:
        """Return the underlying ``QStandardItemModel`` (for tests)."""
        return self._item_model

    def current_style_id(self) -> str | None:
        """Return the selected style's ID, or *None* when nothing is selected."""
        indexes = self._list_view.selectedIndexes()
        if not indexes:
            return None
        style: StyleModel | None = indexes[0].data(Qt.UserRole)  # type: ignore[attr-defined]
        return style.id if style else None

    def current_style(self) -> StyleModel | None:
        """Return the selected :class:`StyleModel`, or *None*."""
        indexes = self._list_view.selectedIndexes()
        if not indexes:
            return None
        return indexes[0].data(Qt.UserRole)  # type: ignore[attr-defined]

    # ------------------------------------------------------------------
    # Internal slots
    # ------------------------------------------------------------------

    def _on_item_clicked(self, index: QModelIndex) -> None:
        style: StyleModel | None = index.data(Qt.UserRole)  # type: ignore[attr-defined]
        if style:
            self.style_selected.emit(style)

    def _on_item_double_clicked(self, index: QModelIndex) -> None:
        style: StyleModel | None = index.data(Qt.UserRole)  # type: ignore[attr-defined]
        if style:
            self.style_apply_requested.emit(style)
