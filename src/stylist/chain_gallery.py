"""ChainGalleryView — QListView-based thumbnail browser for built-in style chains.

The view is populated from a :class:`~src.core.chain_registry.BuiltinChainRegistry`
and emits signals when the user interacts with it.  Invalid chains (whose
YAML references missing styles) are rendered with a grey overlay and a ⚠
badge via :class:`~src.stylist.widgets.thumbnail_delegate.ThumbnailDelegate`.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

from PySide6.QtCore import QModelIndex, QPoint, Qt, Signal
from PySide6.QtGui import QPixmap, QStandardItem, QStandardItemModel
from PySide6.QtWidgets import (
    QAbstractItemView,
    QListView,
    QMenu,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from src.core.chain_models import BuiltinChainModel
from src.core.chain_registry import BuiltinChainRegistry
from src.stylist.widgets.thumbnail_delegate import INVALID_ROLE, ThumbnailDelegate

logger: logging.Logger = logging.getLogger(__name__)

_PLACEHOLDER_SIZE: int = 128
_PROJECT_ROOT: Path = (
    Path(sys.executable).parent
    if getattr(sys, "frozen", False)
    else Path(__file__).resolve().parent.parent.parent
)


def _load_pixmap(path: Path | str | None) -> QPixmap:
    """Return a QPixmap from *path*, or a grey placeholder on failure."""
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


def _format_tooltip(chain: BuiltinChainModel, root: Path) -> str:
    """Build a tooltip string from the chain's YAML step list.

    Loads the YAML lazily (on first call per chain).  Returns the
    ``description`` field as fallback if the YAML is unavailable.
    """
    try:
        from src.core.style_chain_schema import load_style_chain
        sc = load_style_chain(chain.chain_path_resolved(root))
        lines = [f"{s.style}  {s.strength}%" for s in sc.steps]
        header = chain.description or chain.name
        return f"{header}\n\n" + "\n".join(lines)
    except Exception:  # noqa: BLE001
        return chain.description or chain.name


class ChainGalleryView(QWidget):
    """Visual browser for built-in style chains.

    Signals:
        chain_selected(BuiltinChainModel):        Emitted when the user clicks a thumbnail.
        chain_apply_requested(BuiltinChainModel): Emitted on double-click or context *Apply*.
        chain_append_requested(BuiltinChainModel): Emitted on context *Append*.
    """

    chain_selected: Signal = Signal(object)          # payload: BuiltinChainModel
    chain_apply_requested: Signal = Signal(object)   # payload: BuiltinChainModel
    chain_append_requested: Signal = Signal(object)  # payload: BuiltinChainModel

    def __init__(
        self,
        registry: BuiltinChainRegistry,
        invalid_chain_ids: set[str] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._registry: BuiltinChainRegistry = registry
        self._invalid_ids: set[str] = invalid_chain_ids or set()
        self._item_model: QStandardItemModel = QStandardItemModel(self)
        self._build_ui()
        self.refresh()

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)

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

        self._list_view.clicked.connect(self._on_item_clicked)
        self._list_view.doubleClicked.connect(self._on_item_double_clicked)
        self._list_view.customContextMenuRequested.connect(self._on_context_menu_requested)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def refresh(self) -> None:
        """Reload all chains from the registry and repopulate the list."""
        self._item_model.clear()
        for chain in self._registry.list_chains():
            pix = _load_pixmap(chain.preview_path)
            item = QStandardItem(chain.name)
            item.setData(pix, Qt.DecorationRole)  # type: ignore[attr-defined]
            item.setData(chain, Qt.UserRole)  # type: ignore[attr-defined]
            item.setTextAlignment(Qt.AlignHCenter)  # type: ignore[attr-defined]
            item.setEditable(False)
            # Tooltip: lazy-load step list from YAML
            item.setToolTip(_format_tooltip(chain, _PROJECT_ROOT))
            # Invalid badge
            if chain.id in self._invalid_ids:
                item.setData(True, INVALID_ROLE)
            self._item_model.appendRow(item)
        logger.debug("Chain gallery refreshed: %d chains", self._item_model.rowCount())

    def set_invalid_ids(self, invalid_ids: set[str]) -> None:
        """Update the set of invalid chain IDs and refresh the display."""
        self._invalid_ids = invalid_ids
        self.refresh()

    def model(self) -> QStandardItemModel:
        """Return the underlying ``QStandardItemModel`` (for tests)."""
        return self._item_model

    def current_chain(self) -> BuiltinChainModel | None:
        """Return the selected :class:`BuiltinChainModel`, or *None*."""
        indexes = self._list_view.selectedIndexes()
        if not indexes:
            return None
        return indexes[0].data(Qt.UserRole)  # type: ignore[attr-defined]

    # ------------------------------------------------------------------
    # Internal slots
    # ------------------------------------------------------------------

    def _on_item_clicked(self, index: QModelIndex) -> None:
        chain: BuiltinChainModel | None = index.data(Qt.UserRole)  # type: ignore[attr-defined]
        if chain:
            self.chain_selected.emit(chain)

    def _on_item_double_clicked(self, index: QModelIndex) -> None:
        chain: BuiltinChainModel | None = index.data(Qt.UserRole)  # type: ignore[attr-defined]
        if chain:
            self.chain_apply_requested.emit(chain)

    def _on_context_menu_requested(self, pos: QPoint) -> None:
        index = self._list_view.indexAt(pos)
        if not index.isValid():
            return
        chain: BuiltinChainModel | None = index.data(Qt.UserRole)  # type: ignore[attr-defined]
        if not chain:
            return
        menu = QMenu(self)
        apply_action = menu.addAction("Apply")
        append_action = menu.addAction("Append")
        action = menu.exec(self._list_view.viewport().mapToGlobal(pos))
        if action is apply_action:
            self.chain_apply_requested.emit(chain)
        elif action is append_action:
            self.chain_append_requested.emit(chain)
