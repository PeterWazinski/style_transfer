"""Tests for ChainGalleryView."""
from __future__ import annotations

from pathlib import Path

import pytest
from PySide6.QtCore import Qt

from src.core.chain_models import BuiltinChainModel, ChainStore
from src.core.chain_registry import BuiltinChainRegistry
from src.stylist.chain_gallery import ChainGalleryView
from src.stylist.widgets.thumbnail_delegate import INVALID_ROLE


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_chain(chain_id: str = "pastel", name: str = "Pastel") -> BuiltinChainModel:
    return BuiltinChainModel(
        id=chain_id,
        name=name,
        chain_path=f"style_chains/{chain_id}/chain.yml",
        preview_path="",
        description="A test chain",
        step_count=2,
    )


def _make_registry(tmp_path: Path, chains: list[BuiltinChainModel]) -> BuiltinChainRegistry:
    catalog = tmp_path / "catalog.json"
    ChainStore(catalog).save(chains)
    return BuiltinChainRegistry(catalog_path=catalog)


# ---------------------------------------------------------------------------
# Population
# ---------------------------------------------------------------------------

class TestPopulation:
    def test_empty_registry_shows_no_items(self, qtbot, tmp_path: Path) -> None:
        reg = _make_registry(tmp_path, [])
        view = ChainGalleryView(registry=reg)
        qtbot.addWidget(view)
        assert view.model().rowCount() == 0

    def test_items_populated_from_registry(self, qtbot, tmp_path: Path) -> None:
        reg = _make_registry(tmp_path, [_make_chain("pastel"), _make_chain("dense", "Dense")])
        view = ChainGalleryView(registry=reg)
        qtbot.addWidget(view)
        assert view.model().rowCount() == 2

    def test_items_sorted_by_name(self, qtbot, tmp_path: Path) -> None:
        reg = _make_registry(tmp_path, [_make_chain("z", "Zebra"), _make_chain("a", "Alpha")])
        view = ChainGalleryView(registry=reg)
        qtbot.addWidget(view)
        names = [view.model().item(i).text() for i in range(view.model().rowCount())]
        assert names == ["Alpha", "Zebra"]


# ---------------------------------------------------------------------------
# Invalid chain badge
# ---------------------------------------------------------------------------

class TestInvalidBadge:
    def test_invalid_chain_has_invalid_role_set(self, qtbot, tmp_path: Path) -> None:
        reg = _make_registry(tmp_path, [_make_chain("pastel"), _make_chain("dense", "Dense")])
        view = ChainGalleryView(registry=reg, invalid_chain_ids={"pastel"})
        qtbot.addWidget(view)
        # Find the "Pastel" item
        model = view.model()
        items = {model.item(i).text(): model.item(i) for i in range(model.rowCount())}
        assert items["Pastel"].data(INVALID_ROLE) is True
        assert not items["Dense"].data(INVALID_ROLE)

    def test_valid_chain_has_no_invalid_role(self, qtbot, tmp_path: Path) -> None:
        reg = _make_registry(tmp_path, [_make_chain("pastel")])
        view = ChainGalleryView(registry=reg, invalid_chain_ids=set())
        qtbot.addWidget(view)
        item = view.model().item(0)
        assert not item.data(INVALID_ROLE)

    def test_set_invalid_ids_refreshes_view(self, qtbot, tmp_path: Path) -> None:
        reg = _make_registry(tmp_path, [_make_chain("pastel")])
        view = ChainGalleryView(registry=reg, invalid_chain_ids=set())
        qtbot.addWidget(view)
        assert not view.model().item(0).data(INVALID_ROLE)

        view.set_invalid_ids({"pastel"})
        assert view.model().item(0).data(INVALID_ROLE) is True


# ---------------------------------------------------------------------------
# Signals — click / double-click
# ---------------------------------------------------------------------------

class TestSignals:
    def test_click_emits_chain_selected(self, qtbot, tmp_path: Path) -> None:
        reg = _make_registry(tmp_path, [_make_chain("pastel")])
        view = ChainGalleryView(registry=reg)
        qtbot.addWidget(view)
        received: list = []
        view.chain_selected.connect(received.append)
        view._list_view.clicked.emit(view.model().index(0, 0))
        assert len(received) == 1
        assert received[0].id == "pastel"

    def test_double_click_emits_chain_apply_requested(self, qtbot, tmp_path: Path) -> None:
        reg = _make_registry(tmp_path, [_make_chain("pastel")])
        view = ChainGalleryView(registry=reg)
        qtbot.addWidget(view)
        received: list = []
        view.chain_apply_requested.connect(received.append)
        view._list_view.doubleClicked.emit(view.model().index(0, 0))
        assert len(received) == 1
        assert received[0].id == "pastel"


# ---------------------------------------------------------------------------
# Context menu
# ---------------------------------------------------------------------------

class TestContextMenu:
    def test_context_menu_apply_emits_chain_apply_requested(
        self, qtbot, tmp_path: Path
    ) -> None:
        import unittest.mock as mock

        reg = _make_registry(tmp_path, [_make_chain("pastel")])
        view = ChainGalleryView(registry=reg)
        qtbot.addWidget(view)
        received: list = []
        view.chain_apply_requested.connect(received.append)

        first_index = view.model().index(0, 0)
        pos = view._list_view.visualRect(first_index).center()
        with mock.patch("src.stylist.chain_gallery.QMenu") as MockMenu:
            instance = MockMenu.return_value
            apply_action = object()
            append_action = object()
            instance.addAction.side_effect = [apply_action, append_action]
            instance.exec.return_value = apply_action
            view._on_context_menu_requested(pos)

        assert len(received) == 1
        assert received[0].id == "pastel"

    def test_context_menu_append_emits_chain_append_requested(
        self, qtbot, tmp_path: Path
    ) -> None:
        import unittest.mock as mock

        reg = _make_registry(tmp_path, [_make_chain("pastel")])
        view = ChainGalleryView(registry=reg)
        qtbot.addWidget(view)
        received: list = []
        view.chain_append_requested.connect(received.append)

        first_index = view.model().index(0, 0)
        pos = view._list_view.visualRect(first_index).center()
        with mock.patch("src.stylist.chain_gallery.QMenu") as MockMenu:
            instance = MockMenu.return_value
            apply_action = object()
            append_action = object()
            instance.addAction.side_effect = [apply_action, append_action]
            instance.exec.return_value = append_action
            view._on_context_menu_requested(pos)

        assert len(received) == 1
        assert received[0].id == "pastel"
