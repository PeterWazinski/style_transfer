"""Unit tests for BuiltinChainModel and ChainStore."""
from __future__ import annotations

from pathlib import Path

import pytest

from src.core.chain_models import BuiltinChainModel, ChainStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_chain(chain_id: str = "pastel", name: str = "Pastel") -> BuiltinChainModel:
    return BuiltinChainModel(
        id=chain_id,
        name=name,
        chain_path=f"style_chains/{chain_id}/chain.yml",
        preview_path=f"style_chains/{chain_id}/preview.jpg",
        description="A test chain",
        step_count=2,
        tags=["warm", "soft"],
    )


# ---------------------------------------------------------------------------
# BuiltinChainModel
# ---------------------------------------------------------------------------

def test_chainmodel_to_dict_roundtrip() -> None:
    c = _make_chain("pastel")
    d = c.to_dict()
    c2 = BuiltinChainModel.from_dict(d)
    assert c2.id == c.id
    assert c2.name == c.name
    assert c2.chain_path == c.chain_path
    assert c2.tags == c.tags
    assert c2.step_count == c.step_count


def test_chainmodel_from_dict_ignores_unknown_keys() -> None:
    d = _make_chain("pastel").to_dict()
    d["future_field"] = "future_value"
    # should not raise
    BuiltinChainModel.from_dict(d)


def test_chainmodel_tags_default_empty() -> None:
    c = BuiltinChainModel(id="x", name="X", chain_path="style_chains/x/chain.yml")
    assert c.tags == []


def test_chainmodel_resolved_paths(tmp_path: Path) -> None:
    c = _make_chain("pastel")
    assert c.chain_path_resolved(tmp_path) == tmp_path / "style_chains/pastel/chain.yml"
    assert c.preview_path_resolved(tmp_path) == tmp_path / "style_chains/pastel/preview.jpg"


# ---------------------------------------------------------------------------
# ChainStore
# ---------------------------------------------------------------------------

def test_chainstore_empty_when_file_missing(tmp_path: Path) -> None:
    store = ChainStore(tmp_path / "nonexistent.json")
    assert store.load() == []


def test_chainstore_save_and_reload(tmp_path: Path) -> None:
    store = ChainStore(tmp_path / "catalog.json")
    chains = [_make_chain("pastel"), _make_chain("dense", "Dense")]
    store.save(chains)
    loaded = store.load()
    assert len(loaded) == 2
    assert loaded[0].id == "pastel"
    assert loaded[1].id == "dense"


def test_chainstore_save_and_reload_preserves_tags(tmp_path: Path) -> None:
    store = ChainStore(tmp_path / "catalog.json")
    c = BuiltinChainModel(
        id="pastel", name="Pastel", chain_path="style_chains/pastel/chain.yml",
        tags=["warm", "soft"],
    )
    store.save([c])
    loaded = store.load()
    assert loaded[0].tags == ["warm", "soft"]


def test_chainstore_creates_parent_dirs(tmp_path: Path) -> None:
    store = ChainStore(tmp_path / "sub" / "dir" / "catalog.json")
    store.save([_make_chain()])
    assert store.catalog_path.exists()


def test_chainstore_load_empty_catalog(tmp_path: Path) -> None:
    store = ChainStore(tmp_path / "catalog.json")
    store.save([])
    assert store.load() == []
