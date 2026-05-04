"""Unit tests for StyleModel, StyleStore, and StyleRegistry."""
from __future__ import annotations

from pathlib import Path

import pytest

from src.core.models import StyleModel, StyleStore
from src.core.registry import DuplicateStyleError, StyleNotFoundError, StyleRegistry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_style(style_id: str = "candy", name: str = "Candy") -> StyleModel:
    return StyleModel(
        id=style_id,
        name=name,
        model_path=f"styles/{style_id}/model.onnx",
        preview_path=f"styles/{style_id}/preview.jpg",
        description="Test style",
        author="test",
        is_builtin=True,
    )


# ---------------------------------------------------------------------------
# StyleModel — unit
# ---------------------------------------------------------------------------

def test_stylemodel_to_dict_roundtrip() -> None:
    s = _make_style("candy")
    d = s.to_dict()
    s2 = StyleModel.from_dict(d)
    assert s2.id == s.id
    assert s2.name == s.name
    assert s2.model_path == s.model_path


def test_stylemodel_from_dict_ignores_unknown_keys() -> None:
    d = _make_style("candy").to_dict()
    d["future_field"] = "future_value"
    # should not raise
    StyleModel.from_dict(d)


def test_stylemodel_tags_roundtrip() -> None:
    s = StyleModel(id="candy", name="Candy", model_path="styles/candy/model.onnx", tags=["warm", "soft"])
    d = s.to_dict()
    s2 = StyleModel.from_dict(d)
    assert s2.tags == ["warm", "soft"]


def test_stylemodel_tags_default_empty() -> None:
    s = _make_style("candy")
    assert s.tags == []


def test_stylemodel_resolved_paths(tmp_path: Path) -> None:
    s = _make_style("candy")
    assert s.model_path_resolved(tmp_path) == tmp_path / "styles/candy/model.onnx"
    assert s.preview_path_resolved(tmp_path) == tmp_path / "styles/candy/preview.jpg"


# ---------------------------------------------------------------------------
# StyleStore — unit
# ---------------------------------------------------------------------------

def test_stylestore_empty_when_file_missing(tmp_path: Path) -> None:
    store = StyleStore(tmp_path / "nonexistent.json")
    assert store.load() == []


def test_stylestore_save_and_reload(tmp_path: Path) -> None:
    store = StyleStore(tmp_path / "catalog.json")
    styles = [_make_style("candy"), _make_style("mosaic", "Mosaic")]
    store.save(styles)
    loaded = store.load()
    assert len(loaded) == 2
    assert loaded[0].id == "candy"
    assert loaded[1].id == "mosaic"


def test_stylestore_creates_parent_dirs(tmp_path: Path) -> None:
    store = StyleStore(tmp_path / "sub" / "dir" / "catalog.json")
    store.save([_make_style()])
    assert store.catalog_path.exists()


# ---------------------------------------------------------------------------
# StyleRegistry — list / get
# ---------------------------------------------------------------------------

def test_registry_empty_on_new_catalog(tmp_path: Path) -> None:
    r = StyleRegistry(tmp_path / "catalog.json")
    assert r.list_styles() == []


def test_registry_contains_after_add(tmp_path: Path) -> None:
    r = StyleRegistry(tmp_path / "catalog.json")
    r.add(_make_style("candy"))
    assert "candy" in r


def test_registry_not_contains_before_add(tmp_path: Path) -> None:
    r = StyleRegistry(tmp_path / "catalog.json")
    assert "candy" not in r


def test_registry_get_returns_correct_style(tmp_path: Path) -> None:
    r = StyleRegistry(tmp_path / "catalog.json")
    r.add(_make_style("candy"))
    s = r.get("candy")
    assert s.id == "candy"
    assert s.name == "Candy"


def test_registry_get_unknown_raises(tmp_path: Path) -> None:
    r = StyleRegistry(tmp_path / "catalog.json")
    with pytest.raises(StyleNotFoundError):
        r.get("does_not_exist")


def test_registry_list_returns_copy(tmp_path: Path) -> None:
    r = StyleRegistry(tmp_path / "catalog.json")
    r.add(_make_style("candy"))
    lst = r.list_styles()
    lst.clear()  # mutating the returned list must not affect the registry
    assert len(r.list_styles()) == 1


# ---------------------------------------------------------------------------
# StyleRegistry — add
# ---------------------------------------------------------------------------

def test_registry_add_multiple_styles(tmp_path: Path) -> None:
    r = StyleRegistry(tmp_path / "catalog.json")
    r.add(_make_style("candy"))
    r.add(_make_style("mosaic", "Mosaic"))
    assert len(r.list_styles()) == 2


def test_registry_add_duplicate_raises(tmp_path: Path) -> None:
    r = StyleRegistry(tmp_path / "catalog.json")
    r.add(_make_style("candy"))
    with pytest.raises(DuplicateStyleError):
        r.add(_make_style("candy"))


def test_registry_add_persists_to_disk(tmp_path: Path) -> None:
    catalog = tmp_path / "catalog.json"
    r = StyleRegistry(catalog)
    r.add(_make_style("candy"))
    # Load a fresh registry from the same file
    r2 = StyleRegistry(catalog)
    assert "candy" in r2


# ---------------------------------------------------------------------------
# StyleRegistry — update
# ---------------------------------------------------------------------------

def test_registry_update_changes_name(tmp_path: Path) -> None:
    r = StyleRegistry(tmp_path / "catalog.json")
    r.add(_make_style("candy", "Old Name"))
    updated = _make_style("candy", "New Name")
    r.update(updated)
    assert r.get("candy").name == "New Name"


def test_registry_update_unknown_raises(tmp_path: Path) -> None:
    r = StyleRegistry(tmp_path / "catalog.json")
    with pytest.raises(StyleNotFoundError):
        r.update(_make_style("ghost"))


# ---------------------------------------------------------------------------
# StyleRegistry — delete
# ---------------------------------------------------------------------------

def test_registry_delete_removes_entry(tmp_path: Path) -> None:
    r = StyleRegistry(tmp_path / "catalog.json")
    r.add(_make_style("candy"))
    r.delete("candy")
    assert "candy" not in r
    assert len(r.list_styles()) == 0


def test_registry_delete_unknown_raises(tmp_path: Path) -> None:
    r = StyleRegistry(tmp_path / "catalog.json")
    with pytest.raises(StyleNotFoundError):
        r.delete("ghost")


def test_registry_delete_persists_to_disk(tmp_path: Path) -> None:
    catalog = tmp_path / "catalog.json"
    r = StyleRegistry(catalog)
    r.add(_make_style("candy"))
    r.delete("candy")
    r2 = StyleRegistry(catalog)
    assert "candy" not in r2


# ---------------------------------------------------------------------------
# StyleRegistry — import_trained_model
# ---------------------------------------------------------------------------

def test_registry_import_adds_new_style(tmp_path: Path) -> None:
    r = StyleRegistry(tmp_path / "catalog.json")
    style = _make_style("my_style", "My Style")
    style.is_builtin = False
    pth = tmp_path / "my_style.pth"
    onnx = tmp_path / "my_style.onnx"
    result = r.import_trained_model(pth, onnx, style)
    assert "my_style" in r
    assert result.model_path == str(onnx)


def test_registry_import_updates_existing_style(tmp_path: Path) -> None:
    catalog = tmp_path / "catalog.json"
    r = StyleRegistry(catalog)
    r.add(_make_style("my_style", "Old"))
    style = _make_style("my_style", "Updated")
    onnx = tmp_path / "my_style_v2.onnx"
    r.import_trained_model(tmp_path / "a.pth", onnx, style)
    assert r.get("my_style").model_path == str(onnx)
