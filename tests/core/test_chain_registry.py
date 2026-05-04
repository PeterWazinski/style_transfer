"""Unit tests for BuiltinChainRegistry and ChainNotFoundError."""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from src.core.chain_models import BuiltinChainModel, ChainStore
from src.core.chain_registry import BuiltinChainRegistry, ChainNotFoundError
from src.core.models import StyleModel
from src.core.registry import StyleRegistry


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
    )


def _make_catalog(tmp_path: Path, chains: list[BuiltinChainModel]) -> Path:
    catalog = tmp_path / "catalog.json"
    ChainStore(catalog).save(chains)
    return catalog


def _make_style_registry(tmp_path: Path, style_names: list[str]) -> StyleRegistry:
    """Create a StyleRegistry pre-populated with styles having the given names."""
    from src.core.models import StyleStore

    catalog = tmp_path / "styles" / "catalog.json"
    styles = [
        StyleModel(id=n.lower().replace(" ", "-"), name=n, model_path=f"styles/{n}/model.onnx")
        for n in style_names
    ]
    StyleStore(catalog).save(styles)
    return StyleRegistry(catalog_path=catalog)


def _write_yml(path: Path, steps: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.dump({"version": 1, "steps": steps}, f)


# ---------------------------------------------------------------------------
# list_chains / get
# ---------------------------------------------------------------------------

class TestListAndGet:
    def test_empty_catalog(self, tmp_path: Path) -> None:
        reg = BuiltinChainRegistry(catalog_path=tmp_path / "missing.json")
        assert reg.list_chains() == []

    def test_list_chains_sorted(self, tmp_path: Path) -> None:
        catalog = _make_catalog(tmp_path, [_make_chain("z-chain", "Zebra"), _make_chain("a-chain", "Alpha")])
        reg = BuiltinChainRegistry(catalog_path=catalog)
        names = [c.name for c in reg.list_chains()]
        assert names == ["Alpha", "Zebra"]

    def test_get_existing(self, tmp_path: Path) -> None:
        catalog = _make_catalog(tmp_path, [_make_chain("pastel")])
        reg = BuiltinChainRegistry(catalog_path=catalog)
        chain = reg.get("pastel")
        assert chain.id == "pastel"

    def test_get_missing_raises(self, tmp_path: Path) -> None:
        catalog = _make_catalog(tmp_path, [])
        reg = BuiltinChainRegistry(catalog_path=catalog)
        with pytest.raises(ChainNotFoundError):
            reg.get("nonexistent")


# ---------------------------------------------------------------------------
# validate_styles
# ---------------------------------------------------------------------------

class TestValidateStyles:
    def test_empty_catalog_is_valid(self, tmp_path: Path) -> None:
        chain_reg = BuiltinChainRegistry(catalog_path=_make_catalog(tmp_path, []))
        style_reg = _make_style_registry(tmp_path, [])
        assert chain_reg.validate_styles(style_reg, root=tmp_path) == {}

    def test_all_styles_present_is_valid(self, tmp_path: Path) -> None:
        # Write a chain YAML that references existing styles
        yml = tmp_path / "style_chains" / "pastel" / "chain.yml"
        _write_yml(yml, [{"style": "Ukiyo-e", "strength": 150}, {"style": "Cubism", "strength": 80}])
        chain = BuiltinChainModel(
            id="pastel", name="Pastel",
            chain_path="style_chains/pastel/chain.yml",
        )
        chain_reg = BuiltinChainRegistry(catalog_path=_make_catalog(tmp_path, [chain]))
        style_reg = _make_style_registry(tmp_path, ["Ukiyo-e", "Cubism"])
        result = chain_reg.validate_styles(style_reg, root=tmp_path)
        assert result == {}

    def test_missing_style_flagged(self, tmp_path: Path) -> None:
        yml = tmp_path / "style_chains" / "pastel" / "chain.yml"
        _write_yml(yml, [{"style": "Ukiyo-e", "strength": 150}, {"style": "Ghost Style", "strength": 80}])
        chain = BuiltinChainModel(
            id="pastel", name="Pastel",
            chain_path="style_chains/pastel/chain.yml",
        )
        chain_reg = BuiltinChainRegistry(catalog_path=_make_catalog(tmp_path, [chain]))
        style_reg = _make_style_registry(tmp_path, ["Ukiyo-e"])  # Ghost Style missing
        result = chain_reg.validate_styles(style_reg, root=tmp_path)
        assert "pastel" in result
        assert "Ghost Style" in result["pastel"]

    def test_missing_yml_flagged(self, tmp_path: Path) -> None:
        chain = BuiltinChainModel(
            id="pastel", name="Pastel",
            chain_path="style_chains/pastel/chain.yml",  # file not created
        )
        chain_reg = BuiltinChainRegistry(catalog_path=_make_catalog(tmp_path, [chain]))
        style_reg = _make_style_registry(tmp_path, [])
        result = chain_reg.validate_styles(style_reg, root=tmp_path)
        assert "pastel" in result

    def test_invalid_yml_flagged(self, tmp_path: Path) -> None:
        yml = tmp_path / "style_chains" / "bad" / "chain.yml"
        yml.parent.mkdir(parents=True, exist_ok=True)
        yml.write_text("not: valid: yaml: [[[", encoding="utf-8")
        chain = BuiltinChainModel(
            id="bad", name="Bad",
            chain_path="style_chains/bad/chain.yml",
        )
        chain_reg = BuiltinChainRegistry(catalog_path=_make_catalog(tmp_path, [chain]))
        style_reg = _make_style_registry(tmp_path, [])
        result = chain_reg.validate_styles(style_reg, root=tmp_path)
        assert "bad" in result
