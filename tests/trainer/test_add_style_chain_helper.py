"""Tests for training/add_style_chain_helper.py."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from training.add_style_chain_helper import validate_chain_styles


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_styles_catalog(names: list[str]) -> dict:
    return {
        "styles": [
            {"id": n.lower().replace(" ", "_"), "name": n, "model_path": f"styles/{n}/model.onnx"}
            for n in names
        ]
    }


def _make_chains_catalog(chain_ids: list[str]) -> dict:
    return {
        "chains": [
            {"id": cid, "name": cid.capitalize(), "chain_path": f"style_chains/{cid}/chain.yml",
             "preview_path": "", "step_count": 1, "tags": []}
            for cid in chain_ids
        ]
    }


# ---------------------------------------------------------------------------
# validate_chain_styles
# ---------------------------------------------------------------------------

class TestValidateChainStyles:
    def test_all_styles_present_returns_empty(self) -> None:
        catalog = _make_styles_catalog(["Ukiyo-e", "Cubism"])
        steps = [{"style": "Ukiyo-e", "strength": 150}, {"style": "Cubism", "strength": 80}]
        assert validate_chain_styles(steps, catalog) == []

    def test_missing_style_returned(self) -> None:
        catalog = _make_styles_catalog(["Ukiyo-e"])
        steps = [{"style": "Ukiyo-e", "strength": 150}, {"style": "Ghost", "strength": 80}]
        missing = validate_chain_styles(steps, catalog)
        assert missing == ["Ghost"]

    def test_all_missing(self) -> None:
        catalog = _make_styles_catalog([])
        steps = [{"style": "A", "strength": 100}, {"style": "B", "strength": 100}]
        missing = validate_chain_styles(steps, catalog)
        assert set(missing) == {"A", "B"}

    def test_empty_steps_returns_empty(self) -> None:
        catalog = _make_styles_catalog(["Ukiyo-e"])
        assert validate_chain_styles([], catalog) == []

    def test_empty_catalog_all_missing(self) -> None:
        steps = [{"style": "Any", "strength": 100}]
        assert validate_chain_styles(steps, {"styles": []}) == ["Any"]


# ---------------------------------------------------------------------------
# setup() — file-system-level
# ---------------------------------------------------------------------------

class TestSetup:
    def test_setup_returns_context(self, tmp_path: Path) -> None:
        from training.add_style_chain_helper import setup

        # Create minimal required files
        styles_dir = tmp_path / "styles"
        styles_dir.mkdir()
        catalog = {"styles": []}
        (styles_dir / "catalog.json").write_text(json.dumps(catalog), encoding="utf-8")
        sample_dir = tmp_path / "sample_images"
        sample_dir.mkdir()
        (sample_dir / "arch.png").write_bytes(b"\x89PNG\r\n\x1a\n")  # minimal header

        ctx = setup(repo_root=tmp_path)
        assert ctx.repo_root == tmp_path
        assert ctx.existing_chain_ids == set()
        assert ctx.styles_catalog == catalog

    def test_setup_loads_existing_chain_ids(self, tmp_path: Path) -> None:
        from training.add_style_chain_helper import setup

        styles_dir = tmp_path / "styles"
        styles_dir.mkdir()
        (styles_dir / "catalog.json").write_text(json.dumps({"styles": []}), encoding="utf-8")
        sample_dir = tmp_path / "sample_images"
        sample_dir.mkdir()
        (sample_dir / "arch.png").write_bytes(b"\x89PNG\r\n\x1a\n")

        chains_dir = tmp_path / "style_chains"
        chains_dir.mkdir()
        chain_catalog = _make_chains_catalog(["pastel", "dense"])
        (chains_dir / "catalog.json").write_text(json.dumps(chain_catalog), encoding="utf-8")

        ctx = setup(repo_root=tmp_path)
        assert ctx.existing_chain_ids == {"pastel", "dense"}

    def test_setup_raises_when_styles_catalog_missing(self, tmp_path: Path) -> None:
        from training.add_style_chain_helper import setup
        with pytest.raises(AssertionError, match="Styles catalog not found"):
            setup(repo_root=tmp_path)
