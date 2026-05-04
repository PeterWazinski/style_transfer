"""Unit tests for training/add_style_chain_helper.py."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

from training.add_style_chain_helper import validate_chain_styles, install_chain


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_styles_catalog(*names: str) -> dict:
    """Build a minimal styles catalog containing the given style names."""
    return {"styles": [{"id": n.lower().replace(" ", "-"), "name": n} for n in names]}


def _make_fake_image(tmp_path: Path, size: tuple[int, int] = (64, 64)) -> Path:
    p = tmp_path / "content.png"
    arr = np.zeros((*size, 3), dtype=np.uint8)
    Image.fromarray(arr).save(p)
    return p


# ---------------------------------------------------------------------------
# validate_chain_styles
# ---------------------------------------------------------------------------

class TestValidateChainStyles:
    def test_all_known_returns_empty_list(self) -> None:
        catalog = _make_styles_catalog("Ukiyo-e", "Cubism")
        steps = [{"style": "Ukiyo-e", "strength": 100}, {"style": "Cubism", "strength": 80}]
        assert validate_chain_styles(steps, catalog) == []

    def test_missing_style_returned(self) -> None:
        catalog = _make_styles_catalog("Cubism")
        steps = [{"style": "Ghost", "strength": 50}, {"style": "Cubism", "strength": 80}]
        missing = validate_chain_styles(steps, catalog)
        assert missing == ["Ghost"]

    def test_all_missing_all_returned(self) -> None:
        catalog = _make_styles_catalog("Cubism")
        steps = [{"style": "A", "strength": 100}, {"style": "B", "strength": 100}]
        missing = validate_chain_styles(steps, catalog)
        assert set(missing) == {"A", "B"}

    def test_empty_steps_returns_empty_list(self) -> None:
        catalog = _make_styles_catalog("Cubism")
        assert validate_chain_styles([], catalog) == []

    def test_empty_catalog_all_missing(self) -> None:
        catalog: dict = {"styles": []}
        steps = [{"style": "Cubism", "strength": 100}]
        assert validate_chain_styles(steps, catalog) == ["Cubism"]

    def test_missing_styles_key_in_catalog(self) -> None:
        """Catalog without a 'styles' key should treat all steps as missing."""
        catalog: dict = {}
        steps = [{"style": "Cubism", "strength": 100}]
        assert validate_chain_styles(steps, catalog) == ["Cubism"]

    def test_step_without_style_key_returns_none_entry(self) -> None:
        """step.get('style') returns None → None is treated as missing.
        The list comprehension then accesses step['style'] which raises KeyError;
        this documents that callers must always supply the 'style' key."""
        catalog = _make_styles_catalog("Cubism")
        steps = [{"strength": 100}]  # 'style' key absent
        with pytest.raises(KeyError):
            validate_chain_styles(steps, catalog)


# ---------------------------------------------------------------------------
# install_chain  (filesystem only — engine and registry are mocked out)
# ---------------------------------------------------------------------------

class TestInstallChain:
    """Tests for install_chain that verify file layout and catalog updates.

    The ONNX style-transfer engine is fully mocked so these tests run
    without any model files on disk.
    """

    def _mock_engine(self, out_image: Image.Image) -> MagicMock:
        engine = MagicMock()
        engine.is_loaded.return_value = False
        engine.apply.return_value = out_image
        return engine

    def _mock_registry(self) -> MagicMock:
        style = MagicMock()
        style.id = "cubism"
        style.tensor_layout = "NCHW"
        style.model_path_resolved.return_value = Path("styles/cubism/model.onnx")
        registry = MagicMock()
        registry.find_by_name.return_value = style
        return registry

    def _run_install(
        self,
        tmp_path: Path,
        steps: list[dict] | None = None,
        chain_name: str = "My Chain",
    ) -> str:
        chains_dir = tmp_path / "style_chains"
        chains_dir.mkdir()
        catalog_path = chains_dir / "catalog.json"
        chains_catalog: dict = {"chains": []}
        content_image = _make_fake_image(tmp_path)

        if steps is None:
            steps = [{"style": "Cubism", "strength": 80}]

        fake_result = Image.fromarray(
            np.zeros((32, 32, 3), dtype=np.uint8)
        )

        with (
            patch("src.core.engine.StyleTransferEngine",
                  return_value=self._mock_engine(fake_result)),
            patch("src.core.registry.StyleRegistry",
                  return_value=self._mock_registry()),
        ):
            return install_chain(
                steps=steps,
                chain_name=chain_name,
                chain_desc="A test chain",
                chain_tags=["test"],
                chains_dir=chains_dir,
                chains_catalog_path=catalog_path,
                chains_catalog=chains_catalog,
                existing_chain_ids=set(),
                content_image=content_image,
                repo_root=tmp_path,
            )

    def test_returns_chain_id(self, tmp_path: Path) -> None:
        chain_id = self._run_install(tmp_path, chain_name="My Chain")
        assert chain_id == "my-chain"

    def test_chain_id_spaces_become_hyphens(self, tmp_path: Path) -> None:
        chain_id = self._run_install(tmp_path, chain_name="Cool Art Style")
        assert chain_id == "cool-art-style"

    def test_chain_yml_written(self, tmp_path: Path) -> None:
        chain_id = self._run_install(tmp_path, chain_name="Pastel")
        yml = tmp_path / "style_chains" / chain_id / "chain.yml"
        assert yml.exists()
        assert yml.stat().st_size > 0

    def test_preview_jpg_written(self, tmp_path: Path) -> None:
        chain_id = self._run_install(tmp_path, chain_name="Pastel")
        preview = tmp_path / "style_chains" / chain_id / "preview.jpg"
        assert preview.exists()
        assert preview.stat().st_size > 0

    def test_catalog_json_updated(self, tmp_path: Path) -> None:
        self._run_install(tmp_path, chain_name="Pastel")
        catalog_path = tmp_path / "style_chains" / "catalog.json"
        assert catalog_path.exists()
        with catalog_path.open() as f:
            catalog = json.load(f)
        assert len(catalog["chains"]) == 1
        entry = catalog["chains"][0]
        assert entry["id"] == "pastel"
        assert entry["name"] == "Pastel"
        assert entry["description"] == "A test chain"
        assert entry["tags"] == ["test"]
        assert entry["step_count"] == 1

    def test_catalog_chain_path_fields(self, tmp_path: Path) -> None:
        self._run_install(tmp_path, chain_name="Dense")
        catalog_path = tmp_path / "style_chains" / "catalog.json"
        with catalog_path.open() as f:
            entry = json.load(f)["chains"][0]
        assert entry["chain_path"] == "style_chains/dense/chain.yml"
        assert entry["preview_path"] == "style_chains/dense/preview.jpg"

    def test_step_count_reflects_number_of_steps(self, tmp_path: Path) -> None:
        steps = [
            {"style": "Cubism", "strength": 80},
            {"style": "Cubism", "strength": 100},
            {"style": "Cubism", "strength": 60},
        ]
        self._run_install(tmp_path, steps=steps, chain_name="Three Step")
        catalog_path = tmp_path / "style_chains" / "catalog.json"
        with catalog_path.open() as f:
            entry = json.load(f)["chains"][0]
        assert entry["step_count"] == 3

    def test_duplicate_id_raises(self, tmp_path: Path) -> None:
        chains_dir = tmp_path / "style_chains"
        chains_dir.mkdir()
        catalog_path = chains_dir / "catalog.json"
        chains_catalog: dict = {"chains": []}
        content_image = _make_fake_image(tmp_path)
        fake_result = Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8))

        kwargs = dict(
            steps=[{"style": "Cubism", "strength": 80}],
            chain_name="Pastel",
            chain_desc="",
            chain_tags=[],
            chains_dir=chains_dir,
            chains_catalog_path=catalog_path,
            chains_catalog=chains_catalog,
            existing_chain_ids={"pastel"},  # already exists
            content_image=content_image,
            repo_root=tmp_path,
        )
        with (
            patch("src.core.engine.StyleTransferEngine",
                  return_value=self._mock_engine(fake_result)),
            patch("src.core.registry.StyleRegistry",
                  return_value=self._mock_registry()),
        ):
            with pytest.raises(AssertionError, match="already exists"):
                install_chain(**kwargs)

    def test_empty_name_raises(self, tmp_path: Path) -> None:
        chains_dir = tmp_path / "style_chains"
        chains_dir.mkdir()
        catalog_path = chains_dir / "catalog.json"
        chains_catalog: dict = {"chains": []}
        content_image = _make_fake_image(tmp_path)

        with pytest.raises(AssertionError, match="empty"):
            install_chain(
                steps=[{"style": "Cubism", "strength": 80}],
                chain_name="   ",
                chain_desc="",
                chain_tags=[],
                chains_dir=chains_dir,
                chains_catalog_path=catalog_path,
                chains_catalog=chains_catalog,
                existing_chain_ids=set(),
                content_image=content_image,
                repo_root=tmp_path,
            )

    def test_empty_steps_raises(self, tmp_path: Path) -> None:
        chains_dir = tmp_path / "style_chains"
        chains_dir.mkdir()
        catalog_path = chains_dir / "catalog.json"
        chains_catalog: dict = {"chains": []}
        content_image = _make_fake_image(tmp_path)

        with pytest.raises(AssertionError):
            install_chain(
                steps=[],
                chain_name="Empty",
                chain_desc="",
                chain_tags=[],
                chains_dir=chains_dir,
                chains_catalog_path=catalog_path,
                chains_catalog=chains_catalog,
                existing_chain_ids=set(),
                content_image=content_image,
                repo_root=tmp_path,
            )
