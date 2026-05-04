"""Tests for ChainGalleryController — _apply_builtin_chain and _append_builtin_chain."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, call, patch

import numpy as np
import pytest
import yaml
from PIL import Image

from src.core.chain_models import BuiltinChainModel, ChainStore
from src.core.chain_registry import BuiltinChainRegistry
from src.core.engine import StyleTransferEngine
from src.core.models import StyleModel
from src.core.photo_manager import PhotoManager
from src.core.registry import StyleRegistry
from src.core.settings import AppSettings
from src.stylist.main_window import MainWindow
from tests.helpers import make_mock_session


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dummy_image(size: int = 64) -> Image.Image:
    return Image.fromarray(np.zeros((size, size, 3), dtype=np.uint8))


def _write_chain_yml(path: Path, steps: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.dump({"version": 1, "steps": steps}, f)


def _make_chain(chain_id: str, chain_path: str) -> BuiltinChainModel:
    return BuiltinChainModel(
        id=chain_id, name=chain_id.capitalize(),
        chain_path=chain_path,
        step_count=2,
    )


def _make_window(
    qtbot,
    tmp_path: Path,
    style_names: list[str],
    chain_steps: list[dict],
) -> tuple[MainWindow, MagicMock, BuiltinChainModel]:
    """Build a MainWindow with registered styles and one built-in chain."""
    # Write style catalog
    styles: list[StyleModel] = []
    for name in style_names:
        sid = name.lower().replace(" ", "-")
        preview = tmp_path / f"{sid}_preview.jpg"
        _dummy_image().save(preview)
        styles.append(StyleModel(
            id=sid, name=name,
            model_path=str(tmp_path / f"{sid}.onnx"),
            preview_path=str(preview),
        ))
    registry = StyleRegistry(catalog_path=tmp_path / "catalog.json")
    for s in styles:
        registry.add(s)

    # Write chain catalog + YAML
    rel_path = f"style_chains/test-chain/chain.yml"
    abs_yml = tmp_path / "style_chains" / "test-chain" / "chain.yml"
    _write_chain_yml(abs_yml, chain_steps)
    chain_catalog = tmp_path / "chain_catalog.json"
    chain_model = _make_chain("test-chain", rel_path)
    ChainStore(chain_catalog).save([chain_model])
    chain_registry = BuiltinChainRegistry(catalog_path=chain_catalog)

    # Build engine with mocked session
    engine = StyleTransferEngine()
    with (
        patch("src.core.engine._ORT_AVAILABLE", True),
        patch("src.core.engine.ort") as mock_ort,
        patch.object(Path, "exists", return_value=True),
    ):
        mock_ort.InferenceSession.return_value = make_mock_session()
        for s in styles:
            engine.load_model(s.id, Path("dummy/model.onnx"))

    window = MainWindow(
        registry=registry,
        engine=engine,
        photo_manager=PhotoManager(),
        settings=AppSettings(),
        chain_registry=chain_registry,
    )
    qtbot.addWidget(window)
    return window, engine, chain_model


# ---------------------------------------------------------------------------
# _apply_builtin_chain
# ---------------------------------------------------------------------------

class TestApplyBuiltinChain:
    def test_requires_photo_open(self, qtbot, tmp_path: Path) -> None:
        """Without an open photo, shows an info dialog and does nothing."""
        window, engine, chain = _make_window(
            qtbot, tmp_path,
            style_names=["Ukiyo-e"],
            chain_steps=[{"style": "Ukiyo-e", "strength": 100}],
        )
        assert window._current_photo is None
        with patch("src.stylist.chain_gallery_controller.QMessageBox.information") as mock_info:
            window._apply_builtin_chain(chain)
        mock_info.assert_called_once()

    def test_apply_chain_single_step_uses_apply_style(
        self, qtbot, tmp_path: Path
    ) -> None:
        """Single-step chain: first (and only) step calls _apply_style."""
        window, engine, chain = _make_window(
            qtbot, tmp_path,
            style_names=["Ukiyo-e"],
            chain_steps=[{"style": "Ukiyo-e", "strength": 150}],
        )
        window._current_photo = _dummy_image()
        result = _dummy_image()
        apply_calls: list[tuple] = []
        reapply_calls: list[tuple] = []

        original_apply = window._apply_style
        original_reapply = window._reapply_style

        def fake_apply(style_id: str, strength: float) -> None:
            apply_calls.append((style_id, strength))
            window._styled_photo = result  # simulate result
            window._style_log = [{"style": window._current_style_name, "strength": int(strength * 100)}]

        def fake_reapply(style_id: str, strength: float) -> None:
            reapply_calls.append((style_id, strength))

        with (
            patch.object(window, "_apply_style", side_effect=fake_apply),
            patch.object(window, "_reapply_style", side_effect=fake_reapply),
            patch("src.stylist.chain_gallery_controller._get_project_root", return_value=tmp_path),
        ):
            window._apply_builtin_chain(chain)

        assert len(apply_calls) == 1
        assert apply_calls[0] == ("ukiyo-e", 1.5)
        assert len(reapply_calls) == 0

    def test_apply_chain_two_steps_uses_apply_then_reapply(
        self, qtbot, tmp_path: Path
    ) -> None:
        """Two-step chain: first step → _apply_style, second → _reapply_style."""
        window, engine, chain = _make_window(
            qtbot, tmp_path,
            style_names=["Ukiyo-e", "Cubism"],
            chain_steps=[
                {"style": "Ukiyo-e", "strength": 150},
                {"style": "Cubism", "strength": 80},
            ],
        )
        window._current_photo = _dummy_image()
        result = _dummy_image()
        apply_calls: list[tuple] = []
        reapply_calls: list[tuple] = []

        def fake_apply(style_id: str, strength: float) -> None:
            apply_calls.append((style_id, strength))
            window._styled_photo = result
            window._style_log = [{"style": window._current_style_name, "strength": int(strength * 100)}]

        def fake_reapply(style_id: str, strength: float) -> None:
            reapply_calls.append((style_id, strength))
            window._style_log.append({"style": window._current_style_name, "strength": int(strength * 100)})

        with (
            patch.object(window, "_apply_style", side_effect=fake_apply),
            patch.object(window, "_reapply_style", side_effect=fake_reapply),
            patch("src.stylist.chain_gallery_controller._get_project_root", return_value=tmp_path),
        ):
            window._apply_builtin_chain(chain)

        assert len(apply_calls) == 1
        assert apply_calls[0][0] == "ukiyo-e"
        assert len(reapply_calls) == 1
        assert reapply_calls[0][0] == "cubism"

    def test_apply_chain_resets_log_regardless_of_prior_state(
        self, qtbot, tmp_path: Path
    ) -> None:
        """Even if _styled_photo is already set, first step uses _apply_style."""
        window, engine, chain = _make_window(
            qtbot, tmp_path,
            style_names=["Ukiyo-e"],
            chain_steps=[{"style": "Ukiyo-e", "strength": 100}],
        )
        window._current_photo = _dummy_image()
        window._styled_photo = _dummy_image()   # already styled
        window._style_log = [{"style": "Previous", "strength": 80}]
        apply_calls: list = []

        def fake_apply(style_id: str, strength: float) -> None:
            apply_calls.append(style_id)
            window._styled_photo = _dummy_image()
            window._style_log = [{"style": window._current_style_name, "strength": int(strength * 100)}]

        with (
            patch.object(window, "_apply_style", side_effect=fake_apply),
            patch("src.stylist.chain_gallery_controller._get_project_root", return_value=tmp_path),
        ):
            window._apply_builtin_chain(chain)

        assert len(apply_calls) == 1  # fresh: _apply_style called even though _styled_photo was set

    def test_unknown_style_shows_error(self, qtbot, tmp_path: Path) -> None:
        window, engine, chain = _make_window(
            qtbot, tmp_path,
            style_names=["Ukiyo-e"],
            chain_steps=[{"style": "Ghost Style", "strength": 100}],
        )
        window._current_photo = _dummy_image()
        with (
            patch("src.stylist.chain_gallery_controller.QMessageBox.critical") as mock_crit,
            patch("src.stylist.chain_gallery_controller._get_project_root", return_value=tmp_path),
        ):
            window._apply_builtin_chain(chain)
        mock_crit.assert_called_once()


# ---------------------------------------------------------------------------
# _append_builtin_chain
# ---------------------------------------------------------------------------

class TestAppendBuiltinChain:
    def test_append_uses_reapply_when_styled_photo_exists(
        self, qtbot, tmp_path: Path
    ) -> None:
        """With an existing styled photo, _append_builtin_chain calls _reapply_style."""
        window, engine, chain = _make_window(
            qtbot, tmp_path,
            style_names=["Ukiyo-e"],
            chain_steps=[{"style": "Ukiyo-e", "strength": 100}],
        )
        window._current_photo = _dummy_image()
        window._styled_photo = _dummy_image()
        window._style_log = [{"style": "Previous", "strength": 80}]
        reapply_calls: list = []

        def fake_reapply(style_id: str, strength: float) -> None:
            reapply_calls.append(style_id)

        with (
            patch.object(window, "_reapply_style", side_effect=fake_reapply),
            patch("src.stylist.chain_gallery_controller._get_project_root", return_value=tmp_path),
        ):
            window._append_builtin_chain(chain)

        assert len(reapply_calls) == 1
        assert reapply_calls[0] == "ukiyo-e"

    def test_append_uses_apply_when_no_styled_photo(
        self, qtbot, tmp_path: Path
    ) -> None:
        """Without a styled photo, _append_builtin_chain acts like apply."""
        window, engine, chain = _make_window(
            qtbot, tmp_path,
            style_names=["Ukiyo-e"],
            chain_steps=[{"style": "Ukiyo-e", "strength": 100}],
        )
        window._current_photo = _dummy_image()
        window._styled_photo = None
        apply_calls: list = []

        def fake_apply(style_id: str, strength: float) -> None:
            apply_calls.append(style_id)
            window._styled_photo = _dummy_image()
            window._style_log = [{"style": window._current_style_name, "strength": int(strength * 100)}]

        with (
            patch.object(window, "_apply_style", side_effect=fake_apply),
            patch("src.stylist.chain_gallery_controller._get_project_root", return_value=tmp_path),
        ):
            window._append_builtin_chain(chain)

        assert len(apply_calls) == 1
