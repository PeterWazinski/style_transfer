"""Tests for src/core/style_chain_schema.py."""
from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from src.core.style_chain_schema import ReplayLog, ReplayStep, load_style_chain


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_yaml(tmp_path: Path, content: str) -> Path:
    p = tmp_path / "chain.yml"
    p.write_text(textwrap.dedent(content), encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestReplaySchema:
    def test_valid_log_parses(self, tmp_path: Path) -> None:
        """A well-formed YAML file round-trips to the correct model."""
        yml = _write_yaml(
            tmp_path,
            """\
            version: 1
            steps:
              - style: Anime Hayao
                strength: 150
              - style: Van Gogh
                strength: 75
            """,
        )
        replay = load_style_chain(yml)
        assert replay.version == 1
        assert replay.tile_size is None
        assert replay.tile_overlap is None
        assert len(replay.steps) == 2
        assert replay.steps[0] == ReplayStep(style="Anime Hayao", strength=150)
        assert replay.steps[1] == ReplayStep(style="Van Gogh", strength=75)

    def test_valid_log_with_tile_settings(self, tmp_path: Path) -> None:
        """tile_size and tile_overlap are parsed correctly when present."""
        yml = _write_yaml(
            tmp_path,
            """\
            version: 1
            tile_size: 512
            tile_overlap: 64
            steps:
              - style: Candy
                strength: 100
            """,
        )
        replay = load_style_chain(yml)
        assert replay.tile_size == 512
        assert replay.tile_overlap == 64

    def test_missing_steps_raises(self, tmp_path: Path) -> None:
        """A YAML file without a ``steps`` key raises ValueError."""
        yml = _write_yaml(tmp_path, "version: 1\n")
        with pytest.raises(ValueError, match="steps"):
            load_style_chain(yml)

    def test_empty_steps_raises(self, tmp_path: Path) -> None:
        """An empty ``steps`` list raises ValueError."""
        yml = _write_yaml(
            tmp_path,
            """\
            version: 1
            steps: []
            """,
        )
        with pytest.raises(ValueError):
            load_style_chain(yml)

    def test_strength_out_of_range_raises(self, tmp_path: Path) -> None:
        """Strength values outside 1–300 raise ValueError."""
        yml_high = _write_yaml(
            tmp_path,
            """\
            version: 1
            steps:
              - style: Test
                strength: 301
            """,
        )
        with pytest.raises(ValueError):
            load_style_chain(yml_high)

        yml_low = tmp_path / "low.yml"
        yml_low.write_text("version: 1\nsteps:\n  - style: Test\n    strength: 0\n", encoding="utf-8")
        with pytest.raises(ValueError):
            load_style_chain(yml_low)

    def test_unknown_version_raises(self, tmp_path: Path) -> None:
        """A ``version`` other than 1 raises ValueError with a clear message."""
        yml = _write_yaml(
            tmp_path,
            """\
            version: 99
            steps:
              - style: Test
                strength: 100
            """,
        )
        with pytest.raises(ValueError, match="[Uu]nsupported.*version"):
            load_style_chain(yml)

    def test_yaml_syntax_error_raises_valueerror(self, tmp_path: Path) -> None:
        """A malformed YAML file raises ValueError with a human-readable message."""
        bad = tmp_path / "bad.yml"
        bad.write_text("version: 1\nsteps: [\n  unclosed bracket\n", encoding="utf-8")
        with pytest.raises(ValueError, match="YAML syntax error"):
            load_style_chain(bad)
