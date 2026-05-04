"""Style chain schema and loader.

A style chain is a YAML file that records the sequence of style-transfer
steps applied to a photo in the interactive app.  It can be re-applied
headlessly via ``BatchStyler --apply-style-chain``.

Example file::

    # PetersPictureStyler – style chain
    # Created: 2026-04-30 14:32
    version: 1
    tile_size: 1024
    tile_overlap: 128
    steps:
      - style: Anime Hayao
        strength: 150
      - style: Van Gogh
        strength: 75

Schema rules:
- ``version`` must be 1 (allows future format evolution).
- ``steps`` must be a non-empty list.
- Each step requires ``style`` (non-empty string) and ``strength``
  (integer, 1–300, representing a percentage).
- ``tile_size`` and ``tile_overlap`` are optional integers.  When present
  they override the CLI / app defaults during replay.
"""
from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field, ValidationError, field_validator


class ChainStep(BaseModel):
    """One step in a style-transfer chain."""

    style: str = Field(..., min_length=1, description="Display name of the style (case-sensitive)")
    strength: int = Field(..., ge=1, le=300, description="Strength in percent (1–300)")


class StyleChain(BaseModel):
    """A complete style-transfer chain stored in a style-log file."""

    version: int = Field(default=1, description="File format version; must be 1")
    tile_size: int | None = Field(default=None, gt=0, description="Tile size in pixels used during recording")
    tile_overlap: int | None = Field(default=None, gt=0, description="Tile overlap in pixels used during recording")
    steps: list[ChainStep] = Field(..., min_length=1, description="Ordered list of style steps")

    @field_validator("version")
    @classmethod
    def _check_version(cls, v: int) -> int:
        if v != 1:
            raise ValueError(f"Unsupported style chain version: {v}. Only version 1 is supported.")
        return v


def load_style_chain(path: Path) -> StyleChain:
    """Load and validate a style chain YAML file.

    Args:
        path: Path to the ``.yml`` / ``.yaml`` file.

    Returns:
        A validated :class:`StyleChain` instance.

    Raises:
        ValueError: If the file has a YAML syntax error or fails schema
                    validation.  The message is human-readable and suitable
                    for display in a dialog or CLI error output.
    """
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(f"YAML syntax error in '{path.name}':\n{exc}") from exc

    if not isinstance(raw, dict):
        raise ValueError(
            f"'{path.name}' is not a valid style chain — expected a YAML mapping at the top level."
        )

    try:
        return StyleChain.model_validate(raw)
    except ValidationError as exc:
        # Flatten Pydantic's verbose error list into a readable message
        messages = "; ".join(
            f"{' → '.join(str(loc) for loc in e['loc'])}: {e['msg']}"
            for e in exc.errors()
        )
        raise ValueError(f"Invalid style chain '{path.name}': {messages}") from exc


def dump_style_chain(
    log: StyleChain,
    *,
    created_by: str = "PetersPictureStyler",
) -> str:
    """Serialise a :class:`StyleChain` to a YAML string.

    The output starts with a two-line human-readable header comment so the
    file is self-describing when opened in a text editor.

    Args:
        log:        The validated style-chain model to serialise.
        created_by: Application name written into the header comment.

    Returns:
        UTF-8 YAML string, ready to be written to a ``.yml`` file.
    """
    from datetime import datetime  # noqa: PLC0415 — lazy, only when serialising

    header = (
        f"# {created_by} \u2013 style chain\n"
        f"# Created: {datetime.now():%Y-%m-%d %H:%M}\n"
    )
    data: dict[str, object] = {"version": log.version}
    if log.tile_size is not None:
        data["tile_size"] = log.tile_size
    if log.tile_overlap is not None:
        data["tile_overlap"] = log.tile_overlap
    data["steps"] = [{"style": s.style, "strength": s.strength} for s in log.steps]
    return header + yaml.dump(data, allow_unicode=True, sort_keys=False, default_flow_style=False)
