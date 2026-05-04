"""Data models for built-in style-chain catalog entries.

`BuiltinChainModel` is the single source of truth for one built-in style chain.
`ChainStore` serialises/deserialises the chain catalog to/from JSON.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# BuiltinChainModel
# ---------------------------------------------------------------------------

@dataclass
class BuiltinChainModel:
    """Metadata for one built-in style chain."""

    id: str                                      # unique slug, e.g. "pastel"
    name: str                                    # display name, e.g. "Pastel"
    chain_path: str                              # relative path to .yml file
    preview_path: str = ""                       # relative path to thumbnail (optional)
    description: str = ""
    step_count: int = 0                          # denormalised; set by add_style_chain notebook
    tags: list[str] = field(default_factory=list)  # reserved for future filtering

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    def chain_path_resolved(self, root: Path) -> Path:
        """Return absolute chain YAML path relative to *root*."""
        return root / self.chain_path

    def preview_path_resolved(self, root: Path) -> Path:
        """Return absolute preview path relative to *root*."""
        return root / self.preview_path

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BuiltinChainModel":
        # Allow extra keys from future versions; keep only known fields.
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        filtered = {k: v for k, v in data.items() if k in known}
        return cls(**filtered)


# ---------------------------------------------------------------------------
# ChainStore  — JSON persistence (read-only at runtime)
# ---------------------------------------------------------------------------

class ChainStore:
    """Loads and saves the built-in chain catalog JSON file.

    The catalog is a JSON object of the form::

        {
            "chains": [
                {"id": "pastel", "name": "Pastel", ...},
                ...
            ]
        }

    The store never assumes that the file exists up-front; calling
    :meth:`load` on a missing file returns an empty list.
    :meth:`save` is provided for use by the developer notebook only.
    """

    def __init__(self, catalog_path: Path) -> None:
        self.catalog_path: Path = catalog_path

    def load(self) -> list[BuiltinChainModel]:
        """Load all chains from the catalog file.

        Returns an empty list if the file does not exist.
        """
        if not self.catalog_path.exists():
            return []
        with self.catalog_path.open("r", encoding="utf-8") as f:
            raw: dict[str, Any] = json.load(f)
        return [BuiltinChainModel.from_dict(entry) for entry in raw.get("chains", [])]

    def save(self, chains: list[BuiltinChainModel]) -> None:
        """Persist *chains* to the catalog file, overwriting any previous content."""
        self.catalog_path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {
            "chains": [c.to_dict() for c in chains]
        }
        with self.catalog_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
