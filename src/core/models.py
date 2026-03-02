"""Data models for style catalog entries.

`StyleModel` is the single source of truth for one style.
`StyleStore` serialises/deserialises the catalog to/from JSON.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Optional


# ---------------------------------------------------------------------------
# StyleModel
# ---------------------------------------------------------------------------

@dataclass
class StyleModel:
    """Metadata for one style (built-in or user-created)."""

    id: str                                         # unique slug, e.g. "candy"
    name: str                                       # display name, e.g. "Candy"
    model_path: str                                 # relative path to .onnx file
    preview_path: str                               # relative path to thumbnail
    description: str = ""
    author: str = ""
    source_images: list[str] = field(default_factory=list)  # style reference images
    is_builtin: bool = True
    training_config: Optional[dict[str, Any]] = None

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    def model_path_resolved(self, root: Path) -> Path:
        """Return absolute model path relative to *root*."""
        return root / self.model_path

    def preview_path_resolved(self, root: Path) -> Path:
        """Return absolute preview path relative to *root*."""
        return root / self.preview_path

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = asdict(self)
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StyleModel":
        # Allow extra keys from future versions; keep only known fields.
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        filtered = {k: v for k, v in data.items() if k in known}
        return cls(**filtered)


# ---------------------------------------------------------------------------
# StyleStore  — JSON persistence
# ---------------------------------------------------------------------------

class StyleStore:
    """Loads and saves the style catalog JSON file.

    The catalog is a JSON object of the form::

        {
            "styles": [
                {"id": "candy", "name": "Candy", ...},
                ...
            ]
        }

    The store never assumes that the file exists up-front; calling
    :meth:`save` creates missing parent directories automatically.
    """

    def __init__(self, catalog_path: Path) -> None:
        self.catalog_path: Path = catalog_path

    # ------------------------------------------------------------------
    # I/O
    # ------------------------------------------------------------------

    def load(self) -> list[StyleModel]:
        """Load all styles from the catalog file.

        Returns an empty list if the file does not exist.
        """
        if not self.catalog_path.exists():
            return []
        with self.catalog_path.open("r", encoding="utf-8") as f:
            raw: dict[str, Any] = json.load(f)
        return [StyleModel.from_dict(entry) for entry in raw.get("styles", [])]

    def save(self, styles: list[StyleModel]) -> None:
        """Persist *styles* to the catalog file, overwriting any previous content."""
        self.catalog_path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {
            "styles": [s.to_dict() for s in styles]
        }
        with self.catalog_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
