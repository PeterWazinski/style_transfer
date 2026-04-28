"""StyleRegistry — CRUD operations over the style catalog.

All mutations immediately persist the catalog via `StyleStore`,
so the on-disk state is always consistent.

Typical usage::

    registry = StyleRegistry(catalog_path=Path("styles/catalog.json"))
    for style in registry.list_styles():
        print(style.name)

    registry.add(StyleModel(id="my_style", name="My Style", ...))
    registry.delete("my_style")
"""
from __future__ import annotations

import logging
from pathlib import Path

from src.core.models import StyleModel, StyleStore

logger: logging.Logger = logging.getLogger(__name__)


class DuplicateStyleError(ValueError):
    """Raised when trying to add a style whose ID already exists."""


class StyleNotFoundError(KeyError):
    """Raised when a requested style ID is not in the catalog."""


class StyleRegistry:
    """In-memory style catalog backed by a JSON file.

    The catalog is loaded lazily on first access and written back to disk
    after every mutating operation to ensure durability.

    Args:
        catalog_path: Path to the JSON catalog file (need not exist yet).
    """

    def __init__(self, catalog_path: Path) -> None:
        self._store: StyleStore = StyleStore(catalog_path)
        self._styles: list[StyleModel] | None = None  # lazy-loaded

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_loaded(self) -> None:
        if self._styles is None:
            self._styles = self._store.load()

    @property
    def _catalog(self) -> list[StyleModel]:
        self._ensure_loaded()
        assert self._styles is not None
        return self._styles

    def _persist(self) -> None:
        self._store.save(self._catalog)

    def _index_of(self, style_id: str) -> int:
        for i, s in enumerate(self._catalog):
            if s.id == style_id:
                return i
        raise StyleNotFoundError(
            f"Style '{style_id}' not found in the catalog."
        )

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    def list_styles(self) -> list[StyleModel]:
        """Return a shallow copy of all styles in the catalog, sorted alphabetically by name."""
        return sorted(self._catalog, key=lambda s: s.name.casefold())

    def get(self, style_id: str) -> StyleModel:
        """Return the style with the given ID.

        Raises:
            StyleNotFoundError: If the ID is not in the catalog.
        """
        idx = self._index_of(style_id)
        return self._catalog[idx]

    def __contains__(self, style_id: str) -> bool:
        return any(s.id == style_id for s in self._catalog)

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    def add(self, style: StyleModel) -> None:
        """Add *style* to the catalog.

        Raises:
            DuplicateStyleError: If a style with the same ID already exists.
        """
        if style.id in self:
            raise DuplicateStyleError(
                f"A style with ID '{style.id}' already exists. "
                "Use update() to modify it."
            )
        self._catalog.append(style)
        self._persist()
        logger.info("Style added: %s", style.id)

    def update(self, style: StyleModel) -> None:
        """Replace the existing catalog entry for *style.id* with *style*.

        Raises:
            StyleNotFoundError: If the ID is not in the catalog.
        """
        idx = self._index_of(style.id)
        self._catalog[idx] = style
        self._persist()
        logger.info("Style updated: %s", style.id)

    def delete(self, style_id: str) -> None:
        """Remove the style with *style_id* from the catalog.

        Does **not** delete any files (model .onnx / preview image) —
        callers are responsible for cleaning those up if desired.

        Raises:
            StyleNotFoundError: If the ID is not in the catalog.
        """
        idx = self._index_of(style_id)
        removed = self._catalog.pop(idx)
        self._persist()
        logger.info("Style deleted: %s", removed.id)

    def import_trained_model(
        self,
        pth_path: Path,
        onnx_path: Path,
        style: StyleModel,
    ) -> StyleModel:
        """Register a newly trained model in the catalog.

        Typically called after `StyleTrainer.export_onnx()` completes.
        Updates *style.model_path* to point at *onnx_path* before adding.

        Args:
            pth_path:  Path to the PyTorch .pth checkpoint (informational).
            onnx_path: Path to the exported .onnx model file.
            style:     Partially-filled `StyleModel` (id, name, etc.).

        Returns:
            The updated `StyleModel` as stored in the catalog.
        """
        style.model_path = str(onnx_path)
        if style.id in self:
            self.update(style)
        else:
            self.add(style)
        logger.info(
            "Imported trained model '%s' from pth=%s onnx=%s",
            style.id, pth_path, onnx_path,
        )
        return style
