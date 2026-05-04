"""BuiltinChainRegistry — read operations over the built-in chain catalog.

Chains are read-only from the application's perspective; the catalog
is only written by the developer tooling (add_style_chain notebook).

Typical usage::

    registry = BuiltinChainRegistry(catalog_path=Path("style_chains/catalog.json"))
    for chain in registry.list_chains():
        print(chain.name)

    invalid = registry.validate_styles(style_registry)
    for chain_id, missing in invalid.items():
        logger.warning("Chain '%s' missing styles: %s", chain_id, missing)
"""
from __future__ import annotations

import logging
from pathlib import Path

from src.core.chain_models import BuiltinChainModel, ChainStore

logger: logging.Logger = logging.getLogger(__name__)


class ChainNotFoundError(KeyError):
    """Raised when a requested chain ID is not in the catalog."""


class BuiltinChainRegistry:
    """In-memory built-in chain catalog backed by a JSON file.

    The catalog is loaded lazily on first access.

    Args:
        catalog_path: Path to the JSON catalog file (need not exist yet).
    """

    def __init__(self, catalog_path: Path) -> None:
        self._store: ChainStore = ChainStore(catalog_path)
        self._chains: list[BuiltinChainModel] | None = None  # lazy-loaded

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_loaded(self) -> None:
        if self._chains is None:
            self._chains = self._store.load()

    @property
    def _catalog(self) -> list[BuiltinChainModel]:
        self._ensure_loaded()
        assert self._chains is not None
        return self._chains

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    def list_chains(self) -> list[BuiltinChainModel]:
        """Return all chains sorted alphabetically by name."""
        return sorted(self._catalog, key=lambda c: c.name.casefold())

    def get(self, chain_id: str) -> BuiltinChainModel:
        """Return the chain with *chain_id*, or raise :exc:`ChainNotFoundError`."""
        for chain in self._catalog:
            if chain.id == chain_id:
                return chain
        raise ChainNotFoundError(f"Chain '{chain_id}' not found in the catalog.")

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate_styles(
        self,
        style_registry: "StyleRegistry",  # type: ignore[name-defined]  # noqa: F821
        root: Path | None = None,
    ) -> dict[str, list[str]]:
        """Check that every step in every chain references a known style.

        Args:
            style_registry: The application's :class:`~src.core.registry.StyleRegistry`.
            root:           Project root used to resolve relative ``chain_path``
                            values.  When *None* the current working directory
                            is used.

        Returns:
            A ``dict`` mapping ``chain_id`` → ``[missing_style_name, ...]``.
            Only chains with at least one missing style are included.
            An empty dict means all chains are valid.
        """
        from src.core.style_chain_schema import load_style_chain

        if root is None:
            root = Path.cwd()

        invalid: dict[str, list[str]] = {}
        for chain in self._catalog:
            yml_path = chain.chain_path_resolved(root)
            if not yml_path.exists():
                logger.warning(
                    "Built-in chain '%s': YAML not found at %s", chain.id, yml_path
                )
                invalid[chain.id] = [f"<missing file: {yml_path.name}>"]
                continue
            try:
                sc = load_style_chain(yml_path)
            except ValueError as exc:
                logger.warning(
                    "Built-in chain '%s': invalid YAML: %s", chain.id, exc
                )
                invalid[chain.id] = [f"<invalid YAML: {exc}>"]
                continue
            missing = [
                step.style
                for step in sc.steps
                if style_registry.find_by_name(step.style) is None
            ]
            if missing:
                logger.warning(
                    "Built-in chain '%s' references unknown styles: %s",
                    chain.id,
                    missing,
                )
                invalid[chain.id] = missing
        return invalid
