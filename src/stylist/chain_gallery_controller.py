"""ChainGalleryController mixin — built-in chain operations for MainWindow.

Provides ``_apply_builtin_chain`` and ``_append_builtin_chain``.
Mixed into :class:`src.stylist.main_window.MainWindow`.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from PySide6.QtWidgets import QMessageBox

from src.core.chain_models import BuiltinChainModel
from src.core.style_chain_schema import load_style_chain
from src.stylist._utils import _get_project_root

if TYPE_CHECKING:
    from src.stylist.main_window import MainWindow

logger: logging.Logger = logging.getLogger(__name__)


class ChainGalleryController:
    """Mixin that adds built-in chain apply/append to MainWindow."""

    def _apply_builtin_chain(  # type: ignore[misc]
        self: "MainWindow",
        chain: BuiltinChainModel,
    ) -> None:
        """Apply *chain* fresh from the original photo (reset then run all steps)."""
        if self._current_photo is None:
            QMessageBox.information(self, "Apply Chain", "Open a photo first.")  # type: ignore[call-arg]
            return
        self._run_builtin_chain(chain, fresh=True)

    def _append_builtin_chain(  # type: ignore[misc]
        self: "MainWindow",
        chain: BuiltinChainModel,
    ) -> None:
        """Apply *chain* on top of the current photo state (append mode)."""
        if self._current_photo is None:
            QMessageBox.information(self, "Append Chain", "Open a photo first.")  # type: ignore[call-arg]
            return
        self._run_builtin_chain(chain, fresh=False)

    def _run_builtin_chain(  # type: ignore[misc]
        self: "MainWindow",
        chain: BuiltinChainModel,
        *,
        fresh: bool,
    ) -> None:
        """Common implementation shared by apply and append.

        Args:
            chain: The :class:`BuiltinChainModel` to run.
            fresh: When *True*, the first step always uses ``_apply_style``
                   (i.e. applies from the original photo, resetting the log).
                   When *False*, the first step uses ``_apply_style`` only if
                   ``_styled_photo`` is *None*, otherwise ``_reapply_style``.
        """
        root = _get_project_root()
        yml_path = chain.chain_path_resolved(root)
        try:
            sc = load_style_chain(yml_path)
        except ValueError as exc:
            QMessageBox.critical(self, "Apply Chain", str(exc))  # type: ignore[call-arg]
            return

        # Validate that all referenced styles are in the catalog.
        unknown: list[str] = [
            step.style
            for step in sc.steps
            if self._resolve_style_id_by_name(step.style) is None
        ]
        if unknown:
            names = "\n".join(f"  \u2022 {n}" for n in unknown)
            QMessageBox.critical(  # type: ignore[call-arg]
                self,
                "Apply Chain",
                "The following styles were not found in the catalog:\n"
                + names
                + "\n\nChain aborted.",
            )
            return

        dialog_title = "Apply Chain"
        for i, step in enumerate(sc.steps):
            style_id = self._resolve_style_id_by_name(step.style)
            assert style_id is not None
            self._current_style_name = step.style

            # Ensure ONNX model is loaded.
            if not self.engine.is_loaded(style_id):
                project_root = _get_project_root()
                style_obj = self.registry.get(style_id)
                try:
                    self.engine.load_model(
                        style_id,
                        style_obj.model_path_resolved(project_root),
                        tensor_layout=style_obj.tensor_layout,
                    )
                except Exception as exc:  # noqa: BLE001
                    QMessageBox.critical(  # type: ignore[call-arg]
                        self,
                        dialog_title,
                        f"Could not load model for \u2018{step.style}\u2019: {exc}",
                    )
                    return

            # For fresh mode the first step always applies from the original.
            use_apply = (fresh and i == 0) or (not fresh and self._styled_photo is None)
            if use_apply:
                self._apply_style(style_id, step.strength / 100.0)
            else:
                self._reapply_style(style_id, step.strength / 100.0)

        self._status.showMessage(f"Chain applied: {chain.name}")
