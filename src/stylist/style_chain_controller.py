"""StyleChainController mixin — style-chain operations for MainWindow.

Provides ``_resolve_style_id_by_name``, ``_format_style_chain``,
``_copy_style_chain_to_clipboard``, and ``_apply_style_chain``.
Mixed into :class:`src.stylist.main_window.MainWindow`.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from PySide6.QtWidgets import QApplication, QFileDialog, QMessageBox

from src.core.style_chain_schema import load_style_chain, dump_style_chain, ReplayLog, ReplayStep
from src.stylist._utils import _get_project_root

if TYPE_CHECKING:
    from src.stylist.main_window import MainWindow

logger: logging.Logger = logging.getLogger(__name__)


class StyleChainController:
    """Mixin that adds style-chain operations to MainWindow."""

    def _resolve_style_id_by_name(  # type: ignore[misc]
        self: "MainWindow",
        style_name: str,
    ) -> str | None:
        """Return the style id for the given display name (case-insensitive), or None."""
        style = self.registry.find_by_name(style_name)
        return style.id if style is not None else None

    def _format_style_chain(self: "MainWindow") -> str:  # type: ignore[misc]
        """Serialise the current style chain to a YAML string."""
        chain = ReplayLog(
            tile_size=self._settings.tile_size,
            tile_overlap=self._settings.overlap,
            steps=[ReplayStep(style=s["style"], strength=s["strength"])  # type: ignore[arg-type]
                   for s in self._replay_log],
        )
        return dump_style_chain(chain)

    def _copy_style_chain_to_clipboard(self: "MainWindow") -> None:  # type: ignore[misc]
        if not self._replay_log:
            QMessageBox.information(self, "Style Chain", "No styles applied yet \u2014 nothing to copy.")  # type: ignore[call-arg]
            return
        QApplication.clipboard().setText(self._format_style_chain())
        self._status.showMessage("Style chain copied to clipboard.")

    def _apply_style_chain(self: "MainWindow") -> None:  # type: ignore[misc]
        if self._current_photo is None:
            QMessageBox.information(self, "Apply Style Chain", "Open a photo first.")  # type: ignore[call-arg]
            return
        start_dir = self._settings.last_save_dir or self._settings.default_output_dir or ""
        path_str, _ = QFileDialog.getOpenFileName(
            self, "Apply Style Chain", start_dir, "YAML style chain (*.yml *.yaml)"  # type: ignore[call-arg]
        )
        if not path_str:
            return
        try:
            replay = load_style_chain(Path(path_str))
        except ValueError as exc:
            QMessageBox.critical(self, "Apply Style Chain", str(exc))  # type: ignore[call-arg]
            return
        unknown: list[str] = [
            step.style for step in replay.steps
            if self._resolve_style_id_by_name(step.style) is None
        ]
        if unknown:
            names = "\n".join(f"  \u2022 {n}" for n in unknown)
            QMessageBox.critical(
                self, "Apply Style Chain",  # type: ignore[call-arg]
                "The following styles were not found in the catalog:\n" + names + "\n\nChain aborted.",
            )
            return
        if replay.tile_size is not None:
            try:
                self._settings.tile_size = replay.tile_size
            except (ValueError, AttributeError):
                pass
        if replay.tile_overlap is not None:
            try:
                self._settings.overlap = replay.tile_overlap
            except (ValueError, AttributeError):
                pass
        self._styled_photo = None
        self._styled_photo_input = None
        self._clear_undo_stack()
        self._replay_log = []
        self.canvas.reset_styled()
        self._save_action.setEnabled(False)
        self.canvas.set_original(self._pil_to_pixmap(self._current_photo))
        for i, step in enumerate(replay.steps):
            style_id = self._resolve_style_id_by_name(step.style)
            assert style_id is not None
            self._current_style_name = step.style
            if not self.engine.is_loaded(style_id):
                project_root: Path = _get_project_root()
                if style_id in self.registry:
                    style_obj = self.registry.get(style_id)
                    try:
                        self.engine.load_model(
                            style_id,
                            style_obj.model_path_resolved(project_root),
                            tensor_layout=style_obj.tensor_layout,
                        )
                    except Exception as exc:  # noqa: BLE001
                        QMessageBox.critical(  # type: ignore[call-arg]
                            self, "Apply Style Chain",
                            f"Could not load model for \u2018{step.style}\u2019: {exc}",
                        )
                        return
            if i == 0:
                self._apply_style(style_id, step.strength / 100.0)
            else:
                self._reapply_style(style_id, step.strength / 100.0)
        self._status.showMessage(f"Style chain applied: {Path(path_str).name}")
