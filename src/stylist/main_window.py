"""MainWindow — application shell wiring all UI components together.

Layout
------
::

    ┌─ Menu bar (File | Styles | View | Help) ─────────────────────────┐
    │                                                                    │
    │  ┌── Style Gallery (left dock) ──┐  ┌── Photo Canvas (central) ─┐│
    │  │  [thumb] [thumb] [thumb]      │  │   original │ styled        ││
    │  │  + Add Style  – Delete        │  │   Strength: ──●──          ││
    │  └───────────────────────────────┘  │   [Open] [Apply] [Save]    ││
    │                                      └────────────────────────────┘│
    │                                                                    │
    │  Status bar ──────────────────────────────────────────────────────│
    └────────────────────────────────────────────────────────────────────┘
"""
from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from PIL.Image import Image as PILImage
from PySide6.QtGui import QAction, QPixmap
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDockWidget,
    QFileDialog,
    QMainWindow,
    QMessageBox,
    QStatusBar,
    QWidget,
)

from src.core.engine import StyleTransferEngine
from src.core.models import StyleModel
from src.core.photo_manager import PhotoManager, UnsupportedFormatError
from src.core.registry import StyleRegistry
from src.core.settings import AppSettings
from src.stylist.apply_controller import ApplyController
from src.stylist.help_dialogs import show_how_to_use, show_about_nst, show_credits
from src.stylist.photo_canvas import PhotoCanvasView
from src.stylist.settings_dialog import SettingsDialog
from src.stylist.style_chain_controller import StyleChainController
from src.stylist.style_gallery import StyleGalleryView
from src.stylist._utils import _get_project_root

logger: logging.Logger = logging.getLogger(__name__)


@dataclass
class _UndoSnapshot:
    """State captured before each Apply / Re-Apply so the operation can be undone."""

    styled_photo: Optional["PILImage"]
    styled_photo_input: Optional["PILImage"]
    left_pane_pil: Optional["PILImage"]
    has_styled: bool


class MainWindow(ApplyController, StyleChainController, QMainWindow):
    """Top-level application window.

    Args:
        registry:      Injected :class:`StyleRegistry` instance.
        engine:        Injected :class:`StyleTransferEngine` instance.
        photo_manager: Injected :class:`PhotoManager` instance.
        settings:      Application settings; defaults are used if *None*.
        parent:        Optional parent widget.
    """

    def __init__(
        self,
        registry: StyleRegistry,
        engine: StyleTransferEngine,
        photo_manager: PhotoManager,
        settings: Optional[AppSettings] = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        # Expose services as public attrs so tests can inspect state.
        self.registry = registry
        self.engine = engine
        self.photo_manager = photo_manager
        self._settings: AppSettings = settings or AppSettings.load()
        self._current_photo: Optional[PILImage] = None
        self._current_photo_path: Optional[Path] = None
        self._styled_photo: Optional[PILImage] = None
        self._styled_photo_input: Optional[PILImage] = None  # source that produced _styled_photo
        self._left_pane_pil: Optional[PILImage] = None       # PIL mirror of left pane for undo
        self._undo_stack: deque[_UndoSnapshot] = deque(maxlen=3)
        self._current_style_name: str = ""                   # display name of selected style
        self._replay_log: list[dict[str, object]] = []       # {"style": str, "strength": int}

        self.setWindowTitle("Peter's Picture Stylist")
        self.resize(1200, 750)
        # Set window icon (also pins the palette icon to the Windows taskbar)
        from src.stylist.app import _make_palette_icon  # noqa: PLC0415
        self.setWindowIcon(_make_palette_icon())
        self._build_ui()
        self._wire_signals()

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        # Central widget
        self.canvas = PhotoCanvasView(self)
        self.setCentralWidget(self.canvas)

        # Left dock: style gallery
        self.gallery = StyleGalleryView(self.registry, self)
        dock = QDockWidget("Styles", self)
        dock.setWidget(self.gallery)
        dock.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetMovable
            | QDockWidget.DockWidgetFeature.DockWidgetFloatable
        )
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, dock)
        # Initial dock width: 3 columns × (130 px tile + 2×4 px spacing) + margins + scrollbar ≈ 450 px
        self.resizeDocks([dock], [440], Qt.Orientation.Horizontal)

        # Status bar
        self._status = QStatusBar(self)
        self.setStatusBar(self._status)
        self._status.showMessage("Ready")

        # Menu bar
        self._build_menus()

    def _build_menus(self) -> None:
        mb = self.menuBar()

        # File menu
        file_menu = mb.addMenu("File")
        open_action = QAction("Open Photo…", self)
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self._open_photo)
        file_menu.addAction(open_action)

        self._save_action = QAction("Save Result…", self)
        self._save_action.setShortcut("Ctrl+S")
        self._save_action.setEnabled(False)
        self._save_action.triggered.connect(self._save_result)
        file_menu.addAction(self._save_action)

        self._chain_copy_action = QAction("Style Chain to Clipboard", self)
        self._chain_copy_action.setStatusTip("Copy the current style chain as YAML to the clipboard")
        self._chain_copy_action.triggered.connect(self._copy_style_chain_to_clipboard)
        file_menu.addAction(self._chain_copy_action)

        self._chain_append_action = QAction("Append Style Chain…", self)
        self._chain_append_action.setStatusTip("Load a .yml style chain and append it on top of the current photo state")
        self._chain_append_action.triggered.connect(self._append_style_chain)
        file_menu.addAction(self._chain_append_action)

        file_menu.addSeparator()
        settings_action = QAction("Settings\u2026", self)
        settings_action.setShortcut("Ctrl+,")
        settings_action.triggered.connect(self._open_settings_dialog)
        file_menu.addAction(settings_action)

        file_menu.addSeparator()
        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # Help menu
        help_menu = mb.addMenu("Help")
        how_to_action = QAction("How to Use\u2026", self)
        how_to_action.triggered.connect(self._show_how_to_use)
        help_menu.addAction(how_to_action)

        help_menu.addSeparator()
        about_nst_action = QAction("About Neural Style Transfer\u2026", self)
        about_nst_action.triggered.connect(self._show_about_nst)
        help_menu.addAction(about_nst_action)

        help_menu.addSeparator()
        about_action = QAction("Credits\u2026", self)
        about_action.triggered.connect(self._show_credits)
        help_menu.addAction(about_action)

    # ------------------------------------------------------------------
    # Signal wiring
    # ------------------------------------------------------------------

    def _wire_signals(self) -> None:
        # Gallery → canvas
        self.gallery.style_selected.connect(self._on_style_selected)
        self.gallery.style_apply_requested.connect(self._on_style_apply_requested)
        # Canvas → actions
        self.canvas.open_photo_requested.connect(self._open_photo)
        self.canvas.reset_requested.connect(self._reset_photo)
        self.canvas.apply_requested.connect(self._apply_style)
        self.canvas.reapply_requested.connect(self._reapply_style)
        self.canvas.reapply_strength_requested.connect(self._reapply_style_strength)
        self.canvas.save_requested.connect(self._save_result)
        self.canvas.undo_requested.connect(self._perform_undo)

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_style_selected(self, style: StyleModel) -> None:
        self.canvas.set_active_style(style.id)
        self._current_style_name = style.name
        self._status.showMessage(f"Style selected: {style.name}")
        # Preload ONNX if not already loaded
        if not self.engine.is_loaded(style.id):
            # model_path is stored as a str relative to the project root
            project_root: Path = _get_project_root()
            model_path: Path = style.model_path_resolved(project_root)
            try:
                self.engine.load_model(
                    style.id, model_path, tensor_layout=style.tensor_layout
                )
            except Exception as exc:  # noqa: BLE001
                logger.exception("Could not load model '%s' from %s", style.id, model_path)
                self._status.showMessage(f"Could not load model: {exc}")

    def _on_style_apply_requested(self, style: StyleModel) -> None:
        """Double-click on gallery thumbnail: select the style then apply immediately."""
        self._on_style_selected(style)
        if self._current_photo is not None:
            self._apply_style(style.id, self.canvas.strength_slider.strength())

    def _reset_photo(self) -> None:
        """Confirm, then reload the original photo as if it were just opened."""
        if self._current_photo_path is None:
            return
        reply = QMessageBox.question(
            self,
            "Reset Style Filters",
            "Do you really want to reset all style filters?",
            QMessageBox.Yes | QMessageBox.Cancel,  # type: ignore[attr-defined]
            QMessageBox.Cancel,  # type: ignore[attr-defined]
        )
        if reply != QMessageBox.Yes:  # type: ignore[attr-defined]
            return
        try:
            image = self.photo_manager.load(self._current_photo_path, max_megapixels=self._settings.max_megapixels)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Load Error", str(exc))
            return
        self.engine.unload_all_models()  # free DirectML/GPU memory for clean state
        self._current_photo = image
        self._styled_photo = None
        self._styled_photo_input = None
        self._left_pane_pil = image
        self._clear_undo_stack()
        self._replay_log = []
        self.canvas.reset_styled()
        self._save_action.setEnabled(False)
        self.canvas.set_original(self._pil_to_pixmap(image))
        self._status.showMessage(
            f"Reset: {self._current_photo_path.name}  ({image.width}\u00d7{image.height})"
        )

    def _open_photo(self) -> None:
        path_str, _ = QFileDialog.getOpenFileName(
            self,
            "Open Photo",
            self._settings.last_open_dir or "",
            "Images (*.jpg *.jpeg *.png)",
        )
        if not path_str:
            return
        path = Path(path_str)
        try:
            image = self.photo_manager.load(path, max_megapixels=self._settings.max_megapixels)
        except UnsupportedFormatError as exc:
            QMessageBox.warning(self, "Unsupported Format", str(exc))
            return
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Load Error", str(exc))
            return

        self._settings.last_open_dir = str(path.parent)
        self._settings.save()
        self._current_photo = image
        self._current_photo_path = path
        self._styled_photo = None          # clear any previous styled result
        self._styled_photo_input = None
        self._left_pane_pil = image
        self._clear_undo_stack()
        self._replay_log = []
        self.canvas.reset_styled()         # reset right pane + disable Re-Apply/Save
        self._save_action.setEnabled(False)
        # Convert PIL Image → QPixmap for display
        pixmap = self._pil_to_pixmap(image)
        self.canvas.set_original(pixmap)
        mp = image.width * image.height / 1_000_000
        limit = self._settings.max_megapixels
        note = f"  [auto-resized to {mp:.1f} MP]" if limit > 0 and mp < limit * 0.99 else ""
        self._status.showMessage(
            f"Opened: {path.name}  ({image.width}\u00d7{image.height}){note}"
        )

    # ------------------------------------------------------------------
    # Undo stack
    # ------------------------------------------------------------------

    def _push_undo_snapshot(self) -> None:
        """Capture current state onto the undo stack and enable the Undo button."""
        self._undo_stack.append(_UndoSnapshot(
            styled_photo=self._styled_photo,
            styled_photo_input=self._styled_photo_input,
            left_pane_pil=self._left_pane_pil,
            has_styled=self.canvas.has_styled(),
        ))
        self.canvas.set_undo_available(True)

    def _clear_undo_stack(self) -> None:
        """Discard all undo history and disable the Undo button."""
        self._undo_stack.clear()
        self.canvas.set_undo_available(False)

    def _perform_undo(self) -> None:
        """Pop the top snapshot and restore the canvas to that state."""
        if not self._undo_stack:
            return
        snap = self._undo_stack.pop()
        self._styled_photo = snap.styled_photo
        self._styled_photo_input = snap.styled_photo_input
        self._left_pane_pil = snap.left_pane_pil
        # Pop matching replay log entry
        if self._replay_log:
            self._replay_log.pop()
        # Restore left pane
        left_pil = snap.left_pane_pil if snap.left_pane_pil is not None else self._current_photo
        if left_pil is not None:
            self.canvas.split_view.set_original_pixmap(self._pil_to_pixmap(left_pil))
        # Restore right pane
        if snap.has_styled and snap.styled_photo is not None:
            self.canvas.set_styled(self._pil_to_pixmap(snap.styled_photo))
            self._save_action.setEnabled(True)
        else:
            self.canvas.reset_styled()
            self._save_action.setEnabled(False)
        self.canvas.set_undo_available(len(self._undo_stack) > 0)
        self._status.showMessage("Undo.")

    # ------------------------------------------------------------------
    # Save result
    # ------------------------------------------------------------------

    def _save_result(self) -> None:
        if self._styled_photo is None:
            return
        start_dir = self._settings.last_save_dir or self._settings.default_output_dir or ""
        path_str, _ = QFileDialog.getSaveFileName(
            self,
            "Save Result",
            start_dir,
            "JPEG (*.jpg);;PNG (*.png)",
        )
        if not path_str:
            return
        path = Path(path_str)
        try:
            self.photo_manager.save(self._styled_photo, path)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Save Error", str(exc))
            return
        self._settings.last_save_dir = str(path.parent)
        self._settings.save()
        # Auto-save replay log alongside the image if enabled and log is non-empty
        if self._settings.autosave_replay_log and self._replay_log:
            yml_path = path.with_suffix(".yml")
            try:
                yml_path.write_text(self._format_style_chain(), encoding="utf-8")
                self._status.showMessage(f"Saved to: {path.name}  (+ style chain)")
            except OSError as exc:
                logger.warning("Could not auto-save replay log: %s", exc)
                self._status.showMessage(f"Saved to: {path.name}")
        else:
            self._status.showMessage(f"Saved to: {path.name}")

    # ------------------------------------------------------------------
    # Help dialogs (delegates to src.stylist.help_dialogs)
    # ------------------------------------------------------------------

    def _show_how_to_use(self) -> None:
        show_how_to_use(self)

    def _show_about_nst(self) -> None:
        show_about_nst(self)

    def _show_credits(self) -> None:
        show_credits(self)

    def _open_settings_dialog(self) -> None:
        dlg = SettingsDialog(settings=self._settings, parent=self)
        dlg.settings_changed.connect(self._on_settings_changed)
        dlg.exec()

    def _on_settings_changed(self, new_settings: AppSettings) -> None:
        self._settings = new_settings
        self._status.showMessage("Settings saved.")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _pil_to_pixmap(image: PILImage) -> QPixmap:
        """Convert a PIL RGB Image to a QPixmap."""
        from PIL.ImageQt import ImageQt
        qt_image = ImageQt(image.convert("RGB"))
        return QPixmap.fromImage(qt_image)
