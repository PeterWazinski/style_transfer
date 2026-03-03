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
from pathlib import Path
from typing import Optional

from PIL.Image import Image as PILImage
from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QPixmap
from PySide6.QtWidgets import (
    QApplication,
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
from src.core.registry import StyleNotFoundError, StyleRegistry
from src.core.settings import AppSettings
from src.ui.photo_canvas import PhotoCanvasView
from src.ui.settings_dialog import SettingsDialog
from src.ui.style_editor import StyleEditorDialog
from src.ui.style_gallery import StyleGalleryView
from src.ui.training_dialog import TrainingProgressDialog

logger: logging.Logger = logging.getLogger(__name__)


class MainWindow(QMainWindow):
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
        # Keep private aliases for internal use.
        self._registry = registry
        self._engine = engine
        self._photo_manager = photo_manager
        self._settings: AppSettings = settings or AppSettings.load()
        self._current_photo: Optional[PILImage] = None
        self._current_photo_path: Optional[Path] = None
        self._styled_photo: Optional[PILImage] = None
        # Last-opened editor dialog — exposed for tests.
        self.style_editor_dialog: Optional[StyleEditorDialog] = None

        self.setWindowTitle("Style Transfer")
        self.resize(1200, 750)
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
        self.gallery = StyleGalleryView(self._registry, self)
        dock = QDockWidget("Styles", self)
        dock.setWidget(self.gallery)
        dock.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetMovable
            | QDockWidget.DockWidgetFeature.DockWidgetFloatable
        )
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, dock)

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

        file_menu.addSeparator()
        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # Styles menu
        styles_menu = mb.addMenu("Styles")
        add_style_action = QAction("Add Style…", self)
        add_style_action.triggered.connect(self._open_add_style_dialog)
        styles_menu.addAction(add_style_action)
        # Tools menu
        tools_menu = mb.addMenu("Tools")
        settings_action = QAction("Settings\u2026", self)
        settings_action.setShortcut("Ctrl+,")
        settings_action.triggered.connect(self._open_settings_dialog)
        tools_menu.addAction(settings_action)
        # Help menu
        help_menu = mb.addMenu("Help")
        about_action = QAction("About…", self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)

    # ------------------------------------------------------------------
    # Signal wiring
    # ------------------------------------------------------------------

    def _wire_signals(self) -> None:
        # Gallery → canvas
        self.gallery.style_selected.connect(self._on_style_selected)
        # Gallery → dialogs
        self.gallery.add_requested.connect(self._open_add_style_dialog)
        self.gallery.edit_requested.connect(self._open_edit_style_dialog)
        self.gallery.delete_requested.connect(self._delete_style)
        # Canvas → actions
        self.canvas.open_photo_requested.connect(self._open_photo)
        self.canvas.apply_requested.connect(self._apply_style)
        self.canvas.save_requested.connect(self._save_result)

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_style_selected(self, style: StyleModel) -> None:
        self.canvas.set_active_style(style.id)
        self._status.showMessage(f"Style selected: {style.name}")
        # Preload ONNX if not already loaded
        if not self._engine.is_loaded(style.id):
            # model_path is stored as a str relative to the project root
            project_root: Path = Path(__file__).parent.parent.parent
            model_path: Path = style.model_path_resolved(project_root)
            try:
                self._engine.load_model(style.id, model_path)
            except Exception as exc:  # noqa: BLE001
                logger.exception("Could not load model '%s' from %s", style.id, model_path)
                self._status.showMessage(f"Could not load model: {exc}")

    def _open_photo(self) -> None:
        path_str, _ = QFileDialog.getOpenFileName(
            self,
            "Open Photo",
            "",
            "Images (*.jpg *.jpeg *.png)",
        )
        if not path_str:
            return
        path = Path(path_str)
        try:
            image = self._photo_manager.load(path)
        except UnsupportedFormatError as exc:
            QMessageBox.warning(self, "Unsupported Format", str(exc))
            return
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Load Error", str(exc))
            return

        self._current_photo = image
        self._current_photo_path = path
        # Convert PIL Image → QPixmap for display
        pixmap = self._pil_to_pixmap(image)
        self.canvas.set_original(pixmap)
        self._status.showMessage(f"Opened: {path.name}  ({image.width}×{image.height})")

    def _apply_style(self, style_id: str, strength: float) -> None:
        if self._current_photo is None:
            return
        self._status.showMessage("Applying style…")
        self.canvas.apply_button.setEnabled(False)
        QApplication.setOverrideCursor(Qt.WaitCursor)  # type: ignore[attr-defined]
        QApplication.processEvents()  # flush cursor + status bar before blocking
        try:
            result = self._engine.apply(
                self._current_photo,
                style_id,
                strength=strength,
                tile_size=self._settings.tile_size,
                overlap=self._settings.overlap,
                use_float16=self._settings.use_float16,
            )
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Apply Error", str(exc))
            self._status.showMessage("Error during style transfer.")
            return
        finally:
            QApplication.restoreOverrideCursor()
            self.canvas.apply_button.setEnabled(True)
        self._styled_photo = result
        self.canvas.set_styled(self._pil_to_pixmap(result))
        self._save_action.setEnabled(True)
        self._status.showMessage("Style applied.")

    def _save_result(self) -> None:
        if self._styled_photo is None:
            return
        start_dir = self._settings.default_output_dir or ""
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
            self._photo_manager.save(self._styled_photo, path)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Save Error", str(exc))
            return
        self._status.showMessage(f"Saved to: {path.name}")

    def _open_add_style_dialog(self) -> None:
        self.style_editor_dialog = StyleEditorDialog(parent=self)
        self.style_editor_dialog.style_saved.connect(self._on_style_saved)
        self.style_editor_dialog.exec()

    def _open_edit_style_dialog(self, style_id: str) -> None:
        try:
            style = self._registry.get(style_id)
        except StyleNotFoundError:
            return
        self.style_editor_dialog = StyleEditorDialog(style=style, parent=self)
        self.style_editor_dialog.style_saved.connect(self._on_style_updated)
        self.style_editor_dialog.exec()

    def _delete_style(self, style_id: str) -> None:
        try:
            style = self._registry.get(style_id)
        except StyleNotFoundError:
            return
        answer = QMessageBox.question(
            self,
            "Delete Style",
            f"Delete style '{style.name}'?\nThis cannot be undone.",
        )
        if answer == QMessageBox.StandardButton.Yes:
            self._registry.delete(style_id)
            self.gallery.refresh()
            self._status.showMessage(f"Deleted style: {style.name}")

    def _on_style_saved(self, model: StyleModel) -> None:
        try:
            self._registry.add(model)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "Could not add style", str(exc))
            return
        self.gallery.refresh()
        self._status.showMessage(f"Style added: {model.name}")

    def _on_style_updated(self, model: StyleModel) -> None:
        try:
            self._registry.update(model)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "Could not update style", str(exc))
            return
        self.gallery.refresh()
        self._status.showMessage(f"Style updated: {model.name}")

    def _show_about(self) -> None:
        QMessageBox.about(
            self,
            "About Style Transfer",
            "<b>Fast Neural Style Transfer</b><br>"
            "Powered by Johnson et al. (2016) feed-forward network.<br><br>"
            "Pretrained ONNX models courtesy of<br>"
            "<em>yakhyo/fast-neural-style-transfer</em> (MIT)<br>"
            "<em>igreat/fast-style-transfer</em> (MIT)<br>",
        )

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
