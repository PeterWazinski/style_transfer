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
import sys
from pathlib import Path
from typing import Optional

from PIL.Image import Image as PILImage
from PySide6.QtGui import QAction, QPixmap
from PySide6.QtCore import Qt, QEventLoop
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QDockWidget,
    QFileDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressDialog,
    QPushButton,
    QScrollArea,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from src.core.engine import StyleTransferEngine
from src.core.models import StyleModel
from src.core.photo_manager import PhotoManager, UnsupportedFormatError
from src.core.registry import StyleRegistry
from src.core.settings import AppSettings
from src.stylist.apply_worker import ApplyWorker
from src.stylist.photo_canvas import PhotoCanvasView
from src.stylist.settings_dialog import SettingsDialog
from src.stylist.style_gallery import StyleGalleryView

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
        self._styled_photo_input: Optional[PILImage] = None  # source that produced _styled_photo

        self.setWindowTitle("Peter's Picture Stylist")
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

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_style_selected(self, style: StyleModel) -> None:
        self.canvas.set_active_style(style.id)
        self._status.showMessage(f"Style selected: {style.name}")
        # Preload ONNX if not already loaded
        if not self._engine.is_loaded(style.id):
            # model_path is stored as a str relative to the project root
            project_root: Path = (
                Path(sys.executable).parent
                if getattr(sys, "frozen", False)
                else Path(__file__).parent.parent.parent
            )
            model_path: Path = style.model_path_resolved(project_root)
            try:
                self._engine.load_model(
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
            image = self._photo_manager.load(self._current_photo_path, max_megapixels=self._settings.max_megapixels)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Load Error", str(exc))
            return
        self._current_photo = image
        self._styled_photo = None
        self._styled_photo_input = None
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
            image = self._photo_manager.load(path, max_megapixels=self._settings.max_megapixels)
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
    # Progress dialog helpers
    # ------------------------------------------------------------------

    def _create_progress_dialog(self, label: str = "Processing tiles\u2026") -> QProgressDialog:
        """Return a modal :class:`QProgressDialog` centred over this window.

        A *Cancel* button is shown; it requests interruption of the worker
        thread (wired in :meth:`_run_apply_worker`).  The dialog only becomes
        visible after 400 ms so it does not flash for tiny images.
        """
        dlg = QProgressDialog(label, "Cancel", 0, 100, self)
        dlg.setWindowTitle("Applying Style")
        dlg.setWindowModality(Qt.WindowModality.WindowModal)
        dlg.setMinimumDuration(400)
        dlg.setAutoClose(False)
        dlg.setAutoReset(False)
        dlg.setValue(0)
        return dlg

    def _run_apply_worker(
        self,
        source: PILImage,
        style_id: str,
        strength: float,
        dlg: QProgressDialog,
    ) -> PILImage | None:
        """Run :class:`ApplyWorker` in a background thread and wait for it.

        A local :class:`QEventLoop` is executed while the worker is running so
        the main event loop stays alive (the dialog repaints, the Cancel button
        responds).  Returns the styled :class:`PIL.Image` on success, or
        ``None`` on cancellation or error.  Errors are shown in a
        :class:`QMessageBox` before returning.
        """
        worker = ApplyWorker(
            engine=self._engine,
            source=source,
            style_id=style_id,
            strength=strength,
            tile_size=self._settings.tile_size,
            overlap=self._settings.overlap,
            use_float16=self._settings.use_float16,
        )

        result_holder: list[PILImage | None] = [None]
        error_holder: list[str | None] = [None]
        cancelled_holder: list[bool] = [False]
        loop = QEventLoop()

        def _on_progress(done: int, total: int) -> None:
            if total > 0:
                dlg.setValue(int(done / total * 100))

        def _on_finished(img: PILImage) -> None:
            result_holder[0] = img
            loop.quit()

        def _on_error(msg: str) -> None:
            error_holder[0] = msg
            loop.quit()

        def _on_cancelled() -> None:
            cancelled_holder[0] = True
            loop.quit()

        worker.progress.connect(_on_progress)
        worker.finished.connect(_on_finished)
        worker.error.connect(_on_error)
        worker.cancelled.connect(_on_cancelled)
        # Wire Cancel button → request interruption in the worker thread.
        dlg.canceled.connect(worker.requestInterruption)

        worker.start()
        loop.exec()
        worker.wait()   # ensure thread has fully exited before we continue

        if cancelled_holder[0]:
            return None
        if error_holder[0]:
            QMessageBox.critical(self, "Apply Error", error_holder[0])
            return None
        return result_holder[0]

    # ------------------------------------------------------------------
    # Apply / Re-Apply
    # ------------------------------------------------------------------

    def _reapply_style(self, style_id: str, strength: float) -> None:
        """Apply *style_id* to the already-styled photo (chain styles)."""
        source_photo = self._styled_photo
        if source_photo is None:
            return
        self._status.showMessage("Re-applying style\u2026")
        self.canvas.apply_button.setEnabled(False)
        self.canvas.reapply_button.setEnabled(False)
        QApplication.setOverrideCursor(Qt.WaitCursor)  # type: ignore[attr-defined]
        dlg = self._create_progress_dialog("Re-applying style\u2026")
        try:
            result = self._run_apply_worker(source_photo, style_id, strength, dlg)
        finally:
            dlg.close()
            QApplication.restoreOverrideCursor()
            self.canvas.apply_button.setEnabled(True)
            self.canvas.reapply_button.setEnabled(True)
        if result is None:
            self._status.showMessage("Re-apply cancelled." if dlg.wasCanceled() else "Error during style transfer.")
            return
        # Show the previous styled result on the left for comparison
        self.canvas.split_view.set_original_pixmap(self._pil_to_pixmap(source_photo))
        self._styled_photo_input = source_photo   # the input to this new chain step
        self._styled_photo = result
        self.canvas.set_styled(self._pil_to_pixmap(result))
        self._save_action.setEnabled(True)
        self._status.showMessage("Style re-applied.")

    def _reapply_style_strength(self, style_id: str, strength: float) -> None:
        """Re-run the current chain step with a new strength.

        Unlike :meth:`_reapply_style` (Re-Apply button) this does *not* advance
        the chain: the left pane keeps showing p_{n-1} and the right pane is
        updated with the new result.  ``_styled_photo_input`` (p_{n-1}) is
        re-used as the source so the slider never "eats" a chain step.
        """
        source = self._styled_photo_input
        if source is None:
            # Fallback: no recorded input yet — treat like a normal apply
            self._apply_style(style_id, strength)
            return
        self._status.showMessage("Adjusting strength\u2026")
        self.canvas.apply_button.setEnabled(False)
        self.canvas.reapply_button.setEnabled(False)
        QApplication.setOverrideCursor(Qt.WaitCursor)  # type: ignore[attr-defined]
        dlg = self._create_progress_dialog("Adjusting strength\u2026")
        try:
            result = self._run_apply_worker(source, style_id, strength, dlg)
        finally:
            dlg.close()
            QApplication.restoreOverrideCursor()
            self.canvas.apply_button.setEnabled(True)
            self.canvas.reapply_button.setEnabled(True)
        if result is None:
            self._status.showMessage("Adjustment cancelled." if dlg.wasCanceled() else "Error during style transfer.")
            return
        # Keep the left pane unchanged — only update the right pane & buffer
        self._styled_photo = result
        self.canvas.set_styled(self._pil_to_pixmap(result))
        self._save_action.setEnabled(True)
        self._status.showMessage("Strength adjusted.")

    def _apply_style(self, style_id: str, strength: float) -> None:
        if self._current_photo is None:
            return
        self._status.showMessage("Applying style\u2026")
        self.canvas.apply_button.setEnabled(False)
        QApplication.setOverrideCursor(Qt.WaitCursor)  # type: ignore[attr-defined]
        dlg = self._create_progress_dialog()
        try:
            result = self._run_apply_worker(self._current_photo, style_id, strength, dlg)
        finally:
            dlg.close()
            QApplication.restoreOverrideCursor()
            self.canvas.apply_button.setEnabled(True)
        if result is None:
            self._status.showMessage("Apply cancelled." if dlg.wasCanceled() else "Error during style transfer.")
            return
        self._styled_photo_input = self._current_photo   # record what fed this result
        self._styled_photo = result
        self.canvas.set_styled(self._pil_to_pixmap(result))
        self._save_action.setEnabled(True)
        self._status.showMessage("Style applied.")

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
            self._photo_manager.save(self._styled_photo, path)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Save Error", str(exc))
            return
        self._settings.last_save_dir = str(path.parent)
        self._settings.save()
        self._status.showMessage(f"Saved to: {path.name}")

    def _show_link_dialog(self, title: str, html: str) -> None:
        """Show an informational dialog that supports clickable hyperlinks."""
        dlg = QDialog(self)
        dlg.setWindowTitle(title)
        dlg.setMinimumWidth(520)

        label = QLabel(html)
        label.setWordWrap(True)
        label.setOpenExternalLinks(True)
        label.setTextFormat(Qt.RichText)  # type: ignore[attr-defined]
        label.setContentsMargins(4, 4, 4, 4)

        scroll = QScrollArea()
        scroll.setWidget(label)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)

        ok_btn = QPushButton("OK")
        ok_btn.setFixedWidth(80)
        ok_btn.clicked.connect(dlg.accept)

        layout = QVBoxLayout(dlg)
        layout.addWidget(scroll)
        layout.addWidget(ok_btn, alignment=Qt.AlignmentFlag.AlignRight)

        dlg.exec()

    def _show_about_nst(self) -> None:
        self._show_link_dialog(
            "About Neural Style Transfer",
            "<b>How Neural Style Transfer works</b><br><br>"
            "Neural Style Transfer (NST) applies the visual texture of a <i>style image</i> "
            "(e.g. a painting) to your <i>content photo</i> while preserving its "
            "structure and shapes.<br><br>"
            "<b>Feed-forward network (Johnson et al., 2016)</b><br>"
            "Unlike the original iterative optimisation, this app uses a lightweight "
            "convolutional network trained specifically for each style. Once trained, "
            "a single forward pass transforms any photo in milliseconds — "
            "no per-image optimisation required.<br><br>"
            "<b>Tiled inference</b><br>"
            "To handle large photos without running out of GPU memory, the image is "
            "divided into overlapping tiles, each processed independently, "
            "then blended back together seamlessly.<br><br>"
            "<b>Strength slider</b><br>"
            "Blends the styled result with the original photo "
            "(0&nbsp;% = original, 100&nbsp;% = fully styled). "
            "Tile size and overlap can be tuned in <i>File &#8594; Settings</i>.<br><br>"
            "<b>References</b><br>"
            "&#8226; Gatys et al. (2015) &mdash; "
            "<a href='https://arxiv.org/pdf/1508.06576'>A Neural Algorithm of Artistic Style</a> "
            "&mdash; the original NST paper using iterative optimisation.<br>"
            "&#8226; Johnson et al. (2016) &mdash; "
            "<a href='https://arxiv.org/pdf/1603.08155'>Perceptual Losses for Real-Time Style Transfer and Super-Resolution</a> "
            "&mdash; the feed-forward network used in this app.<br>"
            "&#8226; Kaggle notebook &mdash; "
            "<a href='https://www.kaggle.com/code/yashchoudhary/fast-neural-style-transfer'>Fast Neural Style Transfer</a> "
            "by Yash Choudhary.",
        )

    def _show_credits(self) -> None:
        self._show_link_dialog(
            "Credits",
            "<b>Peter's Picture Stylist</b><br><br>"
            "Pretrained ONNX models courtesy of:<br>"
            "&nbsp;&nbsp;<em>yakhyo/fast-neural-style-transfer</em> (MIT)<br>"
            "&nbsp;&nbsp;<em>igreat/fast-style-transfer</em> (MIT)<br><br>"
            "Training infrastructure:<br>"
            "&nbsp;&nbsp;<b>Kaggle</b> &mdash; free GPU compute (T4 x1) "
            "used to train new styles.<br><br>"
            "Built with Python, PySide6, and ONNX Runtime.",
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
