"""SettingsDialog — user-editable application settings.

Exposes all :class:`~src.core.settings.AppSettings` fields as form controls.
Emits :attr:`settings_changed` with the updated :class:`~src.core.settings.AppSettings`
when the user clicks OK.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from src.core.settings import (
    DEFAULT_OVERLAP,
    DEFAULT_TILE_SIZE,
    MAX_MP_CHOICES,
    OVERLAP_CHOICES,
    PROVIDER_CHOICES,
    TILE_SIZE_CHOICES,
    AppSettings,
)

logger: logging.Logger = logging.getLogger(__name__)


class SettingsDialog(QDialog):
    """Dialog for editing :class:`AppSettings`.

    Args:
        settings: Current settings to populate the form with.
        parent:   Optional parent widget.

    Signals:
        settings_changed(AppSettings): Emitted when the user clicks OK.
    """

    settings_changed: Signal = Signal(object)  # payload: AppSettings

    def __init__(
        self,
        settings: Optional[AppSettings] = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._settings: AppSettings = settings or AppSettings()
        self.setWindowTitle("Settings")
        self.setMinimumWidth(420)
        self._build_ui()
        self._populate(self._settings)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _hint(self, text: str) -> QLabel:
        """Return a small grey hint label for display below a form row."""
        lbl = QLabel(f"<small style='color: black;'>{text}</small>", self)
        lbl.setWordWrap(True)
        return lbl

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)

        form = QFormLayout()
        form.setVerticalSpacing(2)

        # --- Tile size ---
        self.tile_size_combo = QComboBox(self)
        for v in TILE_SIZE_CHOICES:
            label = f"{v} px (default)" if v == DEFAULT_TILE_SIZE else f"{v} px"
            self.tile_size_combo.addItem(label, v)
        form.addRow("Tile size:", self.tile_size_combo)
        form.addRow("", self._hint(
            "Large photos are split into tiles of this size before processing. "
            "Bigger tiles produce fewer seams but require more GPU memory."
        ))

        # --- Overlap ---
        self.overlap_combo = QComboBox(self)
        for v in OVERLAP_CHOICES:
            label = f"{v} px (default)" if v == DEFAULT_OVERLAP else f"{v} px"
            self.overlap_combo.addItem(label, v)
        form.addRow("Tile overlap:", self.overlap_combo)
        form.addRow("", self._hint(
            "Adjacent tiles overlap by this many pixels to hide stitching artefacts. "
            "Increase if you see a grid pattern in the result."
        ))

        # --- Default output directory ---
        output_row = QHBoxLayout()
        self.output_dir_edit = QLineEdit(self)
        self.output_dir_edit.setPlaceholderText("(OS default)")
        self.browse_output_btn = QPushButton("Browse…", self)
        self.browse_output_btn.clicked.connect(self._browse_output_dir)
        output_row.addWidget(self.output_dir_edit, 1)
        output_row.addWidget(self.browse_output_btn)
        form.addRow("Default output folder:", output_row)
        form.addRow("", self._hint(
            "Pre-filled directory in the Save dialog. Leave empty to use the OS default."
        ))

        # --- Execution provider ---
        self.provider_combo = QComboBox(self)
        _PROVIDER_LABELS: dict[str, str] = {
            "auto":  "Auto (DML → CUDA → CPU)",
            "dml":   "DirectML (Intel Arc / AMD)",
            "cuda":  "CUDA (Nvidia)",
            "cpu":   "CPU only",
        }
        for key in PROVIDER_CHOICES:
            self.provider_combo.addItem(_PROVIDER_LABELS[key], key)
        form.addRow("Execution provider:", self.provider_combo)
        form.addRow("", self._hint(
            "Hardware back-end for ONNX inference. \"Auto\" tries the fastest "
            "available device (DirectML → CUDA → CPU) automatically."
        ))

        # --- Float16 ---
        self.float16_check = QCheckBox(
            "Float16 inference (faster on GPU/DML, slight quality trade-off)", self
        )
        form.addRow("", self.float16_check)
        form.addRow("", self._hint(
            "Halves the numeric precision of tensors sent to the GPU. "
            "Reduces VRAM usage and speeds up rendering. Has no effect on CPU-only runtimes."
        ))

        # --- Max megapixels ---
        self.max_mp_combo = QComboBox(self)
        _MP_LABELS: dict[float, str] = {
            8.0:  "8 MP  (3264 × 2448 — old phone)",
            12.0: "12 MP (4000 × 3000 — typical phone)",
            20.0: "20 MP (5000 × 4000 — default)",
            40.0: "40 MP (7248 × 5520 — high-res camera)",
            0.0:  "No limit  ⚠ may use a lot of RAM",
        }
        for v in MAX_MP_CHOICES:
            self.max_mp_combo.addItem(_MP_LABELS[v], v)
        form.addRow("Max photo size:", self.max_mp_combo)
        form.addRow("", self._hint(
            "Photos larger than this are automatically scaled down before inference. "
            "Keeps memory usage and processing time in check."
        ))

        root.addLayout(form)

        # --- Info label ---
        info = QLabel(
            "<small><i>Changes take effect on the next Apply.</i></small>", self
        )
        root.addWidget(info)

        # --- Buttons ---
        self._buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
            parent=self,
        )
        self._buttons.accepted.connect(self._on_accept)
        self._buttons.rejected.connect(self.reject)
        root.addWidget(self._buttons)

    # ------------------------------------------------------------------
    # Population
    # ------------------------------------------------------------------

    def _populate(self, settings: AppSettings) -> None:
        # Tile size
        idx = self.tile_size_combo.findData(settings.tile_size)
        if idx >= 0:
            self.tile_size_combo.setCurrentIndex(idx)

        # Overlap
        idx = self.overlap_combo.findData(settings.overlap)
        if idx >= 0:
            self.overlap_combo.setCurrentIndex(idx)

        # Output dir
        self.output_dir_edit.setText(settings.default_output_dir)

        # Provider
        idx = self.provider_combo.findData(settings.execution_provider)
        if idx >= 0:
            self.provider_combo.setCurrentIndex(idx)

        # Float16
        self.float16_check.setChecked(settings.use_float16)

        # Max megapixels
        idx = self.max_mp_combo.findData(settings.max_megapixels)
        if idx >= 0:
            self.max_mp_combo.setCurrentIndex(idx)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def current_settings(self) -> AppSettings:
        """Return an :class:`AppSettings` built from the current control values."""
        return AppSettings(
            tile_size=self.tile_size_combo.currentData(),
            overlap=self.overlap_combo.currentData(),
            default_output_dir=self.output_dir_edit.text().strip(),
            execution_provider=self.provider_combo.currentData(),
            use_float16=self.float16_check.isChecked(),
            max_megapixels=self.max_mp_combo.currentData(),
        )

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _browse_output_dir(self) -> None:
        start = self.output_dir_edit.text().strip() or str(Path.home())
        directory = QFileDialog.getExistingDirectory(
            self, "Select default output folder", start
        )
        if directory:
            self.output_dir_edit.setText(directory)

    def _on_accept(self) -> None:
        try:
            s = self.current_settings()
        except ValueError as exc:
            logger.warning("Invalid settings: %s", exc)
            return
        s.save()
        self.settings_changed.emit(s)
        self.accept()
