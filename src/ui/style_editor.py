"""StyleEditorDialog — Add or edit a custom style.

Fields
------
* **Name** (required) — unique slug used as the style ID when creating.
* **Description** — free-text description.
* **Author** — optional attribution.
* **Reference images** — drag-drop or browse JPEG/PNG files used for training.
* **Advanced** (collapsible) — training hyperparameters.

Emits :attr:`style_saved` ``(StyleModel)`` when the user confirms.
"""
from __future__ import annotations

import logging
import re
import uuid
from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QPushButton,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from src.core.models import StyleModel

logger: logging.Logger = logging.getLogger(__name__)


class StyleEditorDialog(QDialog):
    """Dialog for creating or editing a custom style.

    Args:
        style:  Existing :class:`StyleModel` to edit, or *None* to create a new one.
        parent: Optional parent widget.

    Signals:
        style_saved(StyleModel): Emitted when the user clicks OK and validation passes.
    """

    style_saved: Signal = Signal(object)   # payload: StyleModel

    def __init__(
        self,
        style: Optional[StyleModel] = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._existing: StyleModel | None = style
        self.setWindowTitle("Edit Style" if style else "Add Style")
        self.setMinimumWidth(460)
        self._build_ui()
        if style:
            self._populate(style)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)

        form = QFormLayout()
        form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)

        # --- Name ---
        self.name_edit = QLineEdit(self)
        self.name_edit.setPlaceholderText("e.g. Van Gogh")
        form.addRow("Name *", self.name_edit)

        self._name_error = QLabel("Name is required.", self)
        self._name_error.setStyleSheet("color: red;")
        self._name_error.setVisible(False)
        form.addRow("", self._name_error)

        # --- Description ---
        self.description_edit = QTextEdit(self)
        self.description_edit.setFixedHeight(60)
        form.addRow("Description", self.description_edit)

        # --- Author ---
        self.author_edit = QLineEdit(self)
        form.addRow("Author", self.author_edit)

        root.addLayout(form)

        # --- Reference images ---
        ref_box = QGroupBox("Reference images (style source)", self)
        ref_layout = QVBoxLayout(ref_box)
        self.ref_list = QListWidget(self)
        self.ref_list.setFixedHeight(80)
        ref_layout.addWidget(self.ref_list)
        browse_row = QHBoxLayout()
        self.browse_button = QPushButton("Browse…", self)
        self.remove_ref_button = QPushButton("Remove", self)
        browse_row.addWidget(self.browse_button)
        browse_row.addWidget(self.remove_ref_button)
        browse_row.addStretch()
        ref_layout.addLayout(browse_row)
        root.addWidget(ref_box)

        # --- Advanced hyperparameters (collapsible via QGroupBox checkable) ---
        self.advanced_box = QGroupBox("Advanced training settings", self)
        self.advanced_box.setCheckable(True)
        self.advanced_box.setChecked(False)
        adv_form = QFormLayout(self.advanced_box)

        self.epochs_spin = QSpinBox(self)
        self.epochs_spin.setRange(1, 20)
        self.epochs_spin.setValue(2)
        adv_form.addRow("Epochs", self.epochs_spin)

        self.content_weight_spin = QDoubleSpinBox(self)
        self.content_weight_spin.setRange(1e3, 1e9)
        self.content_weight_spin.setValue(1e5)
        self.content_weight_spin.setDecimals(0)
        adv_form.addRow("Content weight", self.content_weight_spin)

        self.style_weight_spin = QDoubleSpinBox(self)
        self.style_weight_spin.setRange(1e5, 1e12)
        self.style_weight_spin.setValue(1e8)
        self.style_weight_spin.setDecimals(0)
        adv_form.addRow("Style weight", self.style_weight_spin)

        root.addWidget(self.advanced_box)

        # --- Dialog buttons ---
        self._buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
            parent=self,
        )
        root.addWidget(self._buttons)

        # --- Connections ---
        self.browse_button.clicked.connect(self._browse_reference_images)
        self.remove_ref_button.clicked.connect(self._remove_selected_ref)
        self._buttons.accepted.connect(self._on_accept)
        self._buttons.rejected.connect(self.reject)
        self.name_edit.textChanged.connect(lambda _: self._name_error.setVisible(False))

    # ------------------------------------------------------------------
    # Population (edit mode)
    # ------------------------------------------------------------------

    def _populate(self, style: StyleModel) -> None:
        self.name_edit.setText(style.name)
        self.description_edit.setPlainText(style.description)
        self.author_edit.setText(style.author)
        for p in style.source_images:
            self.ref_list.addItem(str(p))
        if style.training_config:
            cfg = style.training_config
            self.epochs_spin.setValue(int(cfg.get("epochs", 2)))
            self.content_weight_spin.setValue(float(cfg.get("content_weight", 1e5)))
            self.style_weight_spin.setValue(float(cfg.get("style_weight", 1e8)))

    # ------------------------------------------------------------------
    # Internal slots
    # ------------------------------------------------------------------

    def _browse_reference_images(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Select reference image(s)",
            "",
            "Images (*.jpg *.jpeg *.png)",
        )
        for p in paths:
            self.ref_list.addItem(p)

    def _remove_selected_ref(self) -> None:
        for item in self.ref_list.selectedItems():
            self.ref_list.takeItem(self.ref_list.row(item))

    def _on_accept(self) -> None:
        name = self.name_edit.text().strip()
        if not name:
            self._name_error.setVisible(True)
            return

        # Build slug: lowercase + replace spaces with hyphens
        if self._existing:
            style_id = self._existing.id
        else:
            slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
            style_id = f"{slug}-{uuid.uuid4().hex[:6]}"

        ref_images: list[Path] = [
            Path(self.ref_list.item(i).text())
            for i in range(self.ref_list.count())
        ]

        training_config: dict[str, object] = {
            "epochs": self.epochs_spin.value(),
            "content_weight": self.content_weight_spin.value(),
            "style_weight": self.style_weight_spin.value(),
        } if self.advanced_box.isChecked() else {}

        model = StyleModel(
            id=style_id,
            name=name,
            model_path=self._existing.model_path if self._existing else f"styles/{style_id}/model.onnx",
            preview_path=self._existing.preview_path if self._existing else f"styles/{style_id}/preview.jpg",
            description=self.description_edit.toPlainText().strip(),
            author=self.author_edit.text().strip(),
            source_images=[str(p) for p in ref_images],
            is_builtin=self._existing.is_builtin if self._existing else False,
            training_config=training_config or None,
        )

        self.style_saved.emit(model)
        self.accept()
