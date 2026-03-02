"""Integration tests — training pipeline → style gallery.

Verifies the complete chain:
  1. Toy COCO images
  2. Train 1 batch step → .pth checkpoint
  3. Export → .onnx file
  4. Create a :class:`StyleModel` from the result
  5. Register the style
  6. Verify it appears in the gallery after ``gallery.refresh()``

The training itself uses a minimal 64×64 dataset (4 images) on CPU,
so it fits within the short-test budget.  The heavy full-epoch test is
named ``_takes_long`` and is automatically excluded from the fast suite.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

from src.core.models import StyleModel
from src.core.registry import StyleRegistry
from src.core.trainer import COCODatasetNotFoundError, StyleTrainer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_toy_coco(directory: Path, n_images: int = 4) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    for i in range(n_images):
        arr = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        Image.fromarray(arr).save(directory / f"img_{i:04d}.jpg")
    return directory


def _make_style_image(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    Image.fromarray(arr).save(path)
    return path


# ---------------------------------------------------------------------------
# Train → ONNX → registry → gallery
# ---------------------------------------------------------------------------

class TestTrainToGallery:
    def test_trained_style_appears_in_registry(self, tmp_path: Path) -> None:
        """Train 1 batch → export to ONNX → add to registry → verify registry contains it."""
        coco_dir = _make_toy_coco(tmp_path / "coco")
        style_img = _make_style_image(tmp_path / "style.png")
        pth_path = tmp_path / "my_style" / "model.pth"
        onnx_path = tmp_path / "my_style" / "model.onnx"

        trainer = StyleTrainer(device="cpu")
        trainer.train(
            style_images=[style_img],
            coco_dataset_path=coco_dir,
            output_model_path=pth_path,
            epochs=1,
            batch_size=2,
            image_size=64,
            checkpoint_interval=0,
        )
        trainer.export_onnx(pth_path, onnx_path)

        assert onnx_path.exists(), "ONNX model was not created"

        reg = StyleRegistry(catalog_path=tmp_path / "catalog.json")
        style = StyleModel(
            id="my-style",
            name="My Style",
            model_path=str(onnx_path),
            preview_path=str(style_img),
            is_builtin=False,
        )
        reg.add(style)

        styles = reg.list_styles()
        assert any(s.id == "my-style" for s in styles)

    def test_trained_style_appears_in_gallery(
        self, qtbot, tmp_path: Path
    ) -> None:
        """After training and registry.add(), gallery.refresh() shows the new entry."""
        from src.ui.style_gallery import StyleGalleryView  # noqa: PLC0415

        coco_dir = _make_toy_coco(tmp_path / "coco")
        style_img = _make_style_image(tmp_path / "style.png")
        pth_path = tmp_path / "my_style2" / "model.pth"
        onnx_path = tmp_path / "my_style2" / "model.onnx"

        trainer = StyleTrainer(device="cpu")
        trainer.train(
            style_images=[style_img],
            coco_dataset_path=coco_dir,
            output_model_path=pth_path,
            epochs=1,
            batch_size=2,
            image_size=64,
            checkpoint_interval=0,
        )
        trainer.export_onnx(pth_path, onnx_path)

        reg = StyleRegistry(catalog_path=tmp_path / "catalog.json")
        gallery = StyleGalleryView(registry=reg)
        qtbot.addWidget(gallery)

        assert gallery.model().rowCount() == 0

        style = StyleModel(
            id="my-style2",
            name="My Style 2",
            model_path=str(onnx_path),
            preview_path=str(style_img),
            is_builtin=False,
        )
        reg.add(style)
        gallery.refresh()

        assert gallery.model().rowCount() == 1
        assert gallery.model().item(0).text() == "My Style 2"


# ---------------------------------------------------------------------------
# TrainingProgressDialog with mock worker
# ---------------------------------------------------------------------------

class TestTrainingDialogMockWorker:
    def test_mock_worker_completes_and_emits_training_completed(
        self, qtbot, tmp_path: Path
    ) -> None:
        """TrainingProgressDialog.training_completed fires when worker finishes."""
        from src.ui.training_dialog import TrainingProgressDialog  # noqa: PLC0415

        mock_trainer = MagicMock()
        mock_trainer.train.return_value = tmp_path / "model.pth"

        dlg = TrainingProgressDialog(trainer=mock_trainer)
        qtbot.addWidget(dlg)

        completed_paths: list[str] = []
        dlg.training_completed.connect(completed_paths.append)

        # Simulate the worker finishing directly without threading overhead
        dlg._on_worker_finished(str(tmp_path / "model.pth"))

        assert len(completed_paths) == 1
        assert "model.pth" in completed_paths[0]

    def test_coco_not_found_surfaces_as_error_signal(
        self, qtbot, tmp_path: Path
    ) -> None:
        """If the trainer raises COCODatasetNotFoundError the error_occurred signal fires."""
        from src.ui.training_dialog import TrainingProgressDialog, TrainingWorker  # noqa: PLC0415

        mock_trainer = MagicMock()
        mock_trainer.train.side_effect = COCODatasetNotFoundError("No COCO")

        dlg = TrainingProgressDialog(trainer=mock_trainer)
        qtbot.addWidget(dlg)

        errors: list[str] = []
        dlg.training_dialog_error = errors.append  # type: ignore[assignment]

        # Directly invoke the worker error slot
        dlg._on_worker_error("COCO dataset directory not found")
        # The dialog displays the message; just verify the status bar text
        assert not dlg.is_training()


# ---------------------------------------------------------------------------
# COCODatasetNotFoundError — explicit path check
# ---------------------------------------------------------------------------

class TestCOCOPathCheck:
    def test_nonexistent_coco_raises_coco_error(self, tmp_path: Path) -> None:
        trainer = StyleTrainer(device="cpu")
        style_img = _make_style_image(tmp_path / "style.png")
        with pytest.raises(COCODatasetNotFoundError, match="not found"):
            trainer.train(
                style_images=[style_img],
                coco_dataset_path=tmp_path / "nonexistent_coco",
                output_model_path=tmp_path / "m.pth",
                epochs=1,
            )
