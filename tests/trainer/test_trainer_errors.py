"""Tests for trainer-specific errors (torch required).

Runs only when ``torch`` is available -- the trainer package imports torch
at the module level, so these are skipped in a minimal Stylist-only install.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.trainer.style_trainer import COCODatasetNotFoundError, StyleTrainer


# ---------------------------------------------------------------------------
# COCODatasetNotFoundError
# ---------------------------------------------------------------------------

class TestCOCODatasetNotFoundError:
    def test_missing_coco_dir_raises(self, tmp_path: Path) -> None:
        trainer = StyleTrainer(device="cpu")
        style_path = tmp_path / "style.png"
        Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8)).save(style_path)

        with pytest.raises(COCODatasetNotFoundError, match="not found"):
            trainer.train(
                style_images=[style_path],
                coco_dataset_path=tmp_path / "no_such_coco",
                output_model_path=tmp_path / "model.pth",
                epochs=1,
                batch_size=1,
                image_size=64,
            )
