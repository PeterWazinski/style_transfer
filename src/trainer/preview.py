"""Preview generation for trained ONNX style models.

Provides :func:`generate_preview`, extracted from ``scripts/setup_models.py``
so that :mod:`main_style_trainer` can call it without duplicating code.

``scripts/setup_models.py`` is intentionally left unchanged (per design
decision D6 -- it is a dev bootstrap and owns its own model definitions).
"""
from __future__ import annotations

from pathlib import Path


def generate_preview(
    onnx_path: Path,
    preview_path: Path,
    content_image: Path,
    size: int = 256,
) -> None:
    """Run *onnx_path* on *content_image* and save a square thumbnail to *preview_path*.

    Args:
        onnx_path:     Path to the exported ``.onnx`` model.
        preview_path:  Destination path for the generated thumbnail (JPEG).
        content_image: Content photo to stylise for the preview.
        size:          Edge length of the output thumbnail in pixels.

    The function is a no-op when *onnx_path* does not exist (e.g. when called
    in a dry-run flow before training has finished).
    """
    if not onnx_path.exists():
        return

    import numpy as np  # noqa: PLC0415
    import onnxruntime as ort  # noqa: PLC0415
    from PIL import Image  # noqa: PLC0415

    # Load and pre-process content image
    img = Image.open(content_image).convert("RGB").resize((size, size))
    arr = np.array(img, dtype=np.float32)          # HWC, [0, 255]
    arr = arr.transpose(2, 0, 1)[np.newaxis]       # 1CHW, [0, 255]

    # Run ONNX inference
    sess = ort.InferenceSession(
        str(onnx_path),
        providers=["DmlExecutionProvider", "CPUExecutionProvider"],
    )
    out = sess.run(None, {sess.get_inputs()[0].name: arr})[0]  # 1CHW

    # Post-process and save
    out = np.clip(out[0].transpose(1, 2, 0), 0, 255).astype(np.uint8)  # HWC
    preview_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(out).save(str(preview_path), quality=85)
