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
    tensor_layout: str = "nchw",
) -> None:
    """Run *onnx_path* on *content_image* and save a square thumbnail to *preview_path*.

    Args:
        onnx_path:      Path to the exported ``.onnx`` model.
        preview_path:   Destination path for the generated thumbnail (JPEG).
        content_image:  Content photo to stylise for the preview.
        size:           Edge length of the output thumbnail in pixels.
        tensor_layout:  ``"nchw"`` (NST default) or ``"nhwc_tanh"`` (AnimeGANv3).

    The function is a no-op when *onnx_path* does not exist (e.g. when called
    in a dry-run flow before training has finished).
    """
    if not onnx_path.exists():
        return

    import numpy as np  # noqa: PLC0415
    import onnxruntime as ort  # noqa: PLC0415
    from PIL import Image  # noqa: PLC0415

    # Load content image
    img = Image.open(content_image).convert("RGB").resize((size, size))

    # Run ONNX inference
    sess = ort.InferenceSession(
        str(onnx_path),
        providers=["DmlExecutionProvider", "CPUExecutionProvider"],
    )

    if tensor_layout == "nhwc_tanh":
        # NHWC [1, H, W, 3], range [-1, 1]  (AnimeGANv3 / TF models)
        arr = np.array(img, dtype=np.float32)[np.newaxis] / 127.5 - 1.0  # 1HWC
        out_raw = sess.run(None, {sess.get_inputs()[0].name: arr})[0]
        out: np.ndarray = np.asarray(out_raw)
        out = np.clip((out[0] + 1.0) / 2.0 * 255.0, 0, 255).astype(np.uint8)  # HWC
    elif tensor_layout == "nchw_tanh":
        # NCHW [1, 3, H, W], range [-1, 1]  (CycleGAN / PyTorch models)
        arr = np.array(img, dtype=np.float32).transpose(2, 0, 1)[np.newaxis]  # 1CHW
        arr = arr / 127.5 - 1.0
        out_raw = sess.run(None, {sess.get_inputs()[0].name: arr})[0]
        out = np.asarray(out_raw)
        out = np.clip((out[0].transpose(1, 2, 0) + 1.0) / 2.0 * 255.0, 0, 255).astype(np.uint8)  # HWC
    else:
        # NCHW [1, 3, H, W], range [0, 255]  (NST TransformerNet)
        arr = np.array(img, dtype=np.float32).transpose(2, 0, 1)[np.newaxis]  # 1CHW
        out_raw = sess.run(None, {sess.get_inputs()[0].name: arr})[0]
        out = np.asarray(out_raw)
        out = np.clip(out[0].transpose(1, 2, 0), 0, 255).astype(np.uint8)  # HWC

    preview_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(out).save(str(preview_path), quality=85)
