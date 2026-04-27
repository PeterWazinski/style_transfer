"""Shared test helper utilities.

Centralises helpers that were previously duplicated across multiple test
modules:
  - mock ONNX inference session factory (was copy-pasted in 4 files)
  - image file writers used to set up test fixtures
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
from PIL import Image


def make_mock_session(
    output_colour: tuple[int, int, int] = (128, 64, 192),
    raise_oom: bool = False,
) -> MagicMock:
    """Return a mock ``ort.InferenceSession``.

    Args:
        output_colour: The solid RGB colour the session outputs for every
                       pixel.  Ignored when *raise_oom* is ``True``.
        raise_oom:     When ``True``, the ``run()`` call raises
                       :exc:`MemoryError` to simulate an out-of-memory tile.
    """
    session = MagicMock()
    inp = MagicMock()
    inp.name = "input"
    session.get_inputs.return_value = [inp]

    def _run(
        output_names: list[str], feed: dict[str, np.ndarray]
    ) -> list[np.ndarray]:
        if raise_oom:
            raise MemoryError("Simulated OOM")
        tensor = feed["input"]  # [1, 3, H, W]
        h, w = tensor.shape[2], tensor.shape[3]
        rgb = np.full((1, 3, h, w), 0.0, dtype=np.float32)
        rgb[0, 0, :, :] = float(output_colour[0])
        rgb[0, 1, :, :] = float(output_colour[1])
        rgb[0, 2, :, :] = float(output_colour[2])
        return [rgb]

    session.run.side_effect = _run
    return session


def make_mock_session_nhwc(
    output_colour: tuple[int, int, int] = (128, 64, 192),
) -> MagicMock:
    """Return a mock ``ort.InferenceSession`` that mimics NHWC models (e.g. AnimeGANv3).

    Expects input ``[1, H, W, 3]`` float32 in ``[-1, 1]`` and returns the same
    shape with the requested solid colour de-normalised to ``[-1, 1]``.
    """
    session = MagicMock()
    inp = MagicMock()
    inp.name = "input"
    session.get_inputs.return_value = [inp]

    def _run(
        output_names: list[str], feed: dict[str, np.ndarray]
    ) -> list[np.ndarray]:
        tensor = feed["input"]  # [1, H, W, 3]
        h, w = tensor.shape[1], tensor.shape[2]
        # Build solid-colour output in [-1, 1] range
        out = np.full((1, h, w, 3), 0.0, dtype=np.float32)
        out[0, :, :, 0] = output_colour[0] / 127.5 - 1.0
        out[0, :, :, 1] = output_colour[1] / 127.5 - 1.0
        out[0, :, :, 2] = output_colour[2] / 127.5 - 1.0
        return [out]

    session.run.side_effect = _run
    return session


def save_jpeg(path: Path, size: tuple[int, int] = (64, 64)) -> Path:
    """Write a random RGB JPEG to *path* and return *path*.

    Args:
        path: Destination file path (parent directory must exist).
        size: ``(width, height)`` of the generated image.
    """
    arr = np.random.randint(0, 255, (size[1], size[0], 3), dtype=np.uint8)
    Image.fromarray(arr).save(path, format="JPEG", quality=95)
    return path


def save_png(path: Path, size: tuple[int, int] = (64, 64)) -> Path:
    """Write a random RGB PNG to *path* and return *path*.

    Args:
        path: Destination file path (parent directory must exist).
        size: ``(width, height)`` of the generated image.
    """
    arr = np.random.randint(0, 255, (size[1], size[0], 3), dtype=np.uint8)
    Image.fromarray(arr).save(path, format="PNG")
    return path
