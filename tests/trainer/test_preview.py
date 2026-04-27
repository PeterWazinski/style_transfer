"""Unit tests for src.trainer.preview.generate_preview."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from PIL import Image

from tests.helpers import make_mock_session, make_mock_session_nhwc


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def content_image(tmp_path: Path) -> Path:
    """A small solid-colour PNG used as the content image."""
    img = Image.fromarray(np.full((64, 64, 3), 100, dtype=np.uint8))
    p = tmp_path / "content.png"
    img.save(str(p))
    return p


@pytest.fixture()
def fake_onnx(tmp_path: Path) -> Path:
    """An empty file that passes the exists() check."""
    p = tmp_path / "model.onnx"
    p.write_bytes(b"")
    return p


# ---------------------------------------------------------------------------
# No-op when model file is missing
# ---------------------------------------------------------------------------

def test_noop_when_onnx_missing(tmp_path: Path, content_image: Path) -> None:
    from src.trainer.preview import generate_preview

    preview_path = tmp_path / "preview.jpg"
    generate_preview(
        onnx_path=tmp_path / "nonexistent.onnx",
        preview_path=preview_path,
        content_image=content_image,
    )
    assert not preview_path.exists()


# ---------------------------------------------------------------------------
# NCHW layout (default)
# ---------------------------------------------------------------------------

def test_nchw_creates_jpeg(tmp_path: Path, fake_onnx: Path, content_image: Path) -> None:
    from src.trainer.preview import generate_preview

    preview_path = tmp_path / "preview.jpg"
    sess = make_mock_session(output_colour=(200, 100, 50))

    with patch("onnxruntime.InferenceSession", return_value=sess):
        generate_preview(
            onnx_path=fake_onnx,
            preview_path=preview_path,
            content_image=content_image,
            size=64,
            tensor_layout="nchw",
        )

    assert preview_path.exists()


def test_nchw_output_size(tmp_path: Path, fake_onnx: Path, content_image: Path) -> None:
    from src.trainer.preview import generate_preview

    preview_path = tmp_path / "preview.jpg"
    sess = make_mock_session()

    with patch("onnxruntime.InferenceSession", return_value=sess):
        generate_preview(
            onnx_path=fake_onnx,
            preview_path=preview_path,
            content_image=content_image,
            size=64,
        )

    img = Image.open(str(preview_path))
    assert img.size == (64, 64)


def test_nchw_colour_correct(tmp_path: Path, fake_onnx: Path, content_image: Path) -> None:
    from src.trainer.preview import generate_preview

    preview_path = tmp_path / "preview.jpg"
    colour = (200, 100, 50)
    sess = make_mock_session(output_colour=colour)

    with patch("onnxruntime.InferenceSession", return_value=sess):
        generate_preview(
            onnx_path=fake_onnx,
            preview_path=preview_path,
            content_image=content_image,
            size=64,
        )

    arr = np.array(Image.open(str(preview_path)))
    # JPEG compression introduces small errors — allow ±5
    assert abs(int(arr[0, 0, 0]) - colour[0]) <= 5
    assert abs(int(arr[0, 0, 1]) - colour[1]) <= 5
    assert abs(int(arr[0, 0, 2]) - colour[2]) <= 5


def test_nchw_input_shape_is_1chw(tmp_path: Path, fake_onnx: Path, content_image: Path) -> None:
    from src.trainer.preview import generate_preview

    preview_path = tmp_path / "preview.jpg"
    sess = make_mock_session()
    captured: list[np.ndarray] = []

    original_run = sess.run.side_effect

    def _capture(output_names, feed):
        captured.append(feed["input"])
        return original_run(output_names, feed)

    sess.run.side_effect = _capture

    with patch("onnxruntime.InferenceSession", return_value=sess):
        generate_preview(
            onnx_path=fake_onnx,
            preview_path=preview_path,
            content_image=content_image,
            size=64,
        )

    assert len(captured) == 1
    assert captured[0].shape == (1, 3, 64, 64)


# ---------------------------------------------------------------------------
# NHWC tanh layout (AnimeGANv3)
# ---------------------------------------------------------------------------

def test_nhwc_creates_jpeg(tmp_path: Path, fake_onnx: Path, content_image: Path) -> None:
    from src.trainer.preview import generate_preview

    preview_path = tmp_path / "preview.jpg"
    sess = make_mock_session_nhwc(output_colour=(100, 180, 60))

    with patch("onnxruntime.InferenceSession", return_value=sess):
        generate_preview(
            onnx_path=fake_onnx,
            preview_path=preview_path,
            content_image=content_image,
            size=64,
            tensor_layout="nhwc_tanh",
        )

    assert preview_path.exists()


def test_nhwc_output_size(tmp_path: Path, fake_onnx: Path, content_image: Path) -> None:
    from src.trainer.preview import generate_preview

    preview_path = tmp_path / "preview.jpg"
    sess = make_mock_session_nhwc()

    with patch("onnxruntime.InferenceSession", return_value=sess):
        generate_preview(
            onnx_path=fake_onnx,
            preview_path=preview_path,
            content_image=content_image,
            size=64,
            tensor_layout="nhwc_tanh",
        )

    img = Image.open(str(preview_path))
    assert img.size == (64, 64)


def test_nhwc_colour_correct(tmp_path: Path, fake_onnx: Path, content_image: Path) -> None:
    from src.trainer.preview import generate_preview

    preview_path = tmp_path / "preview.jpg"
    colour = (100, 180, 60)
    sess = make_mock_session_nhwc(output_colour=colour)

    with patch("onnxruntime.InferenceSession", return_value=sess):
        generate_preview(
            onnx_path=fake_onnx,
            preview_path=preview_path,
            content_image=content_image,
            size=64,
            tensor_layout="nhwc_tanh",
        )

    arr = np.array(Image.open(str(preview_path)))
    assert abs(int(arr[0, 0, 0]) - colour[0]) <= 5
    assert abs(int(arr[0, 0, 1]) - colour[1]) <= 5
    assert abs(int(arr[0, 0, 2]) - colour[2]) <= 5


def test_nhwc_input_shape_is_1hwc(tmp_path: Path, fake_onnx: Path, content_image: Path) -> None:
    from src.trainer.preview import generate_preview

    preview_path = tmp_path / "preview.jpg"
    sess = make_mock_session_nhwc()
    captured: list[np.ndarray] = []

    original_run = sess.run.side_effect

    def _capture(output_names, feed):
        captured.append(feed["input"])
        return original_run(output_names, feed)

    sess.run.side_effect = _capture

    with patch("onnxruntime.InferenceSession", return_value=sess):
        generate_preview(
            onnx_path=fake_onnx,
            preview_path=preview_path,
            content_image=content_image,
            size=64,
            tensor_layout="nhwc_tanh",
        )

    assert len(captured) == 1
    assert captured[0].shape == (1, 64, 64, 3)


def test_nhwc_input_range_is_tanh(tmp_path: Path, fake_onnx: Path, content_image: Path) -> None:
    """Input tensor values must be in [-1, 1]."""
    from src.trainer.preview import generate_preview

    preview_path = tmp_path / "preview.jpg"
    sess = make_mock_session_nhwc()
    captured: list[np.ndarray] = []

    original_run = sess.run.side_effect

    def _capture(output_names, feed):
        captured.append(feed["input"])
        return original_run(output_names, feed)

    sess.run.side_effect = _capture

    with patch("onnxruntime.InferenceSession", return_value=sess):
        generate_preview(
            onnx_path=fake_onnx,
            preview_path=preview_path,
            content_image=content_image,
            size=64,
            tensor_layout="nhwc_tanh",
        )

    arr = captured[0]
    assert arr.min() >= -1.0 - 1e-5
    assert arr.max() <= 1.0 + 1e-5


def test_nhwc_result_is_rgb_image(tmp_path: Path, fake_onnx: Path, content_image: Path) -> None:
    from src.trainer.preview import generate_preview

    preview_path = tmp_path / "preview.jpg"
    sess = make_mock_session_nhwc()

    with patch("onnxruntime.InferenceSession", return_value=sess):
        generate_preview(
            onnx_path=fake_onnx,
            preview_path=preview_path,
            content_image=content_image,
            size=64,
            tensor_layout="nhwc_tanh",
        )

    img = Image.open(str(preview_path))
    assert img.mode == "RGB"
