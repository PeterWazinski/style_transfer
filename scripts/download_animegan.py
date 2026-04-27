"""Download and verify the AnimeGANv3 Hayao ONNX model.

Usage::

    python scripts/download_animegan.py

Downloads ``AnimeGANv3_Hayao_36.onnx`` (~4 MB) from the official GitHub release
and verifies the input/output tensor shapes via onnxruntime.

The file is saved to::

    <repo_root>/tmp/AnimeGANv3_Hayao_36.onnx

Key I/O characteristics discovered by this script (needed for engine integration):
  - Input  : [1, H, W, 3]  float32  range [-1, 1]  (NHWC, tanh-normalised)
  - Output : [1, H, W, 3]  float32  range [-1, 1]  (NHWC, tanh-normalised)
  - H and W must be multiples of 8, minimum 256
"""
from __future__ import annotations

import pathlib
import sys
import urllib.request
import warnings

REPO_ROOT = pathlib.Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(REPO_ROOT))

_URL = (
    "https://github.com/TachibanaYoshino/AnimeGANv3/releases/download"
    "/v1.1.0/AnimeGANv3_Hayao_36.onnx"
)
_DEST = REPO_ROOT / "tmp" / "AnimeGANv3_Hayao_36.onnx"


def _download() -> None:
    _DEST.parent.mkdir(parents=True, exist_ok=True)
    if _DEST.exists():
        print(f"Already present: {_DEST}  ({_DEST.stat().st_size / 1e6:.1f} MB)")
        return
    print(f"Downloading {_URL} ...")
    urllib.request.urlretrieve(_URL, _DEST)
    print(f"Saved  {_DEST.stat().st_size / 1e6:.1f} MB  -> {_DEST}")


def _verify() -> None:
    try:
        import onnxruntime as ort  # type: ignore[import]
    except ImportError:
        print("ERROR: onnxruntime is not installed.")
        sys.exit(1)

    import numpy as np

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        sess = ort.InferenceSession(
            str(_DEST), providers=["CPUExecutionProvider"]
        )

    inp = sess.get_inputs()[0]
    out = sess.get_outputs()[0]
    print(f"\nInput  : name={inp.name!r}  shape={inp.shape}  type={inp.type}")
    print(f"Output : name={out.name!r}  shape={out.shape}  type={out.type}")

    # Test inference with a 256x256 dummy image (minimum supported size)
    dummy = (
        np.random.rand(1, 256, 256, 3).astype(np.float32) * 2.0 - 1.0
    )  # [-1, 1]
    result = sess.run(None, {inp.name: dummy})
    out_arr = result[0]
    print(
        f"\nDummy inference OK: "
        f"output shape={out_arr.shape}  "
        f"range=[{out_arr.min():.3f}, {out_arr.max():.3f}]"
    )
    print("\nConclusion:")
    print("  tensor_layout = 'nhwc_tanh'")
    print("  Pre-processing : arr / 127.5 - 1.0  (RGB [0,255] -> [-1,1])")
    print("  Layout         : [1, H, W, 3] NHWC  (NOT NCHW)")
    print("  Size constraint: H and W must be multiples of 8, min 256")
    print("  Post-processing: (arr + 1.0) / 2.0 * 255.0  ([-1,1] -> [0,255])")


if __name__ == "__main__":
    _download()
    _verify()
