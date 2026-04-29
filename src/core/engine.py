"""StyleTransferEngine — ONNX-based style inference with tiling support."""
from __future__ import annotations

import gc
import logging
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image

from src.core.tiling import TileInfo, merge_tiles, split_tiles

logger: logging.Logger = logging.getLogger(__name__)

# Lazy import: onnxruntime is only needed at inference time
try:
    import onnxruntime as ort  # type: ignore[import]
    _ORT_AVAILABLE: bool = True
except ImportError:
    _ORT_AVAILABLE = False


class StyleModelNotFoundError(FileNotFoundError):
    """Raised when a requested style model file cannot be found."""


class CorruptModelError(RuntimeError):
    """Raised when an ONNX model file exists but cannot be parsed."""


class OOMError(MemoryError):
    """Raised when there is insufficient memory to process a tile."""


# Valid ONNX Runtime execution-provider stacks.
_PROVIDER_STACKS: dict[str, list[str]] = {
    "auto": [
        "DmlExecutionProvider",
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    ],
    "dml":  ["DmlExecutionProvider",  "CPUExecutionProvider"],
    "cuda": ["CUDAExecutionProvider", "CPUExecutionProvider"],
    "cpu":  ["CPUExecutionProvider"],
}


class StyleTransferEngine:
    """Pure-Python inference engine. No Qt dependency.

    Usage::

        engine = StyleTransferEngine()
        engine.load_model("candy", Path("styles/candy/model.onnx"))
        result = engine.apply(photo, "candy", strength=0.8)
    """

    def __init__(self, execution_provider: str = "auto") -> None:
        """Create the engine.

        Args:
            execution_provider: One of ``"auto"``, ``"cpu"``, ``"dml"``,
                                ``"cuda"``.
        """
        if execution_provider not in _PROVIDER_STACKS:
            raise ValueError(
                f"execution_provider must be one of "
                f"{list(_PROVIDER_STACKS)}, got {execution_provider!r}"
            )
        self._execution_provider: str = execution_provider
        self._sessions: dict[str, "ort.InferenceSession"] = {}
        # Per-model tensor layout ("nchw" or "nhwc_tanh")
        self._model_meta: dict[str, str] = {}

    # ------------------------------------------------------------------
    # Model management
    # ------------------------------------------------------------------

    def load_model(
        self,
        style_id: str,
        model_path: Path | str,
        *,
        tensor_layout: str = "nchw",
    ) -> None:
        """Load an ONNX model and register it under *style_id*.

        Args:
            style_id:      Unique identifier (e.g. "candy").
            model_path:    Path (or str) to the .onnx file.
            tensor_layout: Tensor layout of the model.  One of:
                           ``"nchw"``       – standard NST TransformerNet
                           ``"nhwc_tanh"``  – AnimeGANv3-style TF models

        Raises:
            StyleModelNotFoundError: If the file does not exist.
            ImportError: If onnxruntime is not installed.
        """
        model_path = Path(model_path)  # coerce str → Path
        if not _ORT_AVAILABLE:
            raise ImportError(
                "onnxruntime is not installed. "
                "Run: pip install onnxruntime-directml"
            )
        if not model_path.exists():
            raise StyleModelNotFoundError(
                f"ONNX model not found: {model_path}"
            )

        providers: list[str] = _PROVIDER_STACKS[self._execution_provider]
        try:
            import warnings
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="Specified provider .* is not in available provider names",
                    category=UserWarning,
                )
                session: ort.InferenceSession = ort.InferenceSession(
                    str(model_path),
                    providers=providers,
                )
        except Exception as exc:  # noqa: BLE001
            raise CorruptModelError(
                f"Failed to load ONNX model '{model_path}': {exc}"
            ) from exc
        self._sessions[style_id] = session
        self._model_meta[style_id] = tensor_layout
        logger.info("Loaded model '%s' from %s (layout=%s)", style_id, model_path, tensor_layout)

    def is_loaded(self, style_id: str) -> bool:
        return style_id in self._sessions

    def unload_model(self, style_id: str) -> None:
        """Release the ONNX session for *style_id* and free its GPU memory."""
        self._sessions.pop(style_id, None)
        self._model_meta.pop(style_id, None)
        gc.collect()
        logger.info("Unloaded model '%s'.", style_id)

    def unload_all_models(self) -> None:
        """Release all cached ONNX sessions and free GPU/DirectML memory."""
        self._sessions.clear()
        self._model_meta.clear()
        gc.collect()
        logger.info("All ONNX sessions unloaded.")

    # ------------------------------------------------------------------
    # Core inference helpers
    # ------------------------------------------------------------------

    def _infer_tile(
        self,
        session: "ort.InferenceSession",
        tile: Image.Image,
        *,
        use_float16: bool = False,
        tensor_layout: str = "nchw",
    ) -> Image.Image:
        """Run one tile through the ONNX session.

        Args:
            session:     Active onnxruntime session.
            tile:        RGB PIL image of any size supported by the model.
            use_float16: If True, cast the input tensor to float16 before
                         sending to the runtime (faster on GPU/DML).

        Returns:
            Styled PIL image, same size as input tile.

        Raises:
            OOMError: If there is insufficient memory to allocate tile buffers.
        """
        if tensor_layout == "nhwc_tanh":
            return self._infer_tile_nhwc_tanh(session, tile)
        if tensor_layout == "nchw_tanh":
            return self._infer_tile_nchw_tanh(session, tile, use_float16=use_float16)
        try:
            arr = np.array(tile.convert("RGB"), dtype=np.float32)
            # Shape: [1, 3, H, W]
            tensor: np.ndarray = arr.transpose(2, 0, 1)[np.newaxis, ...]
            if use_float16:
                tensor = tensor.astype(np.float16)
            input_name: str = session.get_inputs()[0].name
            output: list[np.ndarray] = session.run(None, {input_name: tensor})
        except MemoryError as exc:
            raise OOMError(
                f"Out of memory processing a tile of size {tile.size}. "
                "Try reducing tile_size in Settings."
            ) from exc
        except Exception as exc:  # noqa: BLE001
            _msg = str(exc).lower()
            if any(k in _msg for k in ("out of memory", "insufficient", "oom", ": 6 :", "error code: 6")):
                raise OOMError(
                    f"GPU/DirectML out of memory processing a tile of size {tile.size}. "
                    "Open a new photo or reduce tile_size in Settings to free memory."
                ) from exc
            raise
        styled = np.clip(output[0][0].transpose(1, 2, 0), 0, 255).astype(np.uint8)
        result = Image.fromarray(styled)
        # Some ONNX models pad input to an alignment boundary, making the
        # output slightly larger than the input tile.  Crop back to the
        # original tile dimensions so merge_tiles array slices always match.
        if result.size != tile.size:
            result = result.crop((0, 0, tile.width, tile.height))
        return result

    def _infer_tile_nhwc_tanh(
        self,
        session: "ort.InferenceSession",
        tile: Image.Image,
    ) -> Image.Image:
        """Inference for NHWC models with tanh-normalised I/O (e.g. AnimeGANv3).

        Input : [1, H, W, 3]  float32  range [-1, 1]
        Output: [1, H, W, 3]  float32  range [-1, 1]
        H and W are rounded down to the nearest multiple of 8 (min 256).
        The result is resized back to the original tile dimensions.
        """
        orig_w, orig_h = tile.size

        def _to_8(x: int) -> int:
            return max(256, x - x % 8)

        w8, h8 = _to_8(orig_w), _to_8(orig_h)
        work_tile: Image.Image = (
            tile.resize((w8, h8), Image.BILINEAR)
            if (w8, h8) != (orig_w, orig_h)
            else tile
        )
        try:
            arr = np.array(work_tile.convert("RGB"), dtype=np.float32)
            # Normalise to [-1, 1], layout [1, H, W, 3]
            arr = arr / 127.5 - 1.0
            tensor: np.ndarray = arr[np.newaxis, ...]
            input_name: str = session.get_inputs()[0].name
            output: list[np.ndarray] = session.run(None, {input_name: tensor})
        except MemoryError as exc:
            raise OOMError(
                f"Out of memory processing a tile of size {tile.size}. "
                "Try reducing tile_size in Settings."
            ) from exc
        except Exception as exc:  # noqa: BLE001
            _msg = str(exc).lower()
            if any(k in _msg for k in ("out of memory", "insufficient", "oom", ": 6 :", "error code: 6")):
                raise OOMError(
                    f"GPU/DirectML out of memory processing a tile of size {tile.size}. "
                    "Open a new photo or reduce tile_size in Settings to free memory."
                ) from exc
            raise
        # De-normalise from [-1, 1] to [0, 255]
        result_arr = np.clip(
            (output[0][0] + 1.0) / 2.0 * 255.0, 0, 255
        ).astype(np.uint8)
        result = Image.fromarray(result_arr)
        if result.size != (orig_w, orig_h):
            result = result.resize((orig_w, orig_h), Image.BILINEAR)
        return result

    def _infer_tile_nchw_tanh(
        self,
        session: "ort.InferenceSession",
        tile: Image.Image,
        *,
        use_float16: bool = False,
    ) -> Image.Image:
        """Inference for NCHW models with tanh-normalised I/O (e.g. CycleGAN).

        Input : [1, 3, H, W]  float32  range [-1, 1]
        Output: [1, 3, H, W]  float32  range [-1, 1]
        """
        try:
            arr = np.array(tile.convert("RGB"), dtype=np.float32)
            arr = arr / 127.5 - 1.0                           # [0, 255] → [-1, 1]
            tensor: np.ndarray = arr.transpose(2, 0, 1)[np.newaxis, ...]  # → [1, 3, H, W]
            if use_float16:
                tensor = tensor.astype(np.float16)
            input_name: str = session.get_inputs()[0].name
            output: list[np.ndarray] = session.run(None, {input_name: tensor})
        except MemoryError as exc:
            raise OOMError(
                f"Out of memory processing a tile of size {tile.size}. "
                "Try reducing tile_size in Settings."
            ) from exc
        except Exception as exc:  # noqa: BLE001
            _msg = str(exc).lower()
            if any(k in _msg for k in ("out of memory", "insufficient", "oom", ": 6 :", "error code: 6")):
                raise OOMError(
                    f"GPU/DirectML out of memory processing a tile of size {tile.size}. "
                    "Open a new photo or reduce tile_size in Settings to free memory."
                ) from exc
            raise
        # De-normalise from [-1, 1] to [0, 255]; output is [1, 3, H, W]
        result_arr = np.clip(
            (output[0][0].transpose(1, 2, 0) + 1.0) / 2.0 * 255.0, 0, 255
        ).astype(np.uint8)
        result = Image.fromarray(result_arr)
        if result.size != tile.size:
            result = result.crop((0, 0, tile.width, tile.height))
        return result

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def apply(
        self,
        content_image: Image.Image,
        style_id: str,
        strength: float = 1.0,
        tile_size: int = 1024,
        overlap: int = 128,
        use_float16: bool = False,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> Image.Image:
        """Apply style transfer to *content_image* (full resolution).

        Large images are processed tile-by-tile with Gaussian blending.
        The final output is alpha-blended with the original according to
        *strength*.

        Args:
            content_image:     Source PIL image (JPEG/PNG, any resolution).
            style_id:          ID of a previously loaded model.
            strength:          Blend/extrapolation factor in [0.0, 3.0].
                               0 = original, 1 = fully styled,
                               >1 = extrapolated (amplified style effect).
            tile_size:         Tile dimension in pixels (default 1024).
            overlap:           Overlap border in pixels (default 128).
            use_float16:       Cast input tiles to float16 (faster on GPU).
            progress_callback: Optional callable(done, total) for progress.

        Returns:
            Styled PIL image of the same size as *content_image*.

        Raises:
            KeyError:              If *style_id* was not loaded.
            ValueError:            If *strength* is outside [0.0, 3.0].
        """
        if style_id not in self._sessions:
            raise KeyError(
                f"Style '{style_id}' is not loaded. Call load_model() first."
            )
        if not 0.0 <= strength <= 3.0:
            raise ValueError(f"strength must be in [0.0, 3.0], got {strength}")

        session = self._sessions[style_id]
        tensor_layout: str = self._model_meta.get(style_id, "nchw")
        original_size = content_image.size
        tiles: list[tuple[TileInfo, Image.Image]] = split_tiles(
            content_image, tile_size=tile_size, overlap=overlap
        )

        styled_tiles: list[tuple[TileInfo, Image.Image]] = []
        total: int = len(tiles)
        for i, (info, tile) in enumerate(tiles):
            logger.debug("Processing tile %d/%d", i + 1, total)
            styled_tile = self._infer_tile(
                session, tile, use_float16=use_float16, tensor_layout=tensor_layout
            )
            styled_tiles.append((info, styled_tile))
            if progress_callback is not None:
                progress_callback(i + 1, total)

        styled_full = merge_tiles(styled_tiles, original_size)

        if strength == 1.0:
            return styled_full
        if strength <= 0.0:
            return content_image.copy()

        content_rgb = content_image.convert("RGB").resize(original_size)
        if strength < 1.0:
            # Blend: result = α × styled + (1-α) × original
            return Image.blend(content_rgb, styled_full, alpha=strength)

        # Extrapolate: result = original + α × (styled - original)  [α > 1]
        # Pushes style effect beyond the model's native output.
        arr_orig = np.array(content_rgb, dtype=np.float32)
        arr_styled = np.array(styled_full, dtype=np.float32)
        arr_result = np.clip(arr_orig + strength * (arr_styled - arr_orig), 0, 255).astype(np.uint8)
        return Image.fromarray(arr_result)

    def preview(
        self,
        content_image: Image.Image,
        style_id: str,
        strength: float = 1.0,
        max_dim: int = 512,
    ) -> Image.Image:
        """Fast preview at reduced resolution (no tiling needed).

        Args:
            content_image: Source image.
            style_id:      Loaded style ID.
            strength:      Blend factor [0.0, 1.0].
            max_dim:       Downscale so longest edge ≤ max_dim before inference.

        Returns:
            Preview PIL image (may be smaller than the original).
        """
        w, h = content_image.size
        scale = min(1.0, max_dim / max(w, h))
        small = content_image.resize(
            (int(w * scale), int(h * scale)), Image.Resampling.LANCZOS
        )
        return self.apply(
            small,
            style_id,
            strength=strength,
            tile_size=max(max_dim, 512),
            overlap=0,
        )
