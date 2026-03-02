"""StyleTransferEngine — ONNX-based style inference with tiling support."""
from __future__ import annotations

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


class StyleTransferEngine:
    """Pure-Python inference engine. No Qt dependency.

    Usage::

        engine = StyleTransferEngine()
        engine.load_model("candy", Path("styles/candy/model.onnx"))
        result = engine.apply(photo, "candy", strength=0.8)
    """

    def __init__(self) -> None:
        self._sessions: dict[str, "ort.InferenceSession"] = {}

    # ------------------------------------------------------------------
    # Model management
    # ------------------------------------------------------------------

    def load_model(self, style_id: str, model_path: Path) -> None:
        """Load an ONNX model and register it under *style_id*.

        Args:
            style_id:   Unique identifier (e.g. "candy").
            model_path: Path to the .onnx file.

        Raises:
            StyleModelNotFoundError: If the file does not exist.
            ImportError: If onnxruntime is not installed.
        """
        if not _ORT_AVAILABLE:
            raise ImportError(
                "onnxruntime is not installed. "
                "Run: pip install onnxruntime-directml"
            )
        if not model_path.exists():
            raise StyleModelNotFoundError(
                f"ONNX model not found: {model_path}"
            )

        providers: list[str] = [
            "DmlExecutionProvider",   # Intel Arc / AMD via DirectML on Windows
            "CUDAExecutionProvider",  # Nvidia (if available)
            "CPUExecutionProvider",   # fallback
        ]
        session: ort.InferenceSession = ort.InferenceSession(
            str(model_path),
            providers=providers,
        )
        self._sessions[style_id] = session
        logger.info("Loaded model '%s' from %s", style_id, model_path)

    def is_loaded(self, style_id: str) -> bool:
        return style_id in self._sessions

    # ------------------------------------------------------------------
    # Core inference helpers
    # ------------------------------------------------------------------

    def _infer_tile(
        self,
        session: "ort.InferenceSession",
        tile: Image.Image,
    ) -> Image.Image:
        """Run one tile through the ONNX session.

        Args:
            session: Active onnxruntime session.
            tile:    RGB PIL image of any size supported by the model.

        Returns:
            Styled PIL image, same size as input tile.
        """
        arr = np.array(tile.convert("RGB"), dtype=np.float32)
        # Shape: [1, 3, H, W]
        tensor = arr.transpose(2, 0, 1)[np.newaxis, ...] 
        input_name: str = session.get_inputs()[0].name
        output: list[np.ndarray] = session.run(None, {input_name: tensor})
        styled = np.clip(output[0][0].transpose(1, 2, 0), 0, 255).astype(np.uint8)
        return Image.fromarray(styled)

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
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> Image.Image:
        """Apply style transfer to *content_image* (full resolution).

        Large images are processed tile-by-tile with Gaussian blending.
        The final output is alpha-blended with the original according to
        *strength*.

        Args:
            content_image:     Source PIL image (JPEG/PNG, any resolution).
            style_id:          ID of a previously loaded model.
            strength:          Blend factor in [0.0, 1.0].
                               0 = original, 1 = fully styled.
            tile_size:         Tile dimension in pixels (default 1024).
            overlap:           Overlap border in pixels (default 128).
            progress_callback: Optional callable(done, total) for progress.

        Returns:
            Styled PIL image of the same size as *content_image*.

        Raises:
            KeyError:              If *style_id* was not loaded.
            ValueError:            If *strength* is outside [0.0, 1.0].
        """
        if style_id not in self._sessions:
            raise KeyError(
                f"Style '{style_id}' is not loaded. Call load_model() first."
            )
        if not 0.0 <= strength <= 1.0:
            raise ValueError(f"strength must be in [0.0, 1.0], got {strength}")

        session = self._sessions[style_id]
        original_size = content_image.size
        tiles: list[tuple[TileInfo, Image.Image]] = split_tiles(
            content_image, tile_size=tile_size, overlap=overlap
        )

        styled_tiles: list[tuple[TileInfo, Image.Image]] = []
        total: int = len(tiles)
        for i, (info, tile) in enumerate(tiles):
            logger.debug("Processing tile %d/%d", i + 1, total)
            styled_tile = self._infer_tile(session, tile)
            styled_tiles.append((info, styled_tile))
            if progress_callback is not None:
                progress_callback(i + 1, total)

        styled_full = merge_tiles(styled_tiles, original_size)

        if strength >= 1.0:
            return styled_full
        if strength <= 0.0:
            return content_image.copy()

        # Blend: result = α × styled + (1-α) × original
        content_rgb = content_image.convert("RGB").resize(original_size)
        return Image.blend(content_rgb, styled_full, alpha=strength)

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
            (int(w * scale), int(h * scale)), Image.LANCZOS
        )
        return self.apply(
            small,
            style_id,
            strength=strength,
            tile_size=max(max_dim, 512),
            overlap=0,
        )
