"""Background worker for style transfer.

:class:`ApplyWorker` runs :meth:`~src.core.engine.StyleTransferEngine.apply`
in a dedicated :class:`QThread` so the main thread (and therefore the Qt event
loop) stays responsive while tiles are being processed.

Signals
-------
progress(done: int, total: int)
    Emitted after every tile.  Both values are positive integers with
    ``1 ≤ done ≤ total``.
finished(result: PIL.Image)
    Emitted when inference completes successfully.
cancelled()
    Emitted when the caller requests interruption and the engine stops early.
error(message: str)
    Emitted when inference raises an unexpected exception.
"""
from __future__ import annotations

from PIL.Image import Image as PILImage
from PySide6.QtCore import QThread, Signal

from src.core.engine import StyleTransferEngine

# DirectML / D3D12 driver-crash error codes (hex) that appear in ONNX error strings.
# 0x887A0020 = DXGI_ERROR_DRIVER_INTERNAL_ERROR
# 0x887A0006 = DXGI_ERROR_DEVICE_HUNG
# 0x887A0005 = DXGI_ERROR_DEVICE_REMOVED
_DML_CRASH_CODES = ("887A0020", "887A0006", "887A0005", "887A0007")


def _friendly_error(exc: Exception) -> str:
    """Return a human-readable error message, with extra guidance for GPU driver crashes."""
    msg = str(exc)
    if any(code in msg for code in _DML_CRASH_CODES):
        return (
            "The GPU driver crashed during inference (DirectML error).\n\n"
            "This usually means the selected style is too demanding for your GPU at the "
            "current tile size, or the driver is unstable.\n\n"
            "Suggestions:\n"
            "  • Reduce the Tile Size in File \u2192 Settings (e.g. 512 px)\n"
            "  • Restart the application and try again\n"
            "  • Update your GPU driver\n\n"
            f"Technical detail: {msg}"
        )
    return msg


class _CancelledError(Exception):
    """Internal sentinel raised by the progress callback on interruption."""


class ApplyWorker(QThread):
    """Run :meth:`StyleTransferEngine.apply` in a background thread.

    Args:
        engine:       The :class:`StyleTransferEngine` that owns the ONNX sessions.
        source:       PIL image to transform.
        style_id:     ID of the loaded model to apply.
        strength:     Blend factor in [0.0, 1.0].
        tile_size:    Tile dimension in pixels.
        overlap:      Overlap border in pixels.
        use_float16:  Cast tiles to float16 before inference.
        parent:       Optional Qt parent.
    """

    # ------------------------------------------------------------------
    # Qt signals
    # ------------------------------------------------------------------
    progress: Signal = Signal(int, int)   # done, total
    finished: Signal = Signal(object)     # PILImage
    cancelled: Signal = Signal()
    error: Signal = Signal(str)

    def __init__(
        self,
        engine: StyleTransferEngine,
        source: PILImage,
        style_id: str,
        strength: float,
        tile_size: int,
        overlap: int,
        use_float16: bool,
        parent: object = None,
    ) -> None:
        super().__init__(parent)  # type: ignore[call-arg]
        self._engine = engine
        self._source = source
        self._style_id = style_id
        self._strength = strength
        self._tile_size = tile_size
        self._overlap = overlap
        self._use_float16 = use_float16

    # ------------------------------------------------------------------
    # QThread entry point
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Execute inference and emit the appropriate signal on completion."""
        try:
            result = self._engine.apply(
                self._source,
                self._style_id,
                strength=self._strength,
                tile_size=self._tile_size,
                overlap=self._overlap,
                use_float16=self._use_float16,
                progress_callback=self._on_progress,
            )
        except _CancelledError:
            self.cancelled.emit()
        except Exception as exc:  # noqa: BLE001
            self.error.emit(_friendly_error(exc))
        else:
            self.finished.emit(result)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _on_progress(self, done: int, total: int) -> None:
        """Emit :attr:`progress` and raise :exc:`_CancelledError` if interrupted."""
        self.progress.emit(done, total)
        if self.isInterruptionRequested():
            raise _CancelledError
