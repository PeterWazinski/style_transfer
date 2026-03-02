"""Application settings — persisted to ~/.style_transfer/settings.json.

All settings that affect inference quality or performance live here so
they can be edited in the UI and honoured by the engine.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

logger: logging.Logger = logging.getLogger(__name__)

# Default location for the settings file.
_DEFAULT_SETTINGS_PATH: Path = Path.home() / ".style_transfer" / "settings.json"

# Valid execution-provider choices.
PROVIDER_CHOICES: tuple[str, ...] = ("auto", "cpu", "dml", "cuda")
# Valid tile-size choices (pixels).
TILE_SIZE_CHOICES: tuple[int, ...] = (512, 768, 1024, 2048)
# Valid overlap choices (pixels).
OVERLAP_CHOICES: tuple[int, ...] = (32, 64, 128, 192, 256)


@dataclass
class AppSettings:
    """All user-configurable runtime settings.

    Attributes:
        tile_size:           Tile dimension used in tiled inference (pixels).
        overlap:             Overlap border between adjacent tiles (pixels).
        default_output_dir:  Pre-populated save-file dialog directory.
                             Empty string = use the OS default.
        execution_provider:  ONNX Runtime execution provider preference.
                             ``"auto"`` picks DML → CUDA → CPU in order.
        use_float16:         Cast input tensors to float16 before inference
                             (faster on GPU/DML at the cost of slight quality
                             reduction; silently ignored on CPU-only runtimes).
    """

    tile_size: int = 1024
    overlap: int = 128
    default_output_dir: str = ""
    execution_provider: str = "auto"
    use_float16: bool = False

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def __post_init__(self) -> None:
        if self.tile_size not in TILE_SIZE_CHOICES:
            raise ValueError(
                f"tile_size must be one of {TILE_SIZE_CHOICES}, "
                f"got {self.tile_size}"
            )
        if self.overlap not in OVERLAP_CHOICES:
            raise ValueError(
                f"overlap must be one of {OVERLAP_CHOICES}, "
                f"got {self.overlap}"
            )
        if self.execution_provider not in PROVIDER_CHOICES:
            raise ValueError(
                f"execution_provider must be one of {PROVIDER_CHOICES}, "
                f"got {self.execution_provider!r}"
            )
        if self.overlap >= self.tile_size // 2:
            raise ValueError(
                f"overlap ({self.overlap}) must be < tile_size // 2 "
                f"({self.tile_size // 2})"
            )

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AppSettings":
        known: set[str] = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in data.items() if k in known}
        return cls(**filtered)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Path | None = None) -> None:
        """Persist settings to *path* (default: ``~/.style_transfer/settings.json``)."""
        target = path or _DEFAULT_SETTINGS_PATH
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), indent=2))
        logger.debug("Settings saved to %s", target)

    @classmethod
    def load(cls, path: Path | None = None) -> "AppSettings":
        """Load settings from *path*; return defaults if the file is absent/corrupt."""
        target = path or _DEFAULT_SETTINGS_PATH
        if not target.exists():
            return cls()
        try:
            data = json.loads(target.read_text())
            return cls.from_dict(data)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Could not read settings from %s (%s). Using defaults.", target, exc
            )
            return cls()
