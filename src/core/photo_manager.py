"""PhotoManager — JPEG/PNG photo I/O with EXIF preservation.

Only JPEG and PNG are supported; all other formats raise
`UnsupportedFormatError` with a clear message.
"""
from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import Optional

from PIL import Image, ExifTags, UnidentifiedImageError
from PIL.Image import Resampling

from src.core.tiling import TileInfo, split_tiles, merge_tiles

logger: logging.Logger = logging.getLogger(__name__)

# Supported extensions (lower-cased)
_SUPPORTED_SUFFIXES: frozenset[str] = frozenset({".jpg", ".jpeg", ".png"})

# EXIF tag id for Orientation
_EXIF_ORIENTATION_TAG: int = next(
    k for k, v in ExifTags.TAGS.items() if v == "Orientation"
)


# ---------------------------------------------------------------------------
# Custom exception
# ---------------------------------------------------------------------------

class UnsupportedFormatError(ValueError):
    """Raised when a file's extension is not JPEG or PNG."""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _check_extension(path: Path) -> None:
    """Raise `UnsupportedFormatError` for non-JPEG/PNG paths."""
    suffix = path.suffix.lower()
    if suffix not in _SUPPORTED_SUFFIXES:
        raise UnsupportedFormatError(
            f"Unsupported file format: '{path.suffix}'. "
            "Only JPEG (.jpg / .jpeg) and PNG (.png) are accepted."
        )


def _auto_rotate(image: Image.Image) -> Image.Image:
    """Return a copy of *image* rotated according to its EXIF orientation tag.

    Uses the public ``Image.getexif()`` API (Pillow ≥ 6.0).  If there is no
    EXIF data, or the Orientation tag is absent/unknown, the image is returned
    unchanged.
    """
    try:
        exif_dict = image.getexif()  # returns an Exif object (dict-like)
    except (AttributeError, Exception):
        return image

    orientation = exif_dict.get(_EXIF_ORIENTATION_TAG)
    if orientation is None:
        return image

    rotations: dict[int, Image.Image] = {
        3: image.rotate(180, expand=True),
        6: image.rotate(270, expand=True),
        8: image.rotate(90, expand=True),
    }
    return rotations.get(orientation, image)


def _get_exif_bytes(image: Image.Image) -> Optional[bytes]:
    """Return raw EXIF bytes from *image*, or None if unavailable."""
    try:
        exif = image.info.get("exif")
        if exif is not None:
            return bytes(exif)
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# PhotoManager
# ---------------------------------------------------------------------------

class PhotoManager:
    """Photo I/O and tile utilities.

    All methods are pure functions with respect to the filesystem — there is
    no internal state, so a single shared instance is safe.

    Example::

        pm = PhotoManager()
        img = pm.load(Path("photo.jpg"))
        thumb = pm.thumbnail(img, (256, 256))
        pm.save(img, Path("output.jpg"))
    """

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    def load(self, path: Path, max_megapixels: float = 0.0) -> Image.Image:
        """Load a JPEG or PNG file from disk.

        Applies EXIF-based auto-rotation so that the returned image is
        always in display orientation.  If *max_megapixels* is > 0 and the
        image exceeds that pixel budget it is down-scaled (aspect ratio
        preserved) before being returned.

        Args:
            path:            Absolute or relative path to the image file.
            max_megapixels:  Maximum pixel count in megapixels (e.g. 20.0).
                             Pass 0.0 (default) to skip the limit.

        Returns:
            RGB PIL Image.

        Raises:
            UnsupportedFormatError: If the file extension is not .jpg/.jpeg/.png.
            FileNotFoundError: If the file does not exist.
            PIL.UnidentifiedImageError: If Pillow cannot parse the file.
        """
        _check_extension(path)
        if not path.exists():
            raise FileNotFoundError(f"Image file not found: {path}")

        img = Image.open(path)
        # Peek the original size from the file header (no pixel decode yet).
        original_size: tuple[int, int] = img.size
        img.load()           # fully decode into memory so file can be closed
        img = _auto_rotate(img)
        img = img.convert("RGB")

        if max_megapixels > 0.0:
            mp = img.width * img.height / 1_000_000
            if mp > max_megapixels:
                scale = (max_megapixels / mp) ** 0.5
                new_w = int(img.width * scale)
                new_h = int(img.height * scale)
                img = img.resize((new_w, new_h), Image.LANCZOS)  # type: ignore[attr-defined]
                logger.info(
                    "Downscaled %s from %dx%d (%.1f MP) to %dx%d (%.1f MP) "
                    "to fit %.0f MP limit",
                    path.name,
                    original_size[0], original_size[1], mp,
                    new_w, new_h, new_w * new_h / 1_000_000,
                    max_megapixels,
                )

        logger.debug("Loaded %s (%s)", path, img.size)
        return img

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def save(
        self,
        image: Image.Image,
        path: Path,
        quality: int = 95,
        source_exif: Optional[Image.Image] = None,
    ) -> None:
        """Save *image* to *path* as JPEG or PNG.

        When *source_exif* is given its EXIF bytes are copied to the output
        (JPEG only; PNG EXIF is not widely supported).

        Args:
            image:       RGB PIL Image to save.
            path:        Destination path; parent dirs are created if needed.
            quality:     JPEG quality (1–95). Ignored for PNG.
            source_exif: Optional image whose EXIF tags are copied to output.

        Raises:
            UnsupportedFormatError: If the target extension is not supported.
        """
        _check_extension(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        img_rgb = image.convert("RGB")
        suffix = path.suffix.lower()

        if suffix in (".jpg", ".jpeg"):
            kwargs: dict = {"format": "JPEG", "quality": quality}
            if source_exif is not None:
                exif_bytes = _get_exif_bytes(source_exif)
                if exif_bytes:
                    kwargs["exif"] = exif_bytes
            img_rgb.save(path, **kwargs)
        else:
            img_rgb.save(path, format="PNG")

        logger.debug("Saved %s (%s)", path, image.size)

    # ------------------------------------------------------------------
    # Thumbnail
    # ------------------------------------------------------------------

    def thumbnail(
        self,
        image: Image.Image,
        max_size: tuple[int, int] = (256, 256),
    ) -> Image.Image:
        """Return a down-sampled copy of *image* that fits within *max_size*.

        Aspect ratio is preserved; the result will never exceed *max_size*
        in either dimension.  Small images are never upscaled.

        Args:
            image:    Source image.
            max_size: (max_width, max_height) bounding box in pixels.

        Returns:
            New PIL Image (mode RGB).
        """
        img = image.convert("RGB").copy()
        img.thumbnail(max_size, Resampling.LANCZOS)
        return img

    # ------------------------------------------------------------------
    # Tile helpers (thin delegation to tiling module)
    # ------------------------------------------------------------------

    def split_tiles(
        self,
        image: Image.Image,
        tile_size: int = 1024,
        overlap: int = 128,
    ) -> list[tuple[TileInfo, Image.Image]]:
        """Split *image* into overlapping tiles.

        Delegates to :func:`src.core.tiling.split_tiles`.  See that function
        for full argument documentation.

        Returns:
            List of (TileInfo, tile_image) pairs.
        """
        return split_tiles(image, tile_size=tile_size, overlap=overlap)

    def merge_tiles(
        self,
        tiles: list[tuple[TileInfo, Image.Image]],
        output_size: tuple[int, int],
    ) -> Image.Image:
        """Merge overlapping tiles back into a single image.

        Delegates to :func:`src.core.tiling.merge_tiles`.  See that function
        for full argument documentation.

        Returns:
            Reconstructed PIL Image of *output_size*.
        """
        return merge_tiles(tiles, output_size)
