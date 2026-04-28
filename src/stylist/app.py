"""Application entry point.

Usage::

    python -m src.app
    # or
    python src/app.py
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QBrush, QColor, QIcon, QPainter, QPainterPath, QPixmap
from PySide6.QtWidgets import QApplication

from src.core.engine import StyleTransferEngine
from src.core.photo_manager import PhotoManager
from src.core.registry import StyleRegistry
from src.core.settings import AppSettings
from src.stylist.main_window import MainWindow


def _project_root() -> Path:
    """Return the data root directory.

    * When running from source: the repo root (two levels above this file).
    * When running as a PyInstaller one-directory bundle: ``sys._MEIPASS``,
      which in onedir mode equals the folder containing the exe.  The
      ``styles\\`` directory sits alongside the exe and is therefore found
      at ``_project_root() / 'styles'`` without any path changes.
    """
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        return Path(sys._MEIPASS)  # type: ignore[attr-defined]
    return Path(__file__).resolve().parent.parent.parent


def _log_path() -> Path:
    """Return a writable path for the application log file.

    When frozen the log is placed next to the ``.exe`` inside the app
    directory (``PetersPictureStyler\\app.log``).
    """
    if getattr(sys, "frozen", False):
        return Path(sys.executable).parent / "app.log"
    return Path(__file__).resolve().parent.parent.parent / "app.log"


_STYLES_ROOT: Path = _project_root() / "styles"
_CATALOG_PATH: Path = _STYLES_ROOT / "catalog.json"
_LOG_PATH: Path = _log_path()


def _setup_logging() -> None:
    """Configure root logger: DEBUG to app.log, INFO to console."""
    _LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter(
        "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    fh = logging.FileHandler(_LOG_PATH, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)

    root.addHandler(fh)
    root.addHandler(ch)


def _make_palette_icon(size: int = 64) -> QIcon:
    """Draw a painter's palette icon programmatically and return it as a QIcon."""
    pix = QPixmap(size, size)
    pix.fill(Qt.transparent)  # type: ignore[attr-defined]

    p = QPainter(pix)
    p.setRenderHint(QPainter.Antialiasing)  # type: ignore[attr-defined]

    # --- Palette body (egg-shaped: wide ellipse offset upward) ---
    body_color = QColor("#C8A87A")   # warm wood/tan
    p.setBrush(QBrush(body_color))
    p.setPen(Qt.NoPen)  # type: ignore[attr-defined]
    path = QPainterPath()
    path.addEllipse(2, 8, size - 4, size - 14)
    p.drawPath(path)

    # --- Thumb hole ---
    p.setBrush(QBrush(Qt.white))  # type: ignore[attr-defined]
    p.drawEllipse(int(size * 0.12), int(size * 0.38), int(size * 0.20), int(size * 0.22))

    # --- Paint blobs (arc of 6 colours) ---
    import math
    blob_colors = ["#E63946", "#F4A261", "#2A9D8F", "#457B9D", "#9B5DE5", "#F9C74F"]
    blob_r = int(size * 0.095)
    cx, cy = size * 0.58, size * 0.42
    radius = size * 0.27
    for i, color in enumerate(blob_colors):
        angle = math.pi * 0.05 + i * math.pi * 0.19
        bx = cx + radius * math.cos(angle)
        by = cy - radius * math.sin(angle)
        p.setBrush(QBrush(QColor(color)))
        p.drawEllipse(int(bx - blob_r), int(by - blob_r), blob_r * 2, blob_r * 2)

    p.end()
    return QIcon(pix)


def main() -> int:
    """Create the Qt application, build the main window, and run the event loop."""
    _setup_logging()
    app = QApplication(sys.argv)
    app.setApplicationName("Peter's Picture Stylist")
    app.setApplicationVersion("0.4.0")
    app.setWindowIcon(_make_palette_icon())

    settings = AppSettings.load()
    registry = StyleRegistry(catalog_path=_CATALOG_PATH)
    engine = StyleTransferEngine(execution_provider=settings.execution_provider)
    photo_manager = PhotoManager()

    window = MainWindow(
        registry=registry,
        engine=engine,
        photo_manager=photo_manager,
        settings=settings,
    )
    window.show()

    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
