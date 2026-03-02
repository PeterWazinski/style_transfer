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

from PySide6.QtWidgets import QApplication

from src.core.engine import StyleTransferEngine
from src.core.photo_manager import PhotoManager
from src.core.registry import StyleRegistry
from src.core.settings import AppSettings
from src.ui.main_window import MainWindow

_STYLES_ROOT: Path = Path(__file__).parent.parent / "styles"
_CATALOG_PATH: Path = _STYLES_ROOT / "catalog.json"
_LOG_PATH: Path = Path(__file__).parent.parent / "app.log"


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


def main() -> int:
    """Create the Qt application, build the main window, and run the event loop."""
    _setup_logging()
    app = QApplication(sys.argv)
    app.setApplicationName("Style Transfer")
    app.setApplicationVersion("0.4.0")

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
