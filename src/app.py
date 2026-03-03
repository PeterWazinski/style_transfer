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


def _project_root() -> Path:
    """Return the data root directory.

    * When running from source: the repo root (two levels above this file).
    * When running as a PyInstaller one-file bundle: ``sys._MEIPASS``, the
      temporary directory where the bundle is extracted at startup.
    """
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        return Path(sys._MEIPASS)  # type: ignore[attr-defined]
    return Path(__file__).resolve().parent.parent


def _log_path() -> Path:
    """Return a writable path for the application log file.

    When frozen the log is placed next to the ``.exe`` (``sys.executable``),
    not inside the read-only extraction temp dir.
    """
    if getattr(sys, "frozen", False):
        return Path(sys.executable).parent / "app.log"
    return Path(__file__).resolve().parent.parent / "app.log"


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
