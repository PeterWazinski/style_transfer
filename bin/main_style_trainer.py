"""Style Trainer CLI entry point (bin/ stub).

Thin wrapper — all logic lives in :mod:`src.trainer.app`.

Usage::

    python bin/main_style_trainer.py train --style my_style.jpg --coco data/train2017 ...
    python bin/main_style_trainer.py preview --model styles/my_style/model.onnx ...

See ``src/trainer/app.py`` for the full argument reference.
"""
from __future__ import annotations

import sys

from src.trainer.app import main

if __name__ == "__main__":
    sys.exit(main())
