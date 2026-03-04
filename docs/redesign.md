# Redesign: Stylist App + Trainer App â€” Clean Separation

**Status:** Approved â€” ready for implementation  
**Date:** 2026-03-04

---

## 1. Rationale

The style-training pipeline (PyTorch â†’ VGG-16 â†’ TransformerNet â†’ ONNX export)
is a **developer / data-science tool**, not a user feature:

- Requires a 13 GB MS-COCO dataset that end-users never have.
- Requires PyTorch + torchvision â€” heavy deps already excluded from the compiled exe.
- Training takes 2â€“40 h; it is not a task anyone does inside a GUI app.
- The wiring between `StyleEditorDialog` â†’ `TrainingProgressDialog` was never
  completed (stub only) â€” the "Add Style" menu item can only register metadata,
  it cannot actually train.
- Bundling trainer deps forces every contributor to install 2+ GB of torch even
  if they only work on the UI / ONNX inference path.

**Decisions (approved):**

| # | Decision |
|---|---|
| D1 | Two named apps: **Stylist** (`main_image_styler.py`) and **Trainer** (`main_style_trainer.py`) |
| D2 | Folder structure mirrors the apps: `src/stylist/`, `src/trainer/`, `src/core/` |
| D3 | Maximum separation â€” no import from `src/stylist` in trainer code and vice versa |
| D4 | `StyleModel.source_images` and `training_config` fields removed (backwards-compatible) |
| D5 | Preview generation extracted to `src/trainer/preview.py` |
| D6 | `scripts/setup_models.py` left as-is (dev bootstrap, not user-facing) |

---

## 2. Final Folder Structure

```
style_transfer/
â”‚
â”œâ”€â”€ main_image_styler.py        â† End-user app entry point  (NEW, thin wrapper)
â”œâ”€â”€ main_style_trainer.py       â† Developer CLI entry point (NEW)
â”‚
â”œâ”€â”€ src/
â”‚   â”œâ”€â”€ core/                   â† Shared by both apps â€” no Qt, no torch
â”‚   â”‚   â”œâ”€â”€ __init__.py
â”‚   â”‚   â”œâ”€â”€ engine.py           (ONNX inference engine)
â”‚   â”‚   â”œâ”€â”€ models.py           (StyleModel simplified â€” see Â§3)
â”‚   â”‚   â”œâ”€â”€ photo_manager.py
â”‚   â”‚   â”œâ”€â”€ registry.py
â”‚   â”‚   â”œâ”€â”€ settings.py
â”‚   â”‚   â””â”€â”€ tiling.py
â”‚   â”‚
â”‚   â”œâ”€â”€ stylist/                â† Stylist app â€” Qt / PySide6, no torch
â”‚   â”‚   â”œâ”€â”€ __init__.py
â”‚   â”‚   â”œâ”€â”€ app.py              (was src/app.py)
â”‚   â”‚   â”œâ”€â”€ main_window.py      (was src/ui/main_window.py â€” training removed)
â”‚   â”‚   â”œâ”€â”€ photo_canvas.py     (was src/ui/photo_canvas.py)
â”‚   â”‚   â”œâ”€â”€ settings_dialog.py  (was src/ui/settings_dialog.py)
â”‚   â”‚   â”œâ”€â”€ style_gallery.py    (was src/ui/style_gallery.py â€” read-only)
â”‚   â”‚   â””â”€â”€ widgets/
â”‚   â”‚       â”œâ”€â”€ __init__.py
â”‚   â”‚       â”œâ”€â”€ strength_slider.py
â”‚   â”‚       â””â”€â”€ thumbnail_delegate.py
â”‚   â”‚
â”‚   â””â”€â”€ trainer/                â† Trainer tool â€” torch, no Qt
â”‚       â”œâ”€â”€ __init__.py
â”‚       â”œâ”€â”€ style_trainer.py    (was src/core/trainer.py)
â”‚       â”œâ”€â”€ train_utils.py      (was src/ml/train_utils.py)
â”‚       â”œâ”€â”€ transformer_net.py  (was src/ml/transformer_net.py)
â”‚       â”œâ”€â”€ vgg_loss.py         (was src/ml/vgg_loss.py)
â”‚       â””â”€â”€ preview.py          (NEW â€” extracted from scripts/setup_models.py)
â”‚
â”œâ”€â”€ tests/
â”‚   â”œâ”€â”€ core/                   â† Tests for src/core (no Qt, no torch)
â”‚   â”‚   â”œâ”€â”€ __init__.py
â”‚   â”‚   â”œâ”€â”€ test_registry.py    (was tests/unit/test_registry.py)
â”‚   â”‚   â”œâ”€â”€ test_engine.py      (was tests/unit/test_engine.py)
â”‚   â”‚   â”œâ”€â”€ test_photo_manager.py (was tests/unit/test_photo_manager.py)
â”‚   â”‚   â””â”€â”€ test_error_handling.py (was tests/unit/ â€” trainer class removed)
â”‚   â”‚
â”‚   â”œâ”€â”€ stylist/                â† Tests for src/stylist
â”‚   â”‚   â”œâ”€â”€ __init__.py
â”‚   â”‚   â”œâ”€â”€ conftest.py         (was tests/ui/conftest.py â€” trainer fixtures removed)
â”‚   â”‚   â”œâ”€â”€ test_style_gallery.py (was tests/ui/ â€” editor-related tests removed)
â”‚   â”‚   â”œâ”€â”€ test_photo_canvas.py (was tests/ui/)
â”‚   â”‚   â”œâ”€â”€ test_end_to_end.py  (was tests/integration/test_end_to_end.py)
â”‚   â”‚   â””â”€â”€ test_apply_full_photo.py (was tests/integration/)
â”‚   â”‚
â”‚   â””â”€â”€ trainer/                â† Tests for src/trainer (require torch)
â”‚       â”œâ”€â”€ __init__.py
â”‚       â””â”€â”€ test_trainer_errors.py (TestTrainerErrors class from test_error_handling.py)
â”‚
â”œâ”€â”€ scripts/
â”‚   â””â”€â”€ setup_models.py         â† UNCHANGED (dev bootstrap)
â”‚
â””â”€â”€ styles/                     â† Shared data (read by both apps)
    â”œâ”€â”€ catalog.json
    â”œâ”€â”€ candy/
    â”œâ”€â”€ mosaic/
    â”œâ”€â”€ rain_princess/
    â”œâ”€â”€ abstract/
    â””â”€â”€ starry_night/
```

**Folders / packages deleted entirely:**

| Deleted path | Replaces |
|---|---|
| `src/ui/` | â†’ `src/stylist/` |
| `src/ml/` | â†’ `src/trainer/` |
| `tests/ui/` | â†’ `tests/stylist/` |
| `tests/unit/` | â†’ `tests/core/` + `tests/trainer/` |
| `tests/integration/` | â†’ `tests/stylist/` (training integration tests deleted) |

---

## 3. Files Deleted (not moved)

| File | Reason |
|---|---|
| `src/ui/style_editor.py` | "Add Style" dialog â€” training UI removed entirely |
| `src/ui/training_dialog.py` | Training progress dialog â€” training UI removed entirely |
| `tests/ui/test_style_editor.py` | 11 tests for deleted dialog |
| `tests/ui/test_training_dialog.py` | 14 tests for deleted dialog |
| `tests/integration/test_training_pipeline.py` | Integration-tests trainer core |
| `tests/integration/test_training_end_to_end.py` | End-to-end trainer test |
| `tests/unit/test_transformer_net.py` | ML-only â€” moves to `tests/trainer/` if kept, else delete |
| `tests/unit/test_vgg_loss.py` | ML-only â€” moves to `tests/trainer/` if kept, else delete |

---

## 4. Complete Import Change Map

Every import that must change when files move, listed explicitly.

### `src/stylist/app.py` (was `src/app.py`)

| Old import | New import |
|---|---|
| `from src.ui.main_window import MainWindow` | `from src.stylist.main_window import MainWindow` |
| `from src.core.*` | unchanged |

### `src/stylist/main_window.py` (was `src/ui/main_window.py`)

| Old import | New import |
|---|---|
| `from src.ui.photo_canvas import PhotoCanvasView` | `from src.stylist.photo_canvas import PhotoCanvasView` |
| `from src.ui.settings_dialog import SettingsDialog` | `from src.stylist.settings_dialog import SettingsDialog` |
| `from src.ui.style_gallery import StyleGalleryView` | `from src.stylist.style_gallery import StyleGalleryView` |
| `from src.ui.style_editor import StyleEditorDialog` | **DELETE** |
| `from src.ui.training_dialog import TrainingProgressDialog` | **DELETE** |
| `from src.core.*` | unchanged |

### `src/stylist/photo_canvas.py` (was `src/ui/photo_canvas.py`)

| Old import | New import |
|---|---|
| `from src.ui.widgets.strength_slider import StrengthSlider` | `from src.stylist.widgets.strength_slider import StrengthSlider` |

### `src/stylist/settings_dialog.py` / `src/stylist/style_gallery.py`

No import changes â€” both only import from `src.core.*` which is unchanged.

### `src/trainer/style_trainer.py` (was `src/core/trainer.py`)

| Old import | New import |
|---|---|
| `from src.ml.train_utils import CocoImageDataset, load_style_tensor` | `from src.trainer.train_utils import CocoImageDataset, load_style_tensor` |
| `from src.ml.transformer_net import TransformerNet` | `from src.trainer.transformer_net import TransformerNet` |
| `from src.ml.vgg_loss import VGGPerceptualLoss` | `from src.trainer.vgg_loss import VGGPerceptualLoss` |

### `main_image_styler.py` (NEW â€” project root)

```python
# Thin entry point â€” delegates to src/stylist/app.py
import sys
from src.stylist.app import main
sys.exit(main())
```

### `main_style_trainer.py` (NEW â€” project root)

```python
# Imports only from src/trainer and src/core â€” never from src/stylist
from src.trainer.style_trainer import StyleTrainer
from src.trainer.preview import generate_preview
from src.core.models import StyleModel
from src.core.registry import StyleRegistry
```

### `scripts/setup_models.py` â€” UNCHANGED

The script inlines its own `TransformerNet` / `_IgreatNet` copies and does
not import from `src/core/trainer.py` or `src/ml/`. No changes required.

### Test import changes

| File (new path) | Old import | New import |
|---|---|---|
| `tests/stylist/conftest.py` | `from src.ui.main_window import MainWindow` | `from src.stylist.main_window import MainWindow` |
| | `from src.ui.photo_canvas import PhotoCanvasView` | `from src.stylist.photo_canvas import PhotoCanvasView` |
| | `from src.ui.style_gallery import StyleGalleryView` | `from src.stylist.style_gallery import StyleGalleryView` |
| | `from src.ui.style_editor import StyleEditorDialog` | **DELETE** |
| | `from src.ui.training_dialog import TrainingProgressDialog` | **DELETE** |
| `tests/stylist/test_style_gallery.py` | `from src.ui.style_gallery import StyleGalleryView` | `from src.stylist.style_gallery import StyleGalleryView` |
| | `from src.ui.style_editor import StyleEditorDialog` | **DELETE** (editor gone) |
| | `from src.ui import style_gallery as _sg` | `from src.stylist import style_gallery as _sg` |
| `tests/stylist/test_photo_canvas.py` | `from src.ui.photo_canvas import PhotoCanvasView` | `from src.stylist.photo_canvas import PhotoCanvasView` |
| | `from src.ui.widgets.strength_slider import StrengthSlider` | `from src.stylist.widgets.strength_slider import StrengthSlider` |
| `tests/core/test_error_handling.py` | `from src.core.trainer import COCODatasetNotFoundError, StyleTrainer` | Move `TestTrainerErrors` class â†’ `tests/trainer/test_trainer_errors.py`; update to `from src.trainer.style_trainer import ...` |
| `tests/trainer/test_trainer_errors.py` | (extracted from above) | `from src.trainer.style_trainer import COCODatasetNotFoundError, StyleTrainer` |

---

## 5. `StyleModel` Simplification

Remove two training-only fields from `src/core/models.py` `StyleModel`:

```python
# REMOVE these two fields:
source_images: list[str] = field(default_factory=list)
training_config: Optional[dict[str, Any]] = None
```

Backwards-compatible: `StyleModel.from_dict()` already silently ignores
unknown keys, so existing `catalog.json` files with these fields load cleanly.

The 5 built-in catalog entries already have `source_images: []` and no
`training_config` â€” no migration of JSON files needed.

---

## 6. `src/trainer/preview.py` â€” New Module

Extract the `_generate_preview()` function from `scripts/setup_models.py`
into `src/trainer/preview.py` so `main_style_trainer.py` can call it without
duplicating code.

```python
# src/trainer/preview.py
def generate_preview(
    onnx_path: Path,
    preview_path: Path,
    content_image: Path,
    size: int = 256,
) -> None:
    """Run the ONNX model on a content image and save a square thumbnail."""
    ...
```

`scripts/setup_models.py` is **not** updated to use this (per D6 â€” leave as-is).

---

## 7. `pyproject.toml` Changes

Move `torch` / `torchvision` from `[project.dependencies]` into an optional
group so `pip install .` installs only what the Stylist app needs:

```toml
[project.dependencies]
# Stylist app only â€” no torch
onnxruntime-directml = ">=1.18.0"
Pillow = ">=10.3.0"
opencv-python = ">=4.9.0"
PySide6 = ">=6.7.0"
numpy = ">=1.26.0"

[project.optional-dependencies]
trainer = [
    "torch>=2.3.0",
    "torchvision>=0.18.0",
]
dev = [
    "torch>=2.3.0",
    "torchvision>=0.18.0",
    "pytest>=8.0",
    "pytest-qt>=4.4",
    "pytest-mock>=3.14",
    "pytest-cov>=5.0",
    "ruff",
    "mypy",
    "tensorboard",
]
```

Install for training work: `pip install -e ".[trainer]"`  
Install for app development: `pip install -e ".[dev]"`

---

## 8. Stylist UI Changes

### `src/stylist/main_window.py`
- Remove `Styles` menu entirely.
- Remove methods: `_open_add_style_dialog`, `_open_edit_style_dialog`,
  `_delete_style`, `_on_style_saved`, `_on_style_updated` (~40 lines).
- Remove attribute: `style_editor_dialog`.
- Remove signal wiring: `gallery.add_requested`, `gallery.edit_requested`,
  `gallery.delete_requested`.

### `src/stylist/style_gallery.py`
- Remove signals: `add_requested`, `edit_requested`, `delete_requested`.
- Remove the "+" add button in the toolbar.
- Remove right-click context menu (Edit / Delete items).
- Gallery becomes **read-only**: single-click selects, double-click applies.

---

## 9. Implementation Order

| Step | Action | Files touched | Risk |
|---|---|---|---|
| 1 | Create `src/stylist/`, `src/trainer/`, `tests/stylist/`, `tests/core/`, `tests/trainer/` | directories | None |
| 2 | Move `src/ui/*` â†’ `src/stylist/`; move `src/ml/*` + `src/core/trainer.py` â†’ `src/trainer/` | ~12 files | Low â€” rename only |
| 3 | Update all internal imports (see Â§4) | ~10 files | Low |
| 4 | Create `main_image_styler.py` and `main_style_trainer.py` | 2 new files | Low |
| 5 | Extract `generate_preview()` into `src/trainer/preview.py` | 1 new + 0 changes | Low |
| 6 | Simplify `StyleModel` (remove `source_images`, `training_config`) | `src/core/models.py` | Low |
| 7 | Strip training UI from `main_window.py` and `style_gallery.py` | 2 files | Low |
| 8 | Delete `src/ui/style_editor.py`, `src/ui/training_dialog.py` | 2 deletes | None |
| 9 | Reorganise tests (move, update imports, delete trainer UI tests) | ~10 files | Medium |
| 10 | Move `torch`/`torchvision` to optional deps in `pyproject.toml` | 1 file | Low |
| 11 | Update `style_transfer.spec` entry point: `src/app.py` â†’ `src/stylist/app.py` | 1 file | Low |
| 12 | Run full test suite â€” expect ~155 passing, 0 failing | â€” | â€” |
| 13 | Commit | â€” | â€” |
