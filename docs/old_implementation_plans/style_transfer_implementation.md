# Fast Neural Style Transfer — Software Architecture & Implementation Proposal

> **Status:** Draft for review — March 2026  
> **Target platform:** Windows desktop (single-user, local files)  
> **Author:** GitHub Copilot

---

## Table of Contents

1. [Algorithm Choice & Theoretical Background](#1-algorithm-choice--theoretical-background)
2. [Predefined Styles & Pretrained Models](#2-predefined-styles--pretrained-models)
3. [Handling 12 MP Real-World Photos](#3-handling-12-mp-real-world-photos)
4. [ML Framework & Library Evaluation](#4-ml-framework--library-evaluation)
5. [Imaging Library Evaluation](#5-imaging-library-evaluation)
6. [GUI Framework Evaluation & Recommendation](#6-gui-framework-evaluation--recommendation)
7. [Component Architecture (Separation of Concerns)](#7-component-architecture-separation-of-concerns)
8. [Project Directory Structure](#8-project-directory-structure)
9. [Component Design Detail](#9-component-design-detail)
   - 9.1 StyleModel — data layer
   - 9.2 StyleRegistry — style management
   - 9.3 StyleTransferEngine — inference
   - 9.4 StyleTrainer — training
   - 9.5 PhotoManager — photo I/O
   - 9.6 UI Layer (PySide6)
10. [Gradual Style Application](#10-gradual-style-application)
11. [Testing Strategy](#11-testing-strategy)
12. [Dependencies & Environment](#12-dependencies--environment)
13. [Implementation Phases / Roadmap](#13-implementation-phases--roadmap)
14. [Decisions & Resolved Risks](#14-decisions--resolved-risks)
15. [Background: What is Perceptual Loss (VGG-16)?](#15-background-what-is-perceptual-loss-vgg-16)
16. [Background: What is ONNX?](#16-background-what-is-onnx)

---

## 1. Algorithm Choice & Theoretical Background

### Selected Algorithm — Johnson et al. Feed-Forward Style Transfer

The implementation uses **"Perceptual Losses for Real-Time Style Transfer and Super-Resolution"** (Johnson, Alahi, Fei-Fei, ECCV 2016, [arxiv 1603.08155](https://arxiv.org/abs/1603.08155)) combined with **Instance Normalization** (Ulyanov et al., 2017).

#### Why this algorithm?

| Property | Optimization-based (Gatys 2015) | **Feed-forward (Johnson 2016)** |
|---|---|---|
| Inference latency | 30–300 s per image | **< 1 s per image** |
| Quality | Highest | Good |
| Laptop-friendly | No | **Yes** |
| Trainable per style | Not separately | **Yes — one model per style** |
| 12 MP support | Memory problems | Manageable with tiling |

#### How it works (summary)

```
Training time (once, per style):
  style_image  ──┐
                 ▼
  MS-COCO imgs → TransformerNet → styled_img
                                      │
              VGG-16 feature extractor │── content_loss + style_loss → backprop
                                      │
                               ground truth content img

Inference time (fast):
  content_photo → TransformerNet(trained) → styled_photo   (< 1 s)
```

- **TransformerNet** is a lightweight ResNet-inspired encoder-decoder with residual blocks.
- **VGG-16** (pretrained on ImageNet, frozen) is used *only* during training as a feature extractor for perceptual loss. It is **not** needed at inference time.
- **Instance Normalization** (rather than Batch Normalization) greatly stabilises training and improves visual quality.
- MS-COCO 2017 training set (~80 k images) is the standard training corpus.

#### References implemented by the target repositories

| Repository | Model format | Pretrained styles |
|---|---|---|
| [yakhyo/fast-neural-style-transfer](https://github.com/yakhyo/fast-neural-style-transfer) | PyTorch `.pth` + ONNX `.onnx` | candy, mosaic, rain-princess, udnie |
| [igreat/fast-style-transfer](https://github.com/igreat/fast-style-transfer) | PyTorch `.pth` | starry_night, rain_princess, mosaic, abstract |

Both implement the same Johnson-2016 / Instance-Norm pipeline. The yakhyo repo additionally exports ONNX, enabling faster CPU inference via ONNX Runtime.

---

## 2. Predefined Styles & Pretrained Models

### 2.1 Available Pretrained Style Models

The following style models can be downloaded and shipped with the application, requiring **zero training** to get started:

| Style Name | Visual Character | Source repo | Download |
|---|---|---|---|
| **Candy** | Bright, cartoon-like colours | yakhyo | GitHub Release v0.1 |
| **Mosaic** | Byzantine tile art | yakhyo + igreat | GitHub Release |
| **Rain Princess** | Dark, impressionist paint | yakhyo + igreat | GitHub Release |
| **Udnie** | Abstract curved forms | yakhyo | GitHub Release |
| **Starry Night** | Van Gogh swirling style | igreat | saved_models/ |
| **Abstract (DALL-E)** | Contemporary abstract | igreat | saved_models/ |

All ONNX weights from yakhyo are already committed in the `weights/` folder of the repository; the PyTorch weights are in `Release v0.1`.

### 2.2 New Custom Style Training

Users can add their own styles by providing a **reference artist image** (e.g., a painting by Van Gogh, Monet, etc.). The `StyleTrainer` component automates:

1. Accept one or more reference images as the style target
2. Launch TransformerNet training against MS-COCO training set
3. Save the resulting `.pth` model into the local `styles/` folder
4. Register the new style in the `StyleRegistry`

> **Training time estimate on a mid-range laptop GPU (RTX 3060):** ~2–4 hours for one epoch over MS-COCO (80 k images, batch=4). On CPU only, training is not practical (~48 h).
>
> **Decision:** The application ships all bundled styles as ready-to-use pretrained ONNX models (no training required by the user). When a user initiates training of a *custom* style, the following UX flow applies:
> 1. A **pre-flight warning dialog** is shown before training starts, stating estimated duration based on detected hardware (GPU/CPU) and dataset size.
> 2. Training runs in a background `QThread` with a **live progress bar** showing images processed, percentage complete, elapsed time, and estimated time remaining (ETA updated every epoch batch).
> 3. Checkpoints are saved every N images so training can be **resumed** after interruption.

### 2.3 VGG-16 vs VGG-19

- VGG-16 is the standard for perceptual loss in this pipeline; it offers a good quality/speed balance and is smaller than VGG-19.
- VGG-19 can be used if a user wants marginally higher quality at the cost of 30% more VRAM during training.
- **Recommendation:** default to VGG-16; expose VGG-19 as an advanced training option.
- Both are available via `torchvision.models` and download automatically.

---

## 3. Handling 12 MP Real-World Photos

A 12 MP image (e.g., 4000 × 3000 pixels) occupies ~144 MB as a 32-bit float RGB tensor. The TransformerNet itself is small, but the intermediate feature maps make full-resolution inference impractical in one pass on a laptop.

### Strategy: Adaptive Tiling with Overlap-Blend (recommended default)

```
┌─────────────────────────────────┐
│         12 MP Source Photo      │
└────────────┬────────────────────┘
             │ split with overlap (128 px border)
    ┌────────▼────────┐  ┌────────▼────────┐
    │  Tile 0 (1024²) │  │  Tile 1 (1024²) │  ...
    └────────┬────────┘  └────────┬────────┘
             │ TransformerNet inference (each tile)
    ┌────────▼────────┐  ┌────────▼────────┐
    │ Styled Tile 0   │  │ Styled Tile 1   │
    └────────┬────────┘  └────────┬────────┘
             │ Gaussian blend at seams
    ┌────────▼────────────────────▼────────┐
    │       Reconstructed 12 MP output     │
    └──────────────────────────────────────┘
```

| Approach | Pros | Cons |
|---|---|---|
| **Tiling + overlap blend** | Works on CPU/GPU, no quality loss at seams with good blending | Slightly longer runtime |
| Downscale → stylize → upscale | Fast | Loses detail; upscaling artefacts |
| Full-resolution single pass | Perfect quality | Requires 8+ GB VRAM; OOM on most laptops |

**Implementation:**
- Default tile size: `1024 × 1024` px (configurable in settings)
- Overlap: `128` px (Gaussian-weighted feathering at borders)
- Progress bar shown during tile processing
- Images are processed as float16 on GPU (halves VRAM requirement)
- Pillow is used for all I/O; tiles are created with numpy slicing

---

## 4. ML Framework & Library Evaluation

### 4.1 PyTorch (recommended)

| Aspect | Detail |
|---|---|
| Maturity | Best-in-class for research and production style transfer |
| Laptop support | CPU inference works; CUDA for Nvidia GPUs, DirectML via `torch-directml` for AMD |
| ONNX export | `torch.onnx.export()` built-in |
| Ecosystem | `torchvision` provides VGG-16/19, transforms, MS-COCO dataloader |
| Testability | Models are pure Python; fully unit-testable without GPU |
| Version | torch 2.x (stable), Python 3.11 |

### 4.2 ONNX Runtime (recommended for inference)

After training, models are exported to ONNX and executed via `onnxruntime`. This:
- Eliminates the PyTorch runtime dependency for end-users who only apply styles (not train)
- Provides automatic CPU/GPU acceleration without CUDA setup
- Enables int8 quantisation for faster CPU inference

### 4.3 Alternatives Considered

| Framework | Verdict |
|---|---|
| TensorFlow/Keras | Heavier, less ergonomic Python API, fewer style transfer tutorials |
| JAX | Excellent but lacks Windows GPU support; steep learning curve |
| TorchScript | Used internally for serialisation but not standalone |

### 4.4 GPU Acceleration — This Laptop (Intel Arc 140V)

The development machine has an **Intel Arc 140V GPU (16 GB)** and no Nvidia GPU, so CUDA is not available.

| Scenario | Strategy | Notes |
|---|---|---|
| **Inference (applying styles)** | `onnxruntime-directml` | DirectML runs on Intel Arc via Windows DirectX 12; fastest option on this machine |
| **Training (custom styles)** | PyTorch CPU | Intel Extension for PyTorch (IPEX XPU) exists for Arc but is complex to set up; CPU training is the practical default |
| Nvidia GPU (other machines) | `torch` CUDA + `onnxruntime-gpu` | For completeness; not needed here |
| Apple Silicon | Not target platform | `mps` backend exists but irrelevant here |

> **Practical recommendation for this laptop:** install `onnxruntime-directml` instead of plain `onnxruntime`. Inference speed on the Arc 140V via DirectML is substantially faster than pure CPU.

---

## 5. Imaging Library Evaluation

### 5.1 Pillow

| | |
|---|---|
| **Pros** | Pure Python; excellent EXIF/metadata handling; supports JPEG and PNG (the two supported formats in this application); lossless PNG I/O; trivial colour space conversion |
| **Cons** | No GPU acceleration; slower than OpenCV for large transforms |
| **Best for** | File load/save, colour management, thumbnail generation, format conversion |

### 5.2 OpenCV (`opencv-python`)

| | |
|---|---|
| **Pros** | Fast C++ backend; excellent for resizing, blending, tiling operations; good for preview generation |
| **Cons** | BGR colour order (must convert for PyTorch); heavy dependency; limited EXIF support |
| **Best for** | Tile splitting/blending, preview scaling, overlap feathering |

### 5.3 Recommendation: Use both, each for its strength

```
Photo Load/Save/EXIF  →  Pillow
Tile ops / blending   →  NumPy arrays + OpenCV
ML tensor             →  torchvision.transforms (ToTensor, Normalize)
Thumbnail previews    →  Pillow (thumbnail method)
```

This gives the best combination: reliable I/O with Pillow, fast numerical operations with OpenCV/NumPy, and clean ML pipeline with torchvision.

---

## 6. GUI Framework Evaluation & Recommendation

### 6.1 Options

| Framework | Language binding | Testability | Packaging | Maturity |
|---|---|---|---|---|
| **PyQt6** | Python | `pytest-qt` (excellent) | `PyInstaller` / `cx_Freeze` | Very mature (Qt 6.x) |
| PySide6 | Python | `pytest-qt` | Same | Official Qt binding (LGPL) |
| tkinter | Python | Limited (`pytest` only) | Built-in | Dated look and feel |
| wxPython | Python | `wxUiTesting` | Complex | Mature but less Python-idiomatic |
| Dear PyGui | Python | Limited | Good | Modern look, game-oriented |
| Kivy | Python | Limited | Android/iOS focus | Not ideal for desktop |

### 6.2 Recommendation: **PySide6** (with PyQt6 as drop-in alternative)

**Rationale:**
- PySide6 is the official Qt Company Python binding (LGPL licence — no GPL licence complications for private distribution)
- API is identical to PyQt6; switching between them is a one-line import change
- `pytest-qt` supports both seamlessly, enabling full automated UI testing
- Qt 6 ships `QML`, `QtConcurrent`, and `QThreadPool` enabling non-blocking background inference/training
- Qt's `QAbstractListModel` + `QListView` pattern is ideal for the visual style gallery
- `QGraphicsView` is ideal for the photo canvas with style-strength slider overlay

> If the user prefers GPL/commercial licensing clarity, PyQt6 is the identical-API alternative.

### 6.3 Key Qt Widgets Used

| UI Area | Qt Component |
|---|---|
| Style gallery (thumbnails) | `QListView` + `QStyledItemDelegate` (custom paint) |
| Photo canvas | `QGraphicsView` / `QLabel` with aspect ratio scaling |
| Strength slider | `QSlider` (0–100) |
| Training progress | `QProgressDialog` + `QThread` worker |
| Style add/edit dialog | `QDialog` with `QFormLayout` |
| Main window | `QMainWindow` with `QDockWidget` panels |
| File picker | `QFileDialog` (native OS dialog) |

---

## 7. Component Architecture (Separation of Concerns)

```
┌─────────────────────────────────────────────────────────────────────┐
│                         UI Layer (PySide6)                          │
│  ┌─────────────────┐  ┌─────────────────┐  ┌──────────────────────┐│
│  │ StyleGalleryView│  │ PhotoCanvasView │  │ TrainingProgressView ││
│  │ (browse/CRUD)   │  │ (apply + slider)│  │ (add custom style)   ││
│  └────────┬────────┘  └────────┬────────┘  └──────────┬───────────┘│
└───────────┼─────────────────────┼────────────────────────┼──────────┘
            │                     │                        │
            │         Application Services (pure Python, no Qt)
            ▼                     ▼                        ▼
┌──────────────────┐  ┌──────────────────────┐  ┌────────────────────┐
│  StyleRegistry   │  │  StyleTransferEngine  │  │   StyleTrainer     │
│                  │  │                       │  │                    │
│ - list()         │  │ - apply(photo, style, │  │ - train(style_img, │
│ - get(id)        │  │         strength)     │  │   dataset, hparams)│
│ - add(style)     │  │ - apply_tiled(...)    │  │ - resume(ckpt)     │
│ - delete(id)     │  │ - preview(photo,style)│  │ - export_onnx()    │
│ - update(style)  │  │                       │  │                    │
└────────┬─────────┘  └──────────┬────────────┘  └─────────┬──────────┘
         │                       │                          │
         ▼                       ▼                          ▼
┌──────────────────┐  ┌──────────────────────┐  ┌────────────────────┐
│  StyleModel /    │  │   TransformerNet     │  │  VGGPerceptualLoss  │
│  StyleStore      │  │   (PyTorch / ONNX)   │  │  (VGG-16, frozen)  │
│  (JSON + .onnx)  │  │                      │  │                    │
└──────────────────┘  └──────────────────────┘  └────────────────────┘
         │                       │
         ▼                       ▼
┌──────────────────────────────────────────────────────────────────────┐
│                        PhotoManager                                  │
│  - load(path) → PIL.Image                                            │
│  - save(image, path, quality)                                        │
│  - thumbnail(image, size) → PIL.Image                                │
│  - split_tiles / merge_tiles (OpenCV + NumPy)                        │
└──────────────────────────────────────────────────────────────────────┘
```

**Key principle:** The UI layer never imports PyTorch or PIL directly. All ML and imaging operations go through the service layer, making the entire core independently testable without a display.

---

## 8. Project Directory Structure

```
style_transfer/
├── docs/
│   └── style_transfer_implementation.md     ← this document
│
├── src/
│   ├── core/
│   │   ├── __init__.py
│   │   ├── models.py          # StyleModel dataclass, StyleStore (JSON persistence)
│   │   ├── registry.py        # StyleRegistry
│   │   ├── engine.py          # StyleTransferEngine (ONNX + PyTorch inference)
│   │   ├── trainer.py         # StyleTrainer (PyTorch training loop)
│   │   ├── tiling.py          # split_tiles / merge_tiles
│   │   └── photo_manager.py   # PhotoManager (Pillow + OpenCV I/O)
│   │
│   ├── ml/
│   │   ├── __init__.py
│   │   ├── transformer_net.py # TransformerNet architecture
│   │   ├── vgg_loss.py        # VGGPerceptualLoss (VGG-16 feature extractor)
│   │   └── train_utils.py     # dataset loaders, augmentation, scheduler
│   │
│   ├── ui/
│   │   ├── __init__.py
│   │   ├── main_window.py     # QMainWindow + docking layout
│   │   ├── style_gallery.py   # StyleGalleryView (QListView + delegate)
│   │   ├── photo_canvas.py    # PhotoCanvasView (QGraphicsView)
│   │   ├── style_editor.py    # Add/Edit/Delete style dialog
│   │   ├── training_dialog.py # Training progress dialog
│   │   └── widgets/
│   │       ├── thumbnail_delegate.py
│   │       └── strength_slider.py
│   │
│   └── app.py                 # entry point: QApplication + MainWindow
│
├── styles/                    # bundled pretrained ONNX models + metadata
│   ├── catalog.json           # [{id, name, preview_path, model_path, ...}]
│   ├── candy/
│   │   ├── model.onnx
│   │   └── preview.jpg
│   ├── mosaic/
│   ├── rain_princess/
│   ├── udnie/
│   ├── starry_night/
│   └── abstract/
│
├── data/                      # optional: MS-COCO symlink/path config
│   └── README.md
│
├── tests/
│   ├── unit/
│   │   ├── test_registry.py
│   │   ├── test_engine.py
│   │   ├── test_tiling.py
│   │   └── test_photo_manager.py
│   ├── integration/
│   │   ├── test_training_pipeline.py
│   │   └── test_apply_full_photo.py
│   └── ui/
│       ├── conftest.py        # pytest-qt fixtures
│       ├── test_style_gallery.py
│       ├── test_photo_canvas.py
│       └── test_style_editor.py
│
├── scripts/
│   ├── download_pretrained.py # downloads all bundled ONNX models from GitHub releases
│   └── benchmark.py           # measures inference time per tile/image size
│
├── pyproject.toml             # PEP 621 project metadata
├── requirements.txt
├── requirements-dev.txt       # pytest, pytest-qt, mypy, ruff, etc.
└── README.md
```

---

## 9. Component Design Detail

### 9.1 `StyleModel` — data layer

```python
# src/core/models.py
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

@dataclass
class StyleModel:
    id: str                        # unique slug, e.g. "candy"
    name: str                      # display name, e.g. "Candy"
    model_path: Path               # path to .onnx (or .pth) file
    preview_path: Path             # thumbnail ~256x256 px
    description: str = ""
    author: str = ""
    source_images: list[Path] = field(default_factory=list)  # style reference images
    is_builtin: bool = True        # False for user-created styles
    training_config: Optional[dict] = None  # stored hyperparameters
```

`StyleStore` provides JSON serialisation of the catalog (`styles/catalog.json`).

### 9.2 `StyleRegistry` — style management

```python
# src/core/registry.py
class StyleRegistry:
    def list_styles(self) -> list[StyleModel]: ...
    def get(self, style_id: str) -> StyleModel: ...
    def add(self, style: StyleModel) -> None: ...
    def update(self, style: StyleModel) -> None: ...
    def delete(self, style_id: str) -> None: ...  # removes model file + catalog entry
    def import_trained_model(self, pth_path: Path, style: StyleModel) -> StyleModel: ...
```

Emits Qt signals (via a thin adapter) when the catalog changes, so the UI gallery refreshes automatically.

### 9.3 `StyleTransferEngine` — inference

```python
# src/core/engine.py
class StyleTransferEngine:
    """Pure Python, no Qt dependency."""

    def load_model(self, model_path: Path) -> None:
        """Loads ONNX model into onnxruntime.InferenceSession."""

    def apply(
        self,
        content_image: Image.Image,
        style_id: str,
        strength: float = 1.0,    # 0.0–1.0
        tile_size: int = 1024,
        overlap: int = 128,
    ) -> Image.Image:
        """Main entry point. Handles tiling automatically for large images."""

    def preview(
        self,
        content_image: Image.Image,
        style_id: str,
        strength: float,
        max_dim: int = 512,
    ) -> Image.Image:
        """Fast preview at reduced resolution, no tiling needed."""
```

**Strength (gradual application):**  
After the styled image is produced, linear interpolation (alpha blending) is applied:

```python
result = Image.blend(content_image_resized, styled_image, alpha=strength)
```

This gives the 10%–100% effect without re-running the network.

### 9.4 `StyleTrainer` — training

```python
# src/core/trainer.py
class StyleTrainer:
    def __init__(self, device: str = "auto"): ...

    def train(
        self,
        style_images: list[Path],        # 1 or more reference images
        coco_dataset_path: Path,
        output_model_path: Path,
        epochs: int = 2,
        batch_size: int = 4,
        image_size: int = 256,
        style_weight: float = 1e8,
        content_weight: float = 1e5,
        learning_rate: float = 1e-3,
        checkpoint_path: Optional[Path] = None,  # resume if present
        progress_callback: Optional[Callable[[int, int, float], None]] = None,
    ) -> Path:                            # returns path to saved .pth model
        ...

    def export_onnx(self, pth_path: Path, onnx_path: Path) -> Path: ...
```

Training runs in a `QThread` so the UI stays responsive. Progress is reported via the `progress_callback`, which emits a Qt signal back to the `TrainingProgressDialog`.

### 9.5 `PhotoManager` — photo I/O

```python
# src/core/photo_manager.py
class PhotoManager:
    def load(self, path: Path) -> Image.Image:
        """Loads with Pillow, preserves EXIF, auto-rotates."""

    def save(self, image: Image.Image, path: Path, quality: int = 95) -> None:
        """Saves with Pillow; copies EXIF from source if available."""

    def thumbnail(self, image: Image.Image, max_size: tuple[int,int]) -> Image.Image: ...

    def split_tiles(
        self, image: Image.Image, tile_size: int, overlap: int
    ) -> list[TileInfo]: ...

    def merge_tiles(
        self, tiles: list[tuple[TileInfo, Image.Image]], output_size: tuple[int,int]
    ) -> Image.Image:
        """Gaussian feather-blend overlapping tile borders."""
```

### 9.6 UI Layer (PySide6) ✔ decided

#### Main Window Layout

```
┌─────────────────────────────────────────────────────────────┐
│  Menu: File | Styles | View | Help                          │
├──────────────────┬──────────────────────────────────────────┤
│  Style Gallery   │          Photo Canvas                    │
│  (dock, left)    │                                          │
│  ┌────┐ ┌────┐  │   ┌──────────────────────────────────┐   │
│  │ 🖼 │ │ 🖼 │  │   │                                  │   │
│  │Tia │ │Mos │  │   │       Original / Styled Photo    │   │
│  └────┘ └────┘  │   │       (split-view or overlay)    │   │
│  ┌────┐ ┌────┐  │   │                                  │   │
│  │ 🖼 │ │ 🖼 │  │   └──────────────────────────────────┘   │
│  │Rn  │ │Van │  │                                          │
│  └────┘ └────┘  │   Style: [Mosaic ▼]     Strength: ──●── │
│  [+ Add Style]  │   [Open Photo]  [Apply]  [Save Result]   │
│  [- Delete]     │                                          │
└──────────────────┴──────────────────────────────────────────┤
│  Status: Ready / Processing tile 3/12 ...                   │
└─────────────────────────────────────────────────────────────┘
```

#### Style Gallery UI
- `QListView` in `IconMode` with custom `QStyledItemDelegate`
- Each item shows: thumbnail image, style name, optional "user-created" badge
- Right-click context menu: Edit, Delete, Export model
- "Add Style" opens `StyleEditorDialog` (provides style name, drag-drop or browse for reference images, training hyperparameters accordion)

#### Photo Canvas UI
- `QGraphicsView` with `QGraphicsPixmapItem` for the photo
- Split-view slider (drag to compare original vs. styled)
- "Strength" `QSlider` (0–100), triggers preview re-render on mouse release (debounced 300 ms)
- Progress overlay during full-resolution processing

---

## 10. Gradual Style Application

The "10%–100%" slider does **not** re-run the neural network each time. Instead:

1. The `StyleTransferEngine` computes a **single fully styled image** (at preview resolution).
2. The strength value `α ∈ [0.0, 1.0]` blends the output with the original:
   ```
   output_pixel = α × styled_pixel + (1 − α) × original_pixel
   ```
3. This is a simple `PIL.Image.blend()` call — instantaneous.
4. Only when the user clicks **Apply / Save** is the full-resolution tiled inference run.

For the full-resolution apply, the slider value at the time of clicking is used as the blend factor after tiling.

---

## 11. Testing Strategy

### Philosophy

> **Tests are written and executed together with every implementation phase — never accumulated for a big bang at the end.**

Each phase has a mandatory "green gate": all tests belonging to that phase must pass before work on the next phase begins. The CI check is `pytest --cov=src --cov-fail-under=85`.

Every component outside platform-specific Qt rendering is covered by automated tests. Manual testing is limited to subjective visual quality assessment of style output (does the styled photo look good?).

### Test Layers

| Layer | Tools | Coverage target | When added |
|---|---|---|---|
| Unit — data models / registry | `pytest`, `pytest-mock` | 95%+ | Phase 2 |
| Unit — photo I/O | `pytest`, synthetic images | 90%+ | Phase 2 |
| Unit — tiling / blending | `pytest`, NumPy asserts | 90%+ | Phase 1 |
| Unit — ML model (TransformerNet) | `pytest`, tiny tensors | forward pass shape, dtype | Phase 1 |
| Unit — perceptual loss | `pytest`, tiny tensors | loss value sanity | Phase 1 |
| Unit — inference engine | `pytest` + candy ONNX fixture | output shape, value range | Phase 1 |
| Integration — training pipeline | `pytest`, toy COCO (10 imgs) | smoke: 1 step completes, loss decreases | Phase 1 |
| Integration — full apply flow | `pytest` + real ONNX | apply to a real JPEG, save, load | Phase 2 |
| UI — widgets | `pytest-qt` (`qtbot`) | 80%+ widget interactions | Phase 3 |
| UI — dialogs | `pytest-qt` | open/close, field validation, signal emission | Phase 3 |
| UI — training dialog | `pytest-qt` + `QThread` mock | progress signals fire correctly | Phase 3 |
| Integration — end-to-end | `pytest-qt` + real ONNX | open photo → apply → file saved to disk | Phase 4 |
| Performance regression | `scripts/benchmark.py` | tile inference < 2 s per 1024² tile on CPU | Phase 4 |

### Per-Phase Test Execution

```
Phase 1 ──► implement ML core ──► pytest tests/unit/test_transformer_net.py
                                   pytest tests/unit/test_vgg_loss.py
                                   pytest tests/unit/test_engine.py
                                   pytest tests/unit/test_tiling.py
                                   pytest tests/integration/test_training_pipeline.py
                                   ✅ All green → proceed to Phase 2

Phase 2 ──► implement data + photo ──► pytest tests/unit/test_registry.py
                                        pytest tests/unit/test_photo_manager.py
                                        pytest tests/integration/test_apply_full_photo.py
                                        ✅ All green → proceed to Phase 3

Phase 3 ──► implement UI ──► pytest tests/ui/test_style_gallery.py
                              pytest tests/ui/test_photo_canvas.py
                              pytest tests/ui/test_style_editor.py
                              pytest tests/ui/test_training_dialog.py
                              ✅ All green → proceed to Phase 4

Phase 4 ──► integration + polish ──► pytest tests/  (full suite)
                                      python scripts/benchmark.py
                                      ✅ Coverage ≥ 85%, benchmark within budget → release
```

### Fixtures & Test Data

| Fixture file | Contents | Used in |
|---|---|---|
| `tests/fixtures/candy.onnx` | Real candy ONNX model (~6 MB, MIT) | engine unit + integration tests |
| `tests/fixtures/sample_256.jpg` | 256×256 JPEG photo | all unit tests |
| `tests/fixtures/sample_4000.jpg` | 4000×3000 JPEG (12 MP) | tiling + full-apply integration |
| `tests/fixtures/style_ref.jpg` | A style reference painting | trainer smoke test |
| `tests/fixtures/mini_coco/` | 10 JPEG photos | trainer smoke test (1 step) |
| `tests/conftest.py` | Shared `pytest` fixtures: `engine`, `registry`, `qtbot` app | all tests |

### Key Test Examples

```python
# tests/unit/test_tiling.py
def test_round_trip_preserves_size(sample_12mp):
    """Splitting and merging must return image of identical dimensions."""
    manager = PhotoManager()
    tiles = manager.split_tiles(sample_12mp, tile_size=1024, overlap=128)
    reconstructed = manager.merge_tiles(tiles, sample_12mp.size)
    assert reconstructed.size == sample_12mp.size

def test_seam_blend_is_smooth(sample_12mp, candy_engine):
    """No hard pixel edge should appear at tile boundaries."""
    result = candy_engine.apply(sample_12mp, "candy", strength=1.0)
    arr = np.array(result).astype(float)
    # gradient across known seam boundary must be below threshold
    seam_col = 1024 - 128  # left seam column
    diff = np.abs(arr[:, seam_col + 1, :] - arr[:, seam_col, :]).mean()
    assert diff < 15.0   # empirically tuned; hard edge would be 50+

# tests/unit/test_engine.py
def test_strength_zero_returns_original(engine, sample_photo):
    result = engine.apply(sample_photo, "candy", strength=0.0)
    assert np.allclose(np.array(result), np.array(sample_photo), atol=1)

def test_strength_one_returns_fully_styled(engine, sample_photo):
    styled = engine.apply(sample_photo, "candy", strength=1.0)
    # styled must differ from original by a measurable amount
    diff = np.abs(np.array(styled).astype(float) - np.array(sample_photo).astype(float)).mean()
    assert diff > 5.0

# tests/ui/test_style_gallery.py  (pytest-qt)
def test_add_button_opens_editor_dialog(qtbot, main_window):
    qtbot.mouseClick(main_window.gallery.add_button, Qt.LeftButton)
    assert main_window.style_editor_dialog.isVisible()

def test_delete_removes_item_from_model(qtbot, main_window, user_style):
    n_before = main_window.gallery.model().rowCount()
    main_window.registry.delete(user_style.id)
    assert main_window.gallery.model().rowCount() == n_before - 1

# tests/ui/test_training_dialog.py
def test_warning_dialog_shown_before_training(qtbot, training_dialog):
    """Pre-flight warning must appear and must require user confirmation."""
    with qtbot.waitSignal(training_dialog.user_confirmed, timeout=500):
        qtbot.mouseClick(training_dialog.start_button, Qt.LeftButton)
    assert training_dialog.warning_label.isVisible()

def test_progress_bar_updates_during_training(qtbot, training_dialog, mock_trainer):
    mock_trainer.emit_progress(batch=5, total=100, eta_seconds=7200)
    assert training_dialog.progress_bar.value() == 5
    assert "2h" in training_dialog.eta_label.text()
```

---

## 12. Dependencies & Environment

### Runtime Dependencies (`requirements.txt`)

```
torch>=2.3.0
torchvision>=0.18.0
onnxruntime-directml>=1.18.0  # DirectML for Intel Arc / AMD on Windows; swap for onnxruntime (CPU) or onnxruntime-gpu (Nvidia)
Pillow>=10.3.0
opencv-python>=4.9.0
PySide6>=6.7.0
numpy>=1.26.0
```

### Development Dependencies (`requirements-dev.txt`)

```
pytest>=8.0
pytest-qt>=4.4
pytest-mock>=3.14
pytest-cov>=5.0
ruff                        # linting + formatting
mypy                        # type checking
tensorboard                 # training monitoring
```

### Python Environment

Requires Python 3.11 installed system-wide (e.g. from [python.org](https://www.python.org/downloads/)).  
A dedicated virtual environment is created with the built-in `venv` module.

```powershell
# Create the virtual environment (run once, from the project root)
python -m venv .venv

# Activate it (PowerShell — required before every session)
.\.venv\Scripts\Activate.ps1

# Install runtime and development dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

> **Note (Windows PowerShell execution policy):** If `Activate.ps1` is blocked, run once:  
> `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`

The `.venv/` folder is listed in `.gitignore` and never committed.

### GPU Acceleration (Intel Arc 140V — this laptop)

This machine has an **Intel Arc 140V GPU** and no Nvidia GPU. CUDA is not available.  
Use `onnxruntime-directml` to enable DirectML inference acceleration on the Intel Arc via Windows DirectX 12:

```powershell
# Replace plain onnxruntime with the DirectML-enabled build
pip uninstall onnxruntime -y
pip install onnxruntime-directml
```

The `StyleTransferEngine` already passes `["DmlExecutionProvider", "CPUExecutionProvider"]` as the provider fallback chain, so no code change is needed.

### Optional: Nvidia CUDA (other machines only)

```powershell
# Replace cu121 with your installed CUDA version (check: nvidia-smi)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip uninstall onnxruntime-directml -y
pip install onnxruntime-gpu
```

---

## 13. Implementation Phases / Roadmap

> **Rule:** Tests for every item are written and run within the same phase. The next phase starts only after all tests are green.

---

### Phase 1 — Core ML Pipeline (no UI)

**Goal:** TransformerNet inference works end-to-end with a pretrained ONNX model.

Implementation tasks:
- [ ] Create project structure + `pyproject.toml` + CI runner (`pytest` on every commit)
- [ ] Implement `TransformerNet` architecture (`src/ml/transformer_net.py`)
- [ ] Implement `VGGPerceptualLoss` with VGG-16 feature layers (`src/ml/vgg_loss.py`)
- [ ] Implement training loop + COCO dataloader + TensorBoard logging (`src/core/trainer.py`, `src/ml/train_utils.py`)
- [ ] Implement `StyleTransferEngine` — ONNX inference + tiling + strength blend (`src/core/engine.py`, `src/core/tiling.py`)
- [ ] Download & commit 6 pretrained ONNX models from GitHub releases (`styles/`)
- [ ] `scripts/download_pretrained.py` — repeatable one-shot download

Test tasks (must be green before Phase 2):
- [ ] `tests/fixtures/` — commit candy.onnx, sample_256.jpg, sample_4000.jpg, mini_coco/ (10 imgs), style_ref.jpg
- [ ] `tests/unit/test_transformer_net.py` — forward pass: correct output shape & dtype
- [ ] `tests/unit/test_vgg_loss.py` — content loss and style loss are positive scalars; style loss decreases when input moves toward style image
- [ ] `tests/unit/test_engine.py` — `strength=0` returns original, `strength=1` returns styled, output size matches input
- [ ] `tests/unit/test_tiling.py` — round-trip preserves size; seam gradient below threshold
- [ ] `tests/integration/test_training_pipeline.py` — one gradient step on toy dataset completes; loss is a finite number
- [ ] `scripts/benchmark.py` — baseline: tile inference < 2 s per 1024² tile on CPU

**Green gate:** `pytest tests/unit tests/integration -v` → all pass, coverage ≥ 85% for `src/ml/` and `src/core/engine.py`.

---

### Phase 2 — Style Management & Photo I/O

**Goal:** Styles can be listed, created, updated, and deleted via pure-Python API; photos can be loaded, saved, and tiled.

Implementation tasks:
- [ ] `StyleModel` dataclass + `StyleStore` JSON persistence (`src/core/models.py`)
- [ ] `StyleRegistry` CRUD operations (`src/core/registry.py`)
- [ ] `PhotoManager` — JPEG/PNG load/save with EXIF, thumbnail, split/merge tiles (`src/core/photo_manager.py`)
- [ ] Validate file-format constraint: reject non-JPEG/PNG with descriptive error

Test tasks (must be green before Phase 3):
- [ ] `tests/unit/test_registry.py` — add/list/get/update/delete; duplicate ID raises; unknown ID raises
- [ ] `tests/unit/test_photo_manager.py` — load JPEG, load PNG, reject unsupported format, thumbnail dimensions, EXIF round-trip
- [ ] `tests/integration/test_apply_full_photo.py` — load sample_4000.jpg → apply candy style → save output → reload and verify dimensions

**Green gate:** `pytest tests/unit tests/integration -v` → all pass, coverage ≥ 90% for `src/core/`.

---

### Phase 3 — Desktop GUI (PySide6)

**Goal:** All UI components are functional and independently tested via `pytest-qt`.

Implementation tasks:
- [ ] Scaffold `QMainWindow` with dock layout (`src/ui/main_window.py`)
- [ ] `StyleGalleryView` — `QListView` with custom thumbnail delegate, right-click menu, Add/Delete buttons (`src/ui/style_gallery.py`, `src/ui/widgets/thumbnail_delegate.py`)
- [ ] `PhotoCanvasView` — `QGraphicsView` with split-view compare and strength slider (`src/ui/photo_canvas.py`, `src/ui/widgets/strength_slider.py`)
- [ ] `StyleEditorDialog` — add/edit style: name, browse reference images, advanced hyperparameters accordion (`src/ui/style_editor.py`)
- [ ] `TrainingProgressDialog` — pre-flight warning, live progress bar with ETA, cancel/resume (`src/ui/training_dialog.py`)
- [ ] Wire all components to service layer through Qt signals/slots

Test tasks (must be green before Phase 4):
- [ ] `tests/ui/conftest.py` — shared `qtbot` fixtures: headless app, pre-loaded registry with candy style
- [ ] `tests/ui/test_style_gallery.py` — Add button opens editor; delete removes item; clicking style thumbnail updates canvas preview; right-click menu shows correct items
- [ ] `tests/ui/test_photo_canvas.py` — open photo updates canvas; strength slider emits signal; split-view boundary is draggable
- [ ] `tests/ui/test_style_editor.py` — empty name blocked; browse button opens file dialog; save emits `style_saved` signal
- [ ] `tests/ui/test_training_dialog.py` — warning shown before start; progress bar value matches emitted signal; cancel stops the QThread

**Green gate:** `pytest tests/ui -v` → all pass, coverage ≥ 80% for `src/ui/`.

---

### Phase 4 — Integration, Polish & Performance

**Goal:** Full user journeys work end-to-end; edge cases are handled gracefully.

Implementation tasks:
- [ ] Error handling: corrupted/missing model file, OOM during tiling, COCO path not found
- [ ] User settings dialog (tile size, default output folder, GPU/CPU preference, overlap width)
- [ ] Performance tuning: optional float16 tile inference; configurable tile size in settings
- [ ] About dialog with attribution (MIT licences for yakhyo + igreat models)

Test tasks:
- [ ] `tests/integration/test_end_to_end.py` — `pytest-qt` full journey: open photo → select style → adjust strength → Apply → file saved to temp dir → verify file exists and dimensions
- [ ] `tests/integration/test_training_end_to_end.py` — toy dataset → train 1 step → ONNX export → new style appears in gallery (using QThread mock to skip real training time)
- [ ] `tests/unit/test_error_handling.py` — missing model file raises `StyleModelNotFoundError`; unsupported format raises `UnsupportedFormatError`; invalid strength value raises `ValueError`
- [ ] `scripts/benchmark.py` — re-run benchmark; assert tile inference stays within budget; log to `benchmarks.log`

**Green gate:** Full `pytest tests/ -v --cov=src --cov-fail-under=85` → all pass; benchmark within budget.

---

### Phase 5 — Packaging (optional)

- [ ] `PyInstaller` spec file for standalone Windows `.exe`
- [ ] Bundle ONNX models, PySide6 binaries, ONNX Runtime in single distributable package
- [ ] Smoke-test the packaged `.exe` on a clean Windows machine

---

## 14. Decisions & Resolved Risks

All risks from the initial draft have been reviewed and decided. The table below records the decision for each item; no open questions remain.

| # | Topic | Decision / Resolution | Status |
|---|---|---|---|
| 1 | **Training time on CPU-only laptop** | **Both** strategies are implemented: (a) all bundled styles are pre-installed as ready-to-use ONNX models — no training required; (b) when the user creates a custom style, a pre-flight warning dialog shows estimated duration before training starts, and a live progress bar with ETA is shown throughout training. Checkpoint-based resume is supported. | ✅ Decided |
| 2 | **MS-COCO dataset size (~15 GB)** | Training dialog guides the user to their local COCO path. `scripts/download_pretrained.py` covers all bundled styles so most users never need to train at all. | ✅ Decided |
| 3 | **Image format support** | **JPEG and PNG only.** No HEIC/HEIF. `PhotoManager` accepts `.jpg`, `.jpeg`, `.png`; all other extensions are rejected with a clear error message in the UI. | ✅ Decided |
| 4 | **12 MP tiling seam artefacts** | Gaussian-weighted overlap blend (128 px default). Overlap width is user-configurable in the Settings dialog. Integration tests verify seam quality with a real ONNX model. | ✅ Decided |
| 5 | **VGG-16 download on first training run** | `torchvision.models.vgg16(pretrained=True)` downloads automatically to `~/.cache/torch`. A loading spinner is shown in the training dialog until the download completes. | ✅ Decided |
| 6 | **Licence of bundled ONNX models** | yakhyo and igreat repos are MIT-licensed. Attribution is shown in the application's **Help → About** dialog. | ✅ Decided |
| 7 | **GUI framework** | **PySide6** (LGPL). This is locked. All UI code uses `from PySide6 import ...`. | ✅ Decided |
| 8 | **Memory for 12 MP float32 tile** | 1024² tile = ~12 MB; completely manageable. No action required. | ✅ Decided |

---

## 15. Background: What is Perceptual Loss (VGG-16)?

### The Core Idea

A naive way to train the TransformerNet would be to compare the styled output pixel-by-pixel with the target style image and minimise the difference. This fails: pixel-level comparison has no notion of visual *meaning*, textures, brush strokes, or artistic patterns. Small pixel shifts that look visually identical can produce large pixel-level losses, and vice versa.

**Perceptual loss** solves this by measuring similarity in the space of *high-level image features*, not raw pixels.

### What VGG-16 Does

VGG-16 is a Convolutional Neural Network trained by the Visual Geometry Group at Oxford on ImageNet (1.2 million photos, 1000 categories — cats, cars, chairs, landscapes, etc.). Through training it learned to recognise meaningful visual patterns at multiple levels of abstraction:

```
Input image (3 × H × W)
        │
   conv1_1  →  edges, lines, basic colour gradients        (pixel-level)
   conv1_2  →
        │  Pool 1
   conv2_1  →  textures, simple patterns (stripes, dots)
   conv2_2  →
        │  Pool 2
   conv3_1  →  complex textures, repeating artistic motifs
   conv3_2  →
   conv3_3  →
        │  Pool 3
   conv4_1  →  object parts (eyes, branches, arches)
   conv4_2  →  ◄── used for CONTENT loss
   conv4_3  →
        │  Pool 4
   conv5_1  →  whole objects, scene layout
   conv5_2  →
   conv5_3  →  ◄── used for STYLE loss (Gram matrix, see below)
        │  Pool 5
   fc6 / fc7 / fc8  →  ImageNet class predictions
```

VGG-16 is **frozen** (its weights never change) during style transfer training. It is used purely as a read-only feature extractor — a sophisticated way to "look at" an image and describe it in terms of patterns rather than raw pixels.

### Content Loss

To preserve the *content* of the photo being styled (the subject, the spatial layout, the key edges), the content loss compares the mid-level feature activations at layer `relu4_2` between the original photo and the styled output:

```
content_loss = MSE( VGG_relu4_2(styled_output) ,  VGG_relu4_2(original_photo) )
```

If the styled photo distorts the scene too much (e.g., a person's face disappears), the feature maps at `relu4_2` diverge, and the content loss grows, pushing the TransformerNet to correct this.

### Style Loss (Gram Matrix)

To capture the *style* — the texture, colour palette, and brushstroke patterns of the reference artist — the style loss uses the **Gram matrix** of feature maps. The Gram matrix captures the *correlation* between different feature channels (how often a stroke pattern co-occurs with a colour gradient), which encodes texture statistics independently of where in the image they appear.

```
For a feature map F of shape [channels × height × width]:

G = F_reshaped × F_reshaped^T      shape: [channels × channels]

style_loss = Σ_l  MSE( G_l(styled_output) ,  G_l(style_reference_image) )
```

Style loss is summed across multiple VGG layers (`relu1_2`, `relu2_2`, `relu3_3`, `relu4_3`, `relu5_3`) to capture texture at both fine (brushstroke-level) and coarse (composition-level) scales.

### Combined Training Objective

The TransformerNet is trained to minimise:

```
total_loss = content_weight × content_loss  +  style_weight × style_loss
```

Typical values: `content_weight = 1e5`, `style_weight = 1e8`.  
The ratio between these two weights controls the balance between *"the photo still looks like its subject"* and *"the photo strongly resembles the artist's style"*.

### Why VGG-16 Specifically?

| Reason | Detail |
|---|---|
| **Rich feature hierarchy** | 13 conv layers capture patterns at many different scales |
| **ImageNet pretraining** | Already understands both low-level textures and high-level content |
| **Established benchmark** | The original Johnson 2016 paper used VGG-16; all reference implementations follow it |
| **Manageable size** | ~138 M parameters, but only a forward pass is needed (no backprop into VGG) |
| **PyTorch built-in** | `torchvision.models.vgg16(weights="DEFAULT")` — one line of code |

After training is complete, VGG-16 is **no longer needed at all**. The TransformerNet has "baked in" the style knowledge it learned. Inference uses only TransformerNet + ONNX Runtime.

---

## 16. Background: What is ONNX?

### The Problem ONNX Solves

The ML world has many runtime frameworks: PyTorch, TensorFlow, JAX, TensorRT, CoreML, and more. A model trained in PyTorch cannot directly run in TensorFlow and vice versa. Deploying a model to different hardware backends (Nvidia GPU, AMD GPU, CPU-only, mobile) requires converting or rewriting it for each target. This is expensive and fragile.

**ONNX (Open Neural Network Exchange)** is an open standard that acts as a universal interchange format and runtime for neural networks. It decouples *how a model is trained* from *how it is executed*.

### What ONNX Is

```
PyTorch training (.pth)
        │
        │  torch.onnx.export()   (one function call after training)
        ▼
  ┌─────────────┐
  │  .onnx file │  ← open standard: defines a computational graph
  │             │    using standardised operators (Conv, ReLU, etc.)
  │  ~6–10 MB   │    with baked-in trained weights
  └──────┬──────┘
         │   onnxruntime.InferenceSession("model.onnx")
         ▼
  ONNX Runtime  ──►  CPU (Windows / Linux / Mac)
                ──►  Nvidia GPU  (CUDA Execution Provider)
                ──►  AMD / Intel GPU  (DirectML Execution Provider, Windows)
                ──►  TensorRT  (Nvidia optimised, production servers)
                ──►  CoreML  (Apple devices)
```

### The ONNX File

An ONNX file is a serialised **computational graph** stored in Protocol Buffer format. It contains:
- **Nodes:** mathematical operations (`Conv`, `InstanceNorm`, `Relu`, `Upsample`, `Add`, ...)
- **Edges:** tensors flowing between nodes with their shape and data type declared
- **Weights:** all trained numerical parameters are embedded directly in the file
- **I/O specification:** named inputs/outputs with declared shapes

A complete candy-style TransformerNet compresses to ~6 MB.

### How ONNX Runtime Executes It

When `onnxruntime.InferenceSession` loads an ONNX file it:
1. **Optimises the graph** — fuses adjacent operators (e.g., `Conv + ReLU → ConvReLU`) and eliminates redundant allocations
2. **Selects the best execution provider** — CUDA if an Nvidia GPU is present; DirectML for AMD; otherwise falls back to the CPU implementation
3. **Runs inference** — accepts a NumPy array as input, returns a NumPy array as output

```python
import onnxruntime as ort
import numpy as np

# Load once at application startup
session = ort.InferenceSession(
    "styles/candy/model.onnx",
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"]  # fallback chain
)

# Run inference for one tile
input_tensor = np.array(tile_image, dtype=np.float32)   # shape [1, 3, H, W]
output = session.run(None, {"input": input_tensor})
styled_tile = output[0]  # shape [1, 3, H, W]
```

Typical throughput: **200–800 ms per 1024×1024 tile on CPU**, **20–80 ms on a mid-range GPU**.

### Why ONNX for This Application

| Benefit | Detail |
|---|---|
| **No PyTorch needed at runtime** | Users who only apply styles never need to install PyTorch (~2 GB) |
| **Faster CPU inference** | ONNX Runtime is 2–3× faster than PyTorch CPU for inference on the same model |
| **Hardware portability** | One `.onnx` file runs on Nvidia, AMD, and Intel graphics via different execution providers |
| **Stable format** | ONNX opset versioning ensures models remain loadable as library versions update |
| **Small file per style** | ~6 MB per style model — all 6 bundled styles total ~40 MB |

### ONNX Export — Key Detail for Tiling

The export must use `dynamic_axes` to allow variable tile dimensions. Without this, the ONNX model is fixed to a single input size and will fail on edge tiles (which may be smaller than 1024×1024 when the image size is not an exact multiple):

```python
torch.onnx.export(
    transformer_net,
    dummy_input,                          # shape [1, 3, 1024, 1024]
    "styles/candy/model.onnx",
    opset_version=11,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={
        "input":  {2: "height", 3: "width"},   # H, W are variable
        "output": {2: "height", 3: "width"},
    }
)
```

This single option is what makes the tiling strategy work correctly for all image sizes.

---

*End of proposal — approved for implementation.*
