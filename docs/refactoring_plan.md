# Refactoring Plan

**Created**: 2026-05-02  
**Revised**: 2026-05-03  
**Scope**: `src/`, `scripts/`, `tests/`, top-level files

Items marked ✅ DONE have been implemented and committed since the plan was first written.  
Items marked ~~crossed out~~ have been found to be non-issues on closer inspection.

---

## Summary table

| # | Severity | Category | File(s) | Approx. saving | Status |
|---|----------|----------|---------|----------------|--------|
| 1 | **critical** | God class | `src/stylist/main_window.py` | −350 lines | open |
| 2 | **serious** | Duplicate logic + missing logging | `src/core/engine.py` | −8 lines / 3 sites | open |
| 3 | **serious** | Duplicate logic | `main_window.py` | −8 lines / 2 sites | open |
| 4 | **serious** | ~~Breaks encapsulation~~ | ~~`batch_styler.py`~~ | — | ✅ DONE |
| 5 | **serious** | Duplicate concept | `main_window.py` + `src/batch_styler/` | −12 lines | open |
| 6 | **serious** | Missing serialisation | `style_chain_schema.py` / `main_window.py` | consolidates YAML logic | open |
| 7 | **serious** | Dead directories | `src/ml/`, `src/ui/` | removes 2 ghost trees | open |
| 8 | **minor** | Dual attr aliasing | `main_window.py` | removes 3 confusing aliases | open |
| 9 | **minor** | Magic number | `src/batch_styler/commands.py` | 1 hardcoded value | open |
| 10 | **minor** | ~~Stale import~~ | ~~`src/stylist/app.py`~~ | — | ~~invalid~~ |
| 11 | **minor** | Structure | `scripts/` (training sub-group) | separation of concerns | open |
| 12 | **minor** | Structure | top-level launchers + `bin/` | entry-point consolidation | open |

---

## 1 (critical) — God class: `MainWindow` (746 lines)

`src/stylist/main_window.py` is 746 lines and mixes six distinct responsibilities in one class:

| Responsibility | Methods |
|----------------|---------|
| Menu/toolbar construction | `_build_menus`, `_build_ui`, `_wire_signals` |
| Photo I/O | `_open_photo`, `_save_result`, `_reset_photo` |
| Style application | `_apply_style`, `_reapply_style`, `_reapply_style_strength` |
| Background worker + progress dialog | `_run_apply_worker`, `_create_progress_dialog` |
| Undo stack | `_push_undo_snapshot`, `_clear_undo_stack`, `_perform_undo` |
| Style-chain / replay log | `_format_replay_log`, `_copy_replay_log_to_clipboard`, `_load_and_apply_replay_log`, `_resolve_style_id_by_name` |
| Help dialogs | `_show_link_dialog`, `_show_how_to_use`, `_show_about_nst`, `_show_credits` |

**Proposed split** (three extraction candidates):

1. **`ApplyController`** (new `src/stylist/apply_controller.py`) — holds the three apply variants, the progress-dialog helper, and `_run_apply_worker`. It owns no Qt widgets; it only receives signals and emits results. `MainWindow` delegates to it.

2. **`StyleChainController`** (new `src/stylist/style_chain_controller.py`) — holds the chain format/copy/load/apply methods and `_resolve_style_id_by_name`. Already a logically complete unit.

3. **Help dialogs** — move the three static HTML blobs and `_show_link_dialog` to a standalone `help_dialogs.py` module so `MainWindow` calls one function rather than hosting the content inline.

After extraction `MainWindow` would shrink to ~350 lines (construction, signal wiring, state bookkeeping).

---

## 2 (serious) — OOM detection duplicated 3 times in `engine.py` + original error not logged

The same 4-line exception guard appears verbatim in `_infer_tile()`, `_infer_tile_nhwc_tanh()`, and `_infer_tile_nchw_tanh()`:

```python
_msg = str(exc).lower()
if any(k in _msg for k in ("out of memory", "insufficient", "oom", ": 6 :", "error code: 6")):
    raise OOMError(...)
```

**Two problems**:

1. **Duplication** — the keyword list is copied verbatim; a future keyword addition must be made three times.
2. **Original ONNX error is silently discarded** — the `OOMError` message is a user-friendly paraphrase, but the raw ONNX exception text (which contains the exact DirectML/CUDA error code and wording) is never written to `app.log`. This makes post-mortem diagnosis hard.

**Proposed fix**:

```python
def _reraise_if_oom(self, exc: Exception, tile_size: tuple[int, int]) -> None:
    """Log the raw ONNX error and re-raise as OOMError if it indicates OOM."""
    _msg = str(exc).lower()
    if any(k in _msg for k in ("out of memory", "insufficient", "oom", ": 6 :", "error code: 6")):
        logger.error("OOM — raw ONNX error: %s", exc)         # ← exact wording in app.log
        raise OOMError(
            f"GPU/DirectML out of memory processing a tile of size {tile_size}. "
            "Open a new photo or reduce tile_size in Settings to free memory."
        ) from exc
```

All three inference methods call `self._reraise_if_oom(exc, tile.size)` in their `except Exception` blocks.  
Saves ≈ 8 lines; one place to update the keyword list; raw ONNX error always captured in `app.log` at ERROR level.  
Note: the `except MemoryError` path (pure Python OOM) should also call `logger.error("OOM — MemoryError: %s", exc)` before re-raising.

---

## 3 (serious) — `project_root` resolution duplicated in `MainWindow` (2 internal sites)

The frozen/dev conditional:

```python
project_root = (
    Path(sys.executable).parent
    if getattr(sys, "frozen", False)
    else Path(__file__).parent.parent.parent
)
```

appears **twice** inside `MainWindow`: at line 227 (in `_on_style_selected()`) and at line 676 (in `_load_and_apply_replay_log()`).

Note: the equivalent pattern in `batch_styler.py` was the module-level `REPO_ROOT`; that is now the canonical implementation in `src/batch_styler/catalog.py`.

**Proposed fix**: add a module-level `_get_project_root() -> Path` helper in `src/core/utils.py` (or directly in `src/stylist/main_window.py` as a module-level constant) called from both `MainWindow` sites.  
Saves ≈ 8 lines; one place to update if the bundle layout ever changes.

---

## 4 (serious) — ✅ DONE — Direct access to `engine._sessions` from `batch_styler.py`

**Resolved in commit `74b3bb7`** when `scripts/batch_styler.py` was refactored into the `src/batch_styler/` package.

Both call-sites in the new `src/batch_styler/commands.py` already use `engine.unload_model(style_id)` (lines 86 and 151).  No action needed.

---

## 5 (serious) — Style name lookup duplicated across the boundary

Two parallel implementations of "look up a style by display name":

| Location | Implementation |
|----------|---------------|
| `main_window.py._resolve_style_id_by_name(name)` | iterates `self._registry.list_styles()` |
| `src/batch_styler/catalog.py.filter_styles_by_name(styles, name)` | iterates a raw dict list loaded from JSON |

Both do the same case-insensitive match; they diverge only because `src/batch_styler/` bypasses `StyleRegistry` and reads the catalog JSON directly into plain dicts.

**Proposed fix**:

1. Let `src/batch_styler/commands.py` instantiate `StyleRegistry(REPO_ROOT / "styles" / "catalog.json")` and use it instead of the raw JSON dict.
2. Add a convenience method `StyleRegistry.find_by_name(name: str) -> StyleModel | None` (case-insensitive) that both callers can use.
3. Delete `filter_styles_by_name()` from `src/batch_styler/catalog.py`.

Also eliminates all the raw-dict boilerplate currently spread across the batch-styler commands.  
Saves ≈ 12 lines; unifies the lookup contract.

---

## 6 (serious) — `style_chain_schema.py` only deserialises; serialisation lives inline in `MainWindow`

> Note: `replay_schema.py` was renamed to `style_chain_schema.py` in commit `3fc7b6a`.

`style_chain_schema.py` exposes `load_style_chain()` but has no matching serialisation function.  The YAML serialisation is implemented inline in `MainWindow._format_replay_log()`.  This means:
- The data format is defined in the schema module but written in the UI layer.
- `src/batch_styler/commands.py` has no way to produce a style-chain YAML without duplicating the format.

**Proposed fix**: add `dump_style_chain(log: StyleChainLog) -> str` (a YAML serialisation function) to `style_chain_schema.py`.  `_format_replay_log()` in `MainWindow` becomes a one-liner that calls it.

---

## 7 (serious) — Ghost directory trees

Two directories exist in the repo but contain no source files (only `__pycache__`):

| Path | Content | Notes |
|------|---------|-------|
| `src/ml/` | only `__pycache__/` | abandoned subpackage placeholder |
| `src/ui/widgets/` | only `__pycache__/` | real widgets are in `src/stylist/widgets/` |

> Note: `tests/unit/` and `tests/integration/` from the original plan have been verified as already removed.

**Proposed fix**: delete both directory trees; ensure `.gitignore` covers any stray `__pycache__` left behind.

---

## 8 (minor) — Dual aliasing of injected services in `MainWindow.__init__`

```python
self.registry = registry        # public
self._registry = registry       # private alias
self.engine = engine
self._engine = engine
self.photo_manager = photo_manager
self._photo_manager = photo_manager
```

Every injected service is stored under two names.  All internal code uses the private alias (`self._registry`, etc.); the public name exists solely so tests can inspect state.  This adds noise and risks one alias going stale.

**Proposed fix**: keep only the public attribute (`self.registry`, `self.engine`, `self.photo_manager`); replace all private-alias usages inside the class with the public name.  Tests already refer to the public names.  
Saves 3 lines; removes the risk of aliases diverging.

---

## 9 (minor) — Magic JPEG quality number in `src/batch_styler/commands.py`

`cmd_apply_style_chain()` hardcodes `quality=92`:

```python
result.save(out_path, format="JPEG", quality=92)   # commands.py line 213
```

`PhotoManager.save()` defaults to `quality=95`.  The two values differ silently.

**Proposed fix**: define `JPEG_QUALITY: int = 92` as a named constant at the top of `commands.py` (or use `PhotoManager.save()` directly so there is a single authoritative default) and reference the constant.

---

## 10 (minor) — ~~Stale import in `src/stylist/app.py`~~ — INVALID

Originally flagged as unused, but `DEFAULT_TILE_SIZE` and `DEFAULT_OVERLAP` are actually used in `app.py` (lines 129–130) to reset settings to defaults.  
No action needed.

---

## 11 (minor) — `scripts/` training sub-group needs a home

`scripts/batch_styler.py` was the user-facing CLI — it is now the `src/batch_styler/` package (committed in `74b3bb7`).  What remains in `scripts/` is a mix of audiences:

| File | Audience | Kind |
|------|----------|------|
| `add_style_helper.py` | ML engineers / developers | Jupyter notebook backend |
| `kaggle_training_helper.py` | ML engineers | Kaggle-specific training CLI |
| `add_CNN_style.ipynb` | ML engineers | CNN training walk-through |
| `add_GAN_style.ipynb` | ML engineers | GAN / CycleGAN walk-through |
| `download_art.ipynb` | researchers | dataset acquisition (stays in `scripts/`) |
| `kaggle_trainer.ipynb` | ML engineers | Kaggle training notebook |
| `kaggle_multi_pic_trainer.ipynb` | ML engineers | Kaggle multi-style notebook |
| `style_analysis.ipynb` | researchers / developers | style-weight exploration |
| `benchmark.py` | developers | inference performance tool |
| `gen_palette_ico_temp.py` | developers | one-off asset generator |
| `index.md` | documentation | notebook index (update after move) |

The kaggle/training group (`kaggle_training_helper.py`, `add_style_helper.py`, and the four training notebooks) forms one coherent functional unit for training new styles.  `add_style_helper.py` is their shared Python backend.

**Decision: `training/` at project root**

```
training/
  add_style_helper.py           ← shared backend for both training notebooks
  kaggle_training_helper.py     ← Kaggle CLI
  add_CNN_style.ipynb
  add_GAN_style.ipynb
  kaggle_trainer.ipynb
  kaggle_multi_pic_trainer.ipynb
scripts/
  download_art.ipynb            ← stays here (not training-specific)
  benchmark.py
  gen_palette_ico_temp.py
  style_analysis.ipynb
  index.md                      ← update to reflect the move
```

`training/` at root makes it a first-class workflow alongside `src/`, `tests/`, and `docs/`.  Developers looking to train a new style know exactly where to start.

`add_style_helper.py` must update its relative imports if it currently uses `..` paths (verify before moving).  `index.md` in `scripts/` must be updated to remove the moved notebooks and add a pointer to `training/`.

---

## 12 (minor) — Top-level launcher files — consistent entry-point pattern

There are currently three application entry points, handled inconsistently:

| Entry point | Current location | Nature |
|-------------|-----------------|--------|
| `PetersPictureStyler.exe` | `src/stylist/app.py` | thin `QApplication` bootstrap |
| `BatchStyler.exe` | `src/batch_styler/app.py` | argparse CLI (≈ 100 lines, real logic) |
| `main_style_trainer.py` (dev only) | project root | full 180-line trainer CLI |
| `main_image_styler.py` (dev only) | project root | 16-line thin wrapper |

`main_style_trainer.py` is the odd one out: it is a full CLI with real logic at the repo root, while the two compiled apps live inside `src/*/app.py`.  `main_image_styler.py` is already a thin stub.

**Decision: implement all three phases.**

---

**Phase 1 — move trainer logic to `src/trainer/app.py`**

Mirror the pattern already established by `src/batch_styler/app.py` and `src/stylist/app.py`:

- Create `src/trainer/app.py` containing the full CLI logic currently in `main_style_trainer.py`.
- Reduce `main_style_trainer.py` to a thin stub (see Phase 3).

After this, all real logic lives in `src/`, and the two root stubs are identical in structure.

**Phase 2 — register all three as `pyproject.toml` console scripts**

```toml
[project.scripts]
stylist       = "src.stylist.app:main"
batch-styler  = "src.batch_styler.app:main"
trainer       = "src.trainer.app:main"
```

Developers can run `stylist`, `batch-styler`, and `trainer` from any directory after `pip install -e .`.  The `style_transfer.spec` build still references the raw `src/*/app.py` files for PyInstaller.

**Phase 3 — move root stubs to `bin/`**

```
bin/
  main_image_styler.py   ← thin stub, delegates to src.stylist.app:main
  main_style_trainer.py  ← thin stub, delegates to src.trainer.app:main
  README.md              ← explains why these stubs must be kept
```

Update `style_transfer.spec` `Analysis()` paths:
- `["src/stylist/app.py"]` — unchanged (GUI exe drives directly from src)
- `["bin/main_image_styler.py"]` if the GUI exe is ever built from the stub instead

`kaggle_training_helper.py` spawns `main_style_trainer.py` as a subprocess via:

```python
cmd = [sys.executable, str(self._repo_dir / "main_style_trainer.py"), "train", ...]
```

After the move to `bin/`, this path becomes `self._repo_dir / "bin" / "main_style_trainer.py"`.  Update `kaggle_training_helper.py` accordingly.

**Content of `bin/README.md`** (to be created):

> These two files are **required subprocess entry points** — they must remain as standalone runnable scripts at a known path.
>
> `kaggle_training_helper.py` (and the Kaggle notebooks) spawn the trainer as a child process:
>
> ```python
> subprocess.run([sys.executable, str(repo_dir / "bin/main_style_trainer.py"), "train", ...])
> ```
>
> **Why a subprocess and not a direct import?**
>
> PyTorch and CUDA/DirectML allocate GPU memory in the process that imports them.  Importing PyTorch inside the Jupyter notebook kernel or the Qt app process would:
> - pin several GB of VRAM for the lifetime of the process, even after training finishes;
> - make it impossible to release that memory without restarting the kernel;
> - risk CUDA context conflicts if another process (e.g. the Qt styler) is also using the GPU.
>
> Spawning `main_style_trainer.py` as a fully isolated child process means PyTorch loads in that process only, the GPU memory is freed the moment training completes, and stdout/stderr can be streamed line-by-line back to the notebook for live progress display.
>
> The logic itself lives in `src/trainer/app.py`; these stubs are thin forwarders that exist solely to give the subprocess launcher a stable file-system path.

---

## Duplicate-pattern savings summary

| Pattern | Sites | Lines saved |
|---------|-------|-------------|
| OOM detection in `engine.py` | 3 | ~8 |
| `project_root` resolution in `main_window.py` | 2 | ~8 |
| ~~`engine._sessions.pop()` in `batch_styler.py`~~ | ~~2~~ | ~~4~~ (✅ done) |
| Style name lookup (`_resolve_style_id_by_name` / `filter_styles_by_name`) | 2 | ~12 |
| **Total remaining** | | **~28 lines** |
