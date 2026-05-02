# Refactoring Plan

**Date**: 2026-05-02  
**Status**: Analysis only — no code changed  
**Scope**: `src/`, `scripts/`, `tests/`, top-level files

---

## Summary table

| # | Severity | Category | File(s) | Approx. saving |
|---|----------|----------|---------|---------------|
| 1 | **critical** | God class | `src/stylist/main_window.py` | −350 lines |
| 2 | **serious** | Duplicate logic | `src/core/engine.py` | −8 lines / 3 sites |
| 3 | **serious** | Duplicate logic | `main_window.py` + `batch_styler.py` | −12 lines / 2 sites |
| 4 | **serious** | Breaks encapsulation | `batch_styler.py` | −4 lines / 2 sites |
| 5 | **serious** | Duplicate concept | `main_window.py` + `batch_styler.py` | −12 lines |
| 6 | **serious** | Missing serialisation | `replay_schema.py` / `main_window.py` | consolidates YAML logic |
| 7 | **serious** | Dead directories | `src/ml/`, `src/ui/`, `tests/unit/`, `tests/integration/` | removes 4 ghost trees |
| 8 | **minor** | Dual attr aliasing | `main_window.py` | removes 4 confusing aliases |
| 9 | **minor** | Magic number | `batch_styler.py` | 1 hardcoded value |
| 10 | **minor** | Stale import | `src/stylist/app.py` | 2 unused names |
| 11 | **minor** | Structure | `scripts/` | separation of concerns |
| 12 | **minor** | Structure | top-level launchers | entry-point consolidation |

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

## 2 (serious) — OOM detection duplicated 3 times in `engine.py`

The same 4-line exception guard appears verbatim in `_infer_tile()`, `_infer_tile_nhwc_tanh()`, and `_infer_tile_nchw_tanh()`:

```
_msg = str(exc).lower()
if any(k in _msg for k in ("out of memory", "insufficient", "oom", ": 6 :", "error code: 6")):
    raise OOMError(...)
```

**Proposed fix**: extract a private `_reraise_if_oom(exc, tile_size)` helper inside `StyleTransferEngine`.  All three inference methods call it in their `except` blocks.  
Saves ≈ 8 lines; removes the risk of the keyword list drifting between copies.

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

appears **twice** inside `MainWindow`: once in `_on_style_selected()` and once in `_load_and_apply_replay_log()`.  A very similar pattern is also duplicated in `batch_styler.py` (as the module-level `REPO_ROOT`).

**Proposed fix**: extract a module-level `_get_project_root() -> Path` helper (or one shared function in `src/core/utils.py`) called from all three sites.  
Saves ≈ 8 lines; one place to update if the bundle layout ever changes.

---

## 4 (serious) — Direct access to `engine._sessions` from `batch_styler.py`

Two commands in `batch_styler.py` reach into the private attribute to free a loaded model:

```python
engine._sessions.pop(catalog_style["id"], None)   # cmd_pdfoverview
engine._sessions.pop(catalog_style["id"], None)   # cmd_replay
```

`StyleTransferEngine.unload_model(style_id)` already exists for exactly this purpose and also clears `_model_meta` and calls `gc.collect()`.

**Proposed fix**: replace both calls with `engine.unload_model(style_id)`.  
No new code needed; 2-line change that removes the encapsulation violation.

---

## 5 (serious) — Style name lookup duplicated across the boundary

Two parallel implementations of "look up a style by display name":

| Location | Implementation |
|----------|---------------|
| `main_window.py._resolve_style_id_by_name(name)` | iterates `self._registry.list_styles()` |
| `batch_styler.py.filter_styles_by_name(styles, name)` | iterates a raw dict list loaded from JSON |

Both do the same case-insensitive match; they diverge only because `batch_styler.py` bypasses `StyleRegistry` and reads the catalog JSON directly into plain dicts.

**Proposed fix**:

1. Let `batch_styler.py` instantiate `StyleRegistry(REPO_ROOT / "styles" / "catalog.json")` and use it instead of the raw JSON dict.
2. Add a convenience method `StyleRegistry.find_by_name(name: str) -> StyleModel | None` (case-insensitive) that both callers can use.
3. Delete `filter_styles_by_name()` from `batch_styler.py`.

Also eliminates all the raw-dict boilerplate currently spread across the batch-styler commands.  
Saves ≈ 12 lines; unifies the lookup contract.

---

## 6 (serious) — `replay_schema.py` only deserialises; serialisation lives inline in `MainWindow`

`replay_schema.py` exposes `load_replay_log()` but has no matching `dump_replay_log()`.  The YAML serialisation is implemented inline in `MainWindow._format_replay_log()`.  This means:
- The data format is defined in the schema module but written in the UI layer.
- `batch_styler.py` has no way to produce a replay log without duplicating the format.

**Proposed fix**: add `dump_style_chain(log: ReplayLog) -> str` (a YAML serialisation function) to `style_chain_schema.py` (the already-planned rename of `replay_schema.py`).  `_format_replay_log()` in `MainWindow` becomes a one-liner that calls it.

---

## 7 (serious) — Ghost directory trees

Four directories exist in the repo but contain no source files (only `__init__.py` or `__pycache__`):

| Path | Content | Notes |
|------|---------|-------|
| `src/ml/` | only `__pycache__/` | appears to be an abandoned subpackage placeholder |
| `src/ui/widgets/` | only `__pycache__/` | real widgets are in `src/stylist/widgets/` |
| `tests/unit/` | only `__pycache__/` | no test files ever added |
| `tests/integration/` | only `__init__.py` | no integration tests exist |

**Proposed fix**: delete all four directory trees; update `.gitignore` if needed.

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

**Proposed fix**: keep only the public attribute (`self.registry`, `self.engine`, `self.photo_manager`); replace all private-alias usages inside the class with the public name.  Tests already refer to the public names anyway.  
Saves 3 lines; removes the risk of aliases diverging.

---

## 9 (minor) — Magic JPEG quality number in `batch_styler.py`

`cmd_replay()` hardcodes `quality=92`:

```python
result.save(out_path, format="JPEG", quality=92)
```

`PhotoManager.save()` defaults to `quality=95`.  The two values differ silently.

**Proposed fix**: define `JPEG_QUALITY: int = 92` (or use `PhotoManager.save()` directly so there is a single authoritative default) and reference the constant.

---

## 10 (minor) — Stale import in `src/stylist/app.py`

```python
from src.core.settings import AppSettings, DEFAULT_TILE_SIZE, DEFAULT_OVERLAP
```

`DEFAULT_TILE_SIZE` and `DEFAULT_OVERLAP` are imported but not used anywhere in `app.py`; they are only consumed by `src/stylist/settings_dialog.py`.

**Proposed fix**: remove the two unused names from the import.

---

## 11 (minor) — `scripts/` mixes user-facing CLI with developer/research tooling

Current contents:

| File | Audience | Kind |
|------|----------|------|
| `batch_styler.py` | end users / CI | compiled as `BatchStyler.exe` |
| `add_style_helper.py` | developers | Jupyter notebook backend |
| `kaggle_training_helper.py` | ML engineers | Kaggle-specific training CLI |
| `benchmark.py` | developers | performance tool |
| `*.ipynb` (5 files) | researchers | exploration notebooks |
| `index.md` | documentation | notebook index |

**Proposed restructure**:

```
scripts/
  batch_styler.py          ← unchanged (user-facing)
  dev/
    add_style_helper.py
    benchmark.py
  training/
    kaggle_training_helper.py
  notebooks/
    *.ipynb
    index.md
```

No renames needed; only moves.  Update `compile.ps1` path references if any point to `scripts/` directly.

---

## 12 (minor) — Top-level launcher files

`main_image_styler.py` and `main_style_trainer.py` are both thin forwarding wrappers (16 and ~80 lines respectively).  Having them at the repo root is convenient for PyInstaller (`style_transfer.spec`) but adds visual clutter.

Two options (choose one):

**Option A — move to `bin/`**: create `bin/main_image_styler.py` and `bin/main_style_trainer.py`; update the two `Analysis()` `script` paths in `style_transfer.spec`.

**Option B — use `pyproject.toml` console scripts**: add
```toml
[project.scripts]
stylist = "src.stylist.app:main"
trainer = "main_style_trainer:main"
```
so the launchers are installed as CLI commands; `style_transfer.spec` still references the raw files during the build.

Neither option is urgent; Option B gives the cleanest developer UX (`stylist` from any directory after `pip install -e .`).

---

## Duplicate-pattern savings summary

| Pattern | Sites | Lines saved |
|---------|-------|-------------|
| OOM detection in `engine.py` | 3 | ~8 |
| `project_root` resolution in `main_window.py` | 2 | ~8 |
| `engine._sessions.pop()` in `batch_styler.py` | 2 | ~4 |
| Style name lookup (`_resolve_style_id_by_name` / `filter_styles_by_name`) | 2 | ~12 |
| **Total** | | **~32 lines** |
