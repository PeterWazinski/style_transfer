# Replay Log — Implementation Plan

## 1. Format recommendation: YAML (`.yml`)

You already used `.yml` in the example command, so that is the natural choice.

**Example file `my_style_chain.yml`:**

```yaml
# PetersPictureStyler – style chain
# Created: 2026-04-30 14:32
version: 1
steps:
  - style: Anime Hayao
    strength: 150
  - style: Van Gogh
    strength: 75
  - style: Monet
    strength: 100
```

### Why YAML over alternatives

| Format | Human-editable | No dep | Comments | Verdict |
|---|---|---|---|---|
| YAML `.yml` | ✅ excellent | ❌ needs `pyyaml` | ✅ yes | **recommended** |
| JSON `.json` | ⚠ verbose | ✅ stdlib | ❌ no | fallback |
| Plain text `Name @ 75%` | ✅ excellent | ✅ stdlib | ✅ yes | simplest but non-standard |

**Strength is stored as integer percentage** (1–300), matching the slider display in the app.  
**`style` is the display name** (e.g. `"Anime Hayao"`), not the internal `id`, so users can read and edit the file without knowing internal ids.

### New dependency

Add `pyyaml>=6.0` to `requirements.txt`.  
Add `pyyaml` to the `hiddenimports` / `--collect-all` list in `compile.ps1` so PyInstaller bundles it.

---

## 2. Feature decisions

✅ **Auto-save `.yml` alongside saved image** — when the user saves `my_styled_foto.jpg`, automatically write `my_styled_foto.yml` with the replay log next to it. Controlled by a new setting **"Autosave replay log"** (default: on). See §3.11 and §6.

✅ **File → Load Replay Log…** — the user picks an existing `.yml`, the app re-runs the chain on the currently open photo. Schema validation is applied before execution. See §3.12.

✅ **`--strength-override N`** — scales every step's strength proportionally without editing the file. E.g. `--strength-override 60` turns steps of 150%/75% into 90%/45%. See §4.5.

✅ **YAML schema validation** — any hand-edited `.yml` is validated against a schema before use (both in app and in BatchStyler). Clear error messages for missing/wrong fields. See §2.1.

✅ **Remove `--fullimage` from BatchStyler** — no longer used; keep only `--pdfoverview` and `--replay`. See §4.

🔵 **Version field** — the `version: 1` field in the YAML allows the format to evolve without breaking old files.

### 2.1 YAML schema

The schema is defined as Pydantic models and validated at load time:

```python
# src/core/replay_schema.py  (new file)
from pydantic import BaseModel, Field, field_validator

class ReplayStep(BaseModel):
    style: str = Field(..., min_length=1)
    strength: int = Field(..., ge=1, le=300)

class ReplayLog(BaseModel):
    version: int = Field(1)
    steps: list[ReplayStep] = Field(..., min_length=1)

    @field_validator("version")
    @classmethod
    def check_version(cls, v: int) -> int:
        if v != 1:
            raise ValueError(f"Unsupported replay log version: {v}")
        return v


def load_replay_log(path: Path) -> ReplayLog:
    """Load and validate a replay YAML file. Raises ValueError with a clear message on error."""
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(f"YAML syntax error in '{path.name}': {exc}") from exc
    try:
        return ReplayLog.model_validate(data)
    except ValidationError as exc:
        raise ValueError(f"Invalid replay log '{path.name}':\n{exc}") from exc
```

Used by both the app (`_load_and_apply_replay_log`) and `cmd_replay` in BatchStyler.

---

## 3. App changes — `src/stylist/main_window.py`

### 3.1 New instance fields (in `__init__`)

```python
self._current_style_name: str = ""           # set when gallery selection changes
self._replay_log: list[dict[str, object]] = []  # {"style": str, "strength": int}
```

### 3.2 Track current style name

In `_on_style_selected`:
```python
self._current_style_name = style.name
```
(`style.name` is the human display name; `style.id` is already used for engine calls.)

### 3.3 Log push — `_apply_style`

After the successful `_push_undo_snapshot()` call, **before** updating `_styled_photo`:
```python
# Apply always starts a new chain from the original photo
self._replay_log = [{"style": self._current_style_name, "strength": int(strength * 100)}]
```

### 3.4 Log append — `_reapply_style`

After the successful `_push_undo_snapshot()` call:
```python
self._replay_log.append({"style": self._current_style_name, "strength": int(strength * 100)})
```

### 3.5 Log update — `_reapply_style_strength`

This method does not push an undo snapshot but does commit a new strength value.  
After a successful `result` (not `None`), update the last entry:
```python
if self._replay_log:
    self._replay_log[-1]["strength"] = int(strength * 100)
```

### 3.6 Log pop — `_perform_undo`

After popping the snapshot, also remove the last replay log entry:
```python
if self._replay_log:
    self._replay_log.pop()
```

### 3.7 Log clear — `_open_photo`, `_reset_photo`

Call `self._replay_log = []` in both methods (alongside `self._clear_undo_stack()`).

### 3.8 YAML serialiser helper

```python
def _format_replay_log(self) -> str:
    import yaml  # lazy import — only needed when user requests clipboard copy
    from datetime import datetime
    header = (
        f"# PetersPictureStyler – style chain\n"
        f"# Created: {datetime.now():%Y-%m-%d %H:%M}\n"
    )
    data = {"version": 1, "steps": self._replay_log}
    return header + yaml.dump(data, allow_unicode=True, sort_keys=False, default_flow_style=False)
```

### 3.9 Clipboard action handler

```python
def _copy_replay_log_to_clipboard(self) -> None:
    if not self._replay_log:
        QMessageBox.information(self, "Replay Log", "No styles applied yet — nothing to copy.")
        return
    QApplication.clipboard().setText(self._format_replay_log())
    self._status.showMessage("Replay log copied to clipboard.")
```

### 3.10 File menu additions

In `_create_menu`, add two new items after the Save action and before its separator:

```python
replay_copy_action = QAction("Replay Log to Clipboard", self)
replay_copy_action.setStatusTip("Copy the current style chain as YAML to the clipboard")
replay_copy_action.triggered.connect(self._copy_replay_log_to_clipboard)
file_menu.addAction(replay_copy_action)

replay_load_action = QAction("Load Replay Log…", self)
replay_load_action.setStatusTip("Load a .yml style chain and apply it to the current photo")
replay_load_action.triggered.connect(self._load_and_apply_replay_log)
file_menu.addAction(replay_load_action)
```

### 3.11 Auto-save `.yml` alongside saved image

In `_save_result`, after `self._photo_manager.save(...)` succeeds:

```python
if self._settings.autosave_replay_log and self._replay_log:
    yml_path = path.with_suffix(".yml")
    try:
        yml_path.write_text(self._format_replay_log(), encoding="utf-8")
        self._status.showMessage(f"Saved to: {path.name}  (+ replay log)")
    except OSError as exc:
        logger.warning("Could not auto-save replay log: %s", exc)
```

### 3.12 Load Replay Log handler

```python
def _load_and_apply_replay_log(self) -> None:
    if self._current_photo is None:
        QMessageBox.information(self, "Load Replay Log", "Open a photo first.")
        return
    path_str, _ = QFileDialog.getOpenFileName(
        self, "Load Replay Log", "", "YAML style chain (*.yml *.yaml)"
    )
    if not path_str:
        return
    try:
        replay = load_replay_log(Path(path_str))   # validates schema
    except ValueError as exc:
        QMessageBox.critical(self, "Invalid Replay Log", str(exc))
        return
    # Reset to original photo, then apply each step in sequence
    self._clear_undo_stack()
    self._replay_log = []
    for i, step in enumerate(replay.steps):
        style_id = self._resolve_style_id_by_name(step.style)
        if style_id is None:
            QMessageBox.warning(self, "Load Replay Log",
                f"Style '{step.style}' not found in catalog — chain aborted at step {i+1}.")
            return
        if i == 0:
            self._apply_style(style_id, step.strength / 100.0)
        else:
            self._reapply_style(style_id, step.strength / 100.0)
    self._status.showMessage(f"Replay log applied: {Path(path_str).name}")
```

`_resolve_style_id_by_name` searches the loaded catalog by display name (case-insensitive).

---

## 4. BatchStyler changes — `scripts/batch_styler.py`

**`--fullimage` mode is removed.** The mode group will contain only `--pdfoverview` and `--replay`.

### 4.1 Argparse changes

- Remove `--fullimage` from `mode_group` and delete `cmd_fullimage()` and `_apply_all_styles()`.
- Add `--replay` to `mode_group`:

```python
mode_group.add_argument(
    "--replay", type=Path, metavar="CHAIN",
    help="Apply a saved style chain YAML to the image.",
)
```

- Add `--strength-override` as an optional standalone argument:

```python
parser.add_argument(
    "--strength-override", type=int, default=None, metavar="PCT",
    help="Scale all replay step strengths by this percentage (e.g. 60 → ×0.60). Only used with --replay.",
)
```

CLI usage:
```
BatchStyler --replay .\my_style_chain.yml .\my_foto.jpg
BatchStyler --replay .\my_style_chain.yml .\my_foto.jpg --strength-override 60
```

Output: `my_foto_my_style_chain.jpg` next to `my_foto.jpg`.

### 4.2 New `cmd_replay` function

```python
def cmd_replay(
    image_path: Path,
    replay_path: Path,
    styles: list[dict],
    tile_size: int,
    overlap: int,
    use_float16: bool,
    strength_override: int | None = None,   # scale factor in %, e.g. 60 → ×0.60
) -> None:
    from src.core.replay_schema import load_replay_log
    try:
        replay = load_replay_log(replay_path)   # validates schema
    except ValueError as exc:
        sys.exit(f"Error: {exc}")

    engine = StyleTransferEngine()
    result = Image.open(image_path).convert("RGB")
    scale = (strength_override / 100.0) if strength_override is not None else 1.0

    for i, step in enumerate(replay.steps, start=1):
        strength: float = (step.strength * scale) / 100.0   # % → 0-3.0
        matched = filter_styles_by_name(styles, step.style)  # exits on unknown name
        catalog_style = matched[0]
        model_path: Path = REPO_ROOT / catalog_style["model_path"]
        if not model_path.exists():
            sys.exit(f"Step {i}: model not found for '{step.style}': {model_path}")
        tensor_layout: str = catalog_style.get("tensor_layout", "nchw")
        effective_pct = int(step.strength * scale)
        print(f"Step {i}/{len(replay.steps)}: '{step.style}' @ {effective_pct}% ...", flush=True)
        engine.load_model(catalog_style["id"], model_path, tensor_layout=tensor_layout)
        result = engine.apply(
            result, catalog_style["id"],
            strength=strength, tile_size=tile_size, overlap=overlap,
            use_float16=use_float16,
        )
        engine._sessions.pop(catalog_style["id"], None)   # free VRAM after each step

    out_path = image_path.parent / f"{image_path.stem}_{replay_path.stem}.jpg"
    result.save(out_path, format="JPEG", quality=92)
    print(f"\nOK  Result written: {out_path}")
```

### 4.3 Dispatch in `main()`

```python
elif args.replay:
    cmd_replay(
        image_path, args.replay.resolve(), styles,
        tile_size=args.tile_size,
        overlap=args.overlap,
        use_float16=args.float16,
        strength_override=args.strength_override,
    )
```

### 4.4 Update `_USAGE` string

Replace old `--fullimage` lines; add `--replay` and `--strength-override` lines and examples.

### 4.5 `--strength-override` behaviour

- Only valid with `--replay`; print warning and exit if used with `--pdfoverview`.
- Scale is applied as: `effective_strength = step.strength_pct × (override / 100) / 100.0`.
- Result is clamped to the engine's accepted range (0.01–3.0).

---

## 5. Tests

### `tests/core/test_replay_schema.py` (new file)

| Test class | Test | Assertion |
|---|---|---|
| `TestReplaySchema` | `test_valid_log_parses` | valid YAML round-trips correctly |
| | `test_missing_steps_raises` | `ValidationError` when `steps` absent |
| | `test_empty_steps_raises` | `ValidationError` when `steps: []` |
| | `test_strength_out_of_range_raises` | `ValidationError` when strength > 300 or < 1 |
| | `test_unknown_version_raises` | `ValidationError` when `version: 99` |
| | `test_yaml_syntax_error_raises_valueerror` | malformed YAML → `ValueError` with clear message |

### `tests/stylist/test_main_window_replay.py` (new file)

| Test class | Test | Assertion |
|---|---|---|
| `TestReplayLog` | `test_log_empty_initially` | `_replay_log == []` |
| | `test_apply_starts_new_chain` | after Apply, log has 1 entry with correct style name and strength |
| | `test_reapply_appends_entry` | after Re-Apply, log has 2 entries |
| | `test_strength_adjust_updates_last_entry` | after `_reapply_style_strength`, last entry strength updated |
| | `test_undo_pops_entry` | after Undo, log shrinks by 1 |
| | `test_open_photo_clears_log` | after open, log == [] |
| | `test_reset_clears_log` | after reset, log == [] |
| | `test_format_replay_log_yaml` | `_format_replay_log()` parses back to correct dict |
| | `test_copy_to_clipboard_when_empty_shows_dialog` | messagebox shown, clipboard unchanged |
| | `test_autosave_yml_written_on_save` | when `autosave_replay_log=True`, `.yml` file created next to saved image |
| | `test_autosave_yml_skipped_when_disabled` | when `autosave_replay_log=False`, no `.yml` written |
| | `test_load_replay_log_applies_chain` | valid `.yml` → each step applied in order |
| | `test_load_replay_log_invalid_schema_shows_error` | bad `.yml` → `QMessageBox.critical` called |

### `tests/scripts/test_batch_styler.py` (extend existing)

| Test class | Test | Assertion |
|---|---|---|
| `TestCmdReplay` | `test_replay_applies_steps_in_order` | engine.apply called twice in correct order |
| | `test_replay_output_filename` | stem is `{image_stem}_{chain_stem}.jpg` |
| | `test_replay_unknown_style_exits` | `SystemExit` raised |
| | `test_replay_strength_converted_to_float` | engine called with `strength=0.75` when yaml has `75` |
| | `test_replay_strength_override_scales_all_steps` | `--strength-override 50` halves all step strengths |
| | `test_replay_invalid_schema_exits` | invalid YAML schema → `SystemExit` with clear message |

---

## 6. New setting — `autosave_replay_log`

In `src/core/settings.py`:
```python
autosave_replay_log: bool = True
```

In `src/stylist/settings_dialog.py`, add a checkbox in the existing settings panel:
- Label: **"Autosave replay log (.yml) when saving image"**
- Default: checked
- Bound to `AppSettings.autosave_replay_log`

---

## 7. Files changed

| File | Change |
|---|---|
| `requirements.txt` | add `pyyaml>=6.0` |
| `compile.ps1` | bundle pyyaml with PyInstaller |
| `src/core/replay_schema.py` | **new** — `ReplayStep`, `ReplayLog` Pydantic models + `load_replay_log()` |
| `src/core/settings.py` | add `autosave_replay_log: bool = True` |
| `src/stylist/settings_dialog.py` | add autosave checkbox |
| `src/stylist/main_window.py` | replay log fields + all wiring + 2 File menu items + auto-save in `_save_result` |
| `scripts/batch_styler.py` | remove `--fullimage`; add `--replay` + `--strength-override` + `cmd_replay()` |
| `tests/core/test_replay_schema.py` | **new** — 6 tests |
| `tests/stylist/test_main_window_replay.py` | **new** — 13 tests |
| `tests/scripts/test_batch_styler.py` | extend with 6 replay tests |

---

## 8. Implementation phases

### Phase 1 — Schema foundation  *(smallest releasable unit)*

1. Add `pyyaml>=6.0` to `requirements.txt`, install it
2. Add `pyyaml` to `compile.ps1` PyInstaller args
3. `src/core/replay_schema.py` — `ReplayStep`, `ReplayLog`, `load_replay_log()`
4. **Tests**: `tests/core/test_replay_schema.py` — 6 tests
5. `pytest -k test_replay_schema` → all green
6. **Commit**: `feat: add replay log YAML schema and loader`

---

### Phase 2 — App integration

1. `src/core/settings.py` — add `autosave_replay_log: bool = True`
2. `src/stylist/settings_dialog.py` — autosave checkbox
3. `src/stylist/main_window.py`:
   - New fields: `_current_style_name`, `_replay_log`
   - Track style name in `_on_style_selected`
   - Log push/append/update/pop/clear in `_apply_style`, `_reapply_style`, `_reapply_style_strength`, `_perform_undo`, `_open_photo`, `_reset_photo`
   - `_format_replay_log()`, `_copy_replay_log_to_clipboard()`, `_load_and_apply_replay_log()`, `_resolve_style_id_by_name()`
   - Two new File menu items: "Replay Log to Clipboard", "Load Replay Log…"
   - Auto-save `.yml` in `_save_result`
4. **Tests**: `tests/stylist/test_main_window_replay.py` — 13 tests
5. `pytest -k test_main_window_replay` → all green; full suite → all green
6. **Commit**: `feat: replay log — in-app chain recording, clipboard copy, load & auto-save`

---

### Phase 3 — BatchStyler CLI

1. `scripts/batch_styler.py`:
   - Remove `--fullimage`, `cmd_fullimage()`, `_apply_all_styles()`
   - Add `--replay` + `--strength-override` to `argparse`
   - Add `cmd_replay()`
   - Update `_USAGE` string
2. **Tests**: extend `tests/scripts/test_batch_styler.py` — 6 new replay tests; remove/update any `--fullimage` tests
3. Full suite → all green
4. **Commit**: `feat: BatchStyler --replay with schema validation and --strength-override`
5. Recompile to update `BatchStyler.exe`
