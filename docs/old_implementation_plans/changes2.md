# Change: "Append Style Chain" — replaces "Apply Style Chain"

## Summary

The existing "Apply Style Chain" menu item resets the photo to its original
state, then applies all chain steps from scratch.  
The replacement "Append Style Chain" instead applies the chain steps **on top
of whatever the photo currently looks like**, preserving all prior manual
transformations and the undo/replay log.

---

## Decided constraints

| # | Decision |
|---|----------|
| 1 | Single menu entry — "Append Style Chain…" replaces "Apply Style Chain…" |
| 2 | tile_size / tile_overlap values in the YAML are **ignored** — user's current session settings are used |
| 3 | Undo granularity unchanged — each appended step gets its own snapshot (deque maxlen=3) |

---

## Behaviour before → after

### Before ("Apply Style Chain")
1. Clears `_replay_log`, undo stack, `_styled_photo`
2. Resets canvas to original photo
3. Applies chain tile/overlap settings to `_settings`
4. Step 1 via `_apply_style` (sets `_replay_log = [t3]`)
5. Step 2+ via `_reapply_style` (appends to `_replay_log`)

### After ("Append Style Chain")
1. Nothing is reset — `_styled_photo`, `_replay_log`, undo stack all preserved
2. tile/overlap settings from YAML are **not applied**
3. Every chain step uses `_reapply_style` if a `_styled_photo` already exists,
   else `_apply_style` for the very first step then `_reapply_style` for the rest

#### Example walkthrough
| Moment | `_replay_log` | Right pane |
|--------|--------------|-----------|
| after manual t1, t2 | [t1, t2] | t2 result |
| after appending chain [t3, t4] | [t1, t2, t3, t4] | t4 result |
| after Undo | [t1, t2, t3] | t3 result |
| saved → YAML | steps: t1, t2, t3, t4 | — |

---

## Files changed

### 1. `src/stylist/style_chain_controller.py`

**Rename method** `_apply_style_chain` → `_append_style_chain`.

**Remove the reset block** (lines that currently do):
```python
self._styled_photo = None
self._styled_photo_input = None
self._clear_undo_stack()
self._replay_log = []
self.canvas.reset_styled()
self._save_action.setEnabled(False)
self.canvas.set_original(self._pil_to_pixmap(self._current_photo))
```

**Remove tile/overlap override** (lines that currently do):
```python
if replay.tile_size is not None:
    self._settings.tile_size = replay.tile_size
if replay.tile_overlap is not None:
    self._settings.overlap = replay.tile_overlap
```

**Change the loop** — replace the `if i == 0 / else` pattern with a
single check on whether a styled result already exists:
```python
for step in replay.steps:
    style_id = self._resolve_style_id_by_name(step.style)
    assert style_id is not None
    self._current_style_name = step.style
    # ... model loading unchanged ...
    if self._styled_photo is None:
        self._apply_style(style_id, step.strength / 100.0)
    else:
        self._reapply_style(style_id, step.strength / 100.0)
```

**Update status message**:
```python
self._status.showMessage(f"Style chain appended: {Path(path_str).name}")
```

---

### 2. `src/stylist/main_window.py`

**`_build_menus`** — three one-line changes in the action definition:
```python
# before
self._chain_apply_action = QAction("Apply Style Chain\u2026", self)
self._chain_apply_action.setStatusTip("Load a .yml style chain and apply it to the current photo")
self._chain_apply_action.triggered.connect(self._apply_style_chain)

# after
self._chain_append_action = QAction("Append Style Chain\u2026", self)
self._chain_append_action.setStatusTip("Load a .yml style chain and append it on top of the current photo state")
self._chain_append_action.triggered.connect(self._append_style_chain)
```

Also update `file_menu.addAction(self._chain_apply_action)` →
`file_menu.addAction(self._chain_append_action)`.

No enable/disable changes needed — the action was and remains always enabled;
the guard inside the method handles the "no photo open" case.

---

### 3. `tests/stylist/test_main_window_replay.py`

#### Tests to update (rename + adjust expectations)

| Old test | Change |
|----------|--------|
| `test_apply_style_chain_applies_chain` | Rename to `test_append_style_chain_appends_to_existing_log`; pre-load one manual style step first; assert `replay_log` length is **2** after the 1-step chain |
| `test_load_replay_applies_tile_settings` | **Delete** — this behaviour is removed |
| `test_apply_chain_preflight_unknown_style_shows_error` | Rename to `test_append_chain_preflight_unknown_style_shows_error`; call `_append_style_chain` |
| `test_load_replay_log_invalid_schema_shows_error` | Rename to `test_append_chain_invalid_schema_shows_error`; call `_append_style_chain` |

#### New tests to add

| Test | Asserts |
|------|---------|
| `test_append_chain_no_prior_style_starts_new_log` | With photo open but no style applied yet, appending a 2-step chain → `replay_log` length 2 |
| `test_append_chain_preserves_prior_replay_log` | Apply t1 manually, append [t2, t3] → `replay_log == [t1, t2, t3]` |
| `test_undo_after_append_removes_only_last_chain_step` | Apply t1, append [t2] → undo → `replay_log == [t1]` |

---

## Out of scope

- No change to `_apply_style`, `_reapply_style`, undo stack size, or any other method
- No change to the YAML schema or `dump_style_chain` / `load_style_chain`
- No change to how tile/overlap are applied in the existing Batch Styler CLI

---

# Change: Rename `replay` → `style_log` / `StyleChain`

## Summary

The "replay" concept was an early implementation name that leaked into
public symbols.  The module is already called `style_chain_schema.py` and
its functions already say `load_style_chain` / `dump_style_chain`, so the
remaining `Replay*` names are inconsistent.  This change brings all symbols
into alignment with the user-facing "style chain" / "style log" vocabulary.

---

## Symbol mapping

| Old name | New name | Kind |
|----------|----------|------|
| `ReplayLog` | `StyleChain` | Pydantic model class |
| `ReplayStep` | `ChainStep` | Pydantic model class |
| `_replay_log` | `_style_log` | MainWindow instance attribute |
| `autosave_replay_log` | `autosave_style_log` | `AppSettings` dataclass field |
| local variable `replay` | `chain` | local in controllers / batch styler |
| `TestReplaySchema` | `TestStyleChainSchema` | test class |
| `test_replay_*` (8 methods) | `test_chain_*` | test methods in `test_batch_styler.py` |
| UI label "Autosave replay log…" | "Autosave style log…" | `settings_dialog.py` checkbox label |

---

## Files changed

### 1. `src/core/style_chain_schema.py`
- Rename class `ReplayStep` → `ChainStep`
- Rename class `ReplayLog` → `StyleChain`
- Update all internal references and error messages

### 2. `src/core/settings.py`
- Rename field `autosave_replay_log` → `autosave_style_log`
- **Silent migration** in `from_dict`: before the existing key-filter step,
  copy the old key's value if present and the new key is absent:
  ```python
  if "autosave_replay_log" in data and "autosave_style_log" not in data:
      data = {**data, "autosave_style_log": data["autosave_replay_log"]}
  ```
  The old key is then dropped by the existing unknown-key filter.

### 3. `src/stylist/apply_controller.py`
- Rename all `_replay_log` references → `_style_log`

### 4. `src/stylist/main_window.py`
- Rename `_replay_log` → `_style_log` (init, `_open_photo`, `_reset_photo`,
  `_perform_undo`, `_apply_style_chain` / `_append_style_chain`, `_save_result`)
- Rename `autosave_replay_log` → `autosave_style_log`

### 5. `src/stylist/style_chain_controller.py`
- Rename import `ReplayLog` → `StyleChain`, `ReplayStep` → `ChainStep`
- Rename local variable `replay` → `chain`
- Rename `_replay_log` → `_style_log`

### 6. `src/stylist/settings_dialog.py`
- Update checkbox label text: `"Autosave replay log (.yml) when saving image"`
  → `"Autosave style log (.yml) when saving image"`
- Rename widget `autosave_replay_check` → `autosave_style_check` (or keep
  widget name as-is since it is a private detail — preference: rename for
  consistency)

### 7. `src/batch_styler/commands.py`
- Rename import `ReplayLog` → `StyleChain`
- Rename local variable `replay` → `chain`

### 8. `tests/core/test_style_chain_schema.py`
- Rename import `ReplayLog` → `StyleChain`, `ReplayStep` → `ChainStep`
- Rename test class `TestReplaySchema` → `TestStyleChainSchema`

### 9. `tests/stylist/test_main_window_replay.py`
- Rename `_replay_log` references → `_style_log` throughout

### 10. `tests/scripts/test_batch_styler.py`
- Rename 8 test methods: `test_replay_*` → `test_chain_*`
- Update any local variables named `replay` → `chain`

---

## Order of implementation

1. `style_chain_schema.py` (defines the renamed classes — do first so imports cascade)
2. `settings.py` (field rename + migration)
3. `apply_controller.py`, `style_chain_controller.py`, `main_window.py`,
   `settings_dialog.py`, `commands.py` (all import from the above)
4. All test files last

Run the full test suite after step 3 and again after step 4.

---

## Out of scope

- No change to YAML file format or schema version — `.yml` files produced by
  the old code remain valid input
- No change to `load_style_chain` / `dump_style_chain` function names
