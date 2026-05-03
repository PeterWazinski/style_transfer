# bin/ — subprocess entry points

These two files are **required subprocess entry points** — they must remain as
standalone runnable scripts at a known path.

`kaggle_training_helper.py` (and the Kaggle notebooks) spawn the trainer as a
child process:

```python
subprocess.run([sys.executable, str(repo_dir / "bin/main_style_trainer.py"), "train", ...])
```

## Why a subprocess and not a direct import?

PyTorch and CUDA/DirectML allocate GPU memory in the process that imports them.
Importing PyTorch inside the Jupyter notebook kernel or the Qt app process would:

- pin several GB of VRAM for the lifetime of the process, even after training
  finishes;
- make it impossible to release that memory without restarting the kernel;
- risk CUDA context conflicts if another process (e.g. the Qt styler) is also
  using the GPU.

Spawning `main_style_trainer.py` as a fully isolated child process means
PyTorch loads in that process only, the GPU memory is freed the moment training
completes, and stdout/stderr can be streamed line-by-line back to the notebook
for live progress display.

## Where the logic lives

The logic itself lives in `src/trainer/app.py`; these stubs are thin forwarders
that exist solely to give the subprocess launcher a stable file-system path.

| Stub | Delegates to |
|------|-------------|
| `bin/main_style_trainer.py` | `src.trainer.app:main` |
| `bin/main_image_styler.py`  | `src.stylist.app:main` |
