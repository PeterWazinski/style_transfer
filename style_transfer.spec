# -*- mode: python ; coding: utf-8 -*-
# style_transfer.spec  –  PyInstaller spec for PetersPictureStyler
#
# Produces a portable app directory: dist\PetersPictureStyler\
#   PetersPictureStylist.exe  ← the GUI executable
#   BatchStyler.exe           ← headless CLI for batch processing
#   styles\                   ← editable; add new styles here without recompiling
#   app.log                   ← written at runtime
#   (+ all onnxruntime / Qt DLLs)
#
# To rebuild:  .\compile.ps1
#
# NOTE: torch / torchvision are intentionally excluded.  They are only
#       required for custom style *training* (a developer workflow).
#       Inference runs entirely through ONNX Runtime and needs no torch.
#
# IMPORTANT: styles\ is NOT bundled into the exe.  compile.ps1 copies it
#       into the output directory after the build so users can add new
#       styles by dropping a folder + updating catalog.json — no recompile.

from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs

block_cipher = None

# ── Runtime files needed by onnxruntime (provider configs, native DLLs) ──
ort_datas    = collect_data_files("onnxruntime", include_py_files=False)
ort_binaries = collect_dynamic_libs("onnxruntime")

a = Analysis(
    ["src/stylist/app.py"],
    pathex=["."],
    binaries=ort_binaries,
    datas=[
        # ── onnxruntime provider config / XML schemas ────────────────────
        # styles\ and sample_images\ are NOT bundled — compile.ps1 copies
        # styles\ into the output directory after the build.
        *ort_datas,
    ],
    hiddenimports=[
        # onnxruntime loads its pybind extension lazily; declare it explicitly
        "onnxruntime",
        "onnxruntime.capi",
        "onnxruntime.capi._pybind_state",
        # Pillow can probe for tkinter during wheel discovery
        "PIL._tkinter_finder",
        # cv2 headless binding
        "cv2",
        # replay log YAML support
        "yaml",
        # schema validation
        "pydantic",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    # ── Exclude heavy training-only / dev dependencies ───────────────────
    excludes=[
        "torch",
        "torchvision",
        "torchaudio",
        "tensorboard",
        "matplotlib",
        "scipy",
        "pandas",
        "IPython",
        "ipykernel",
        "jupyter",
        "pytest",
        "mypy",
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

# ── Analysis for BatchStyler (headless CLI, no Qt) ───────────────────────
b = Analysis(
    ["scripts/batch_styler.py"],
    pathex=["."],
    binaries=ort_binaries,
    datas=[*ort_datas],
    hiddenimports=[
        "onnxruntime",
        "onnxruntime.capi",
        "onnxruntime.capi._pybind_state",
        "PIL._tkinter_finder",
        # replay log YAML support
        "yaml",
        # schema validation
        "pydantic",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "torch", "torchvision", "torchaudio", "tensorboard",
        "PySide6", "PySide2", "PyQt5", "PyQt6",
        "matplotlib", "scipy", "pandas",
        "IPython", "ipykernel", "jupyter",
        "pytest", "mypy",
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz_b = PYZ(b.pure, b.zipped_data, cipher=block_cipher)

# ── One-directory bundle ─────────────────────────────────────────────────
# EXE contains only the bootloader + compressed Python code.
# COLLECT places the exe, all DLLs, and onnxruntime data into one folder.
# In onedir mode sys._MEIPASS == the folder containing the exe, so
# _project_root() in app.py resolves styles\ correctly without any changes.
exe = EXE(
    pyz,
    a.scripts,
    [],                             # ← onedir: no data packing into the exe
    name="PetersPictureStylist",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    # UPX compresses the exe; skip Qt / runtime DLLs (UPX can corrupt them)
    upx=True,
    upx_exclude=[
        "vcruntime140.dll",
        "python3*.dll",
        "Qt6Core.dll",
        "Qt6Gui.dll",
        "Qt6Widgets.dll",
        "Qt6Network.dll",
        "Qt6OpenGL.dll",
        "DirectML.dll",
    ],
    runtime_tmpdir=None,
    console=False,                  # no black console window for GUI app
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='assets/palette.ico',
)

exe_b = EXE(
    pyz_b,
    b.scripts,
    [],                             # ← onedir: no data packing into the exe
    name="BatchStyler",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[
        "vcruntime140.dll",
        "python3*.dll",
        "DirectML.dll",
    ],
    runtime_tmpdir=None,
    console=True,                   # CLI tool — output shown in console window
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='assets/palette.ico',
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    exe_b,
    b.binaries,
    b.zipfiles,
    b.datas,
    strip=False,
    upx=True,
    upx_exclude=[
        "vcruntime140.dll",
        "python3*.dll",
        "Qt6Core.dll",
        "Qt6Gui.dll",
        "Qt6Widgets.dll",
        "Qt6Network.dll",
        "Qt6OpenGL.dll",
        "DirectML.dll",
    ],
    name="PetersPictureStyler",     # → dist\PetersPictureStyler\
)
