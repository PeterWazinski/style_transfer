# -*- mode: python ; coding: utf-8 -*-
# style_transfer.spec  –  PyInstaller spec for StyleTransfer
#
# Produces a single portable Windows exe: dist\StyleTransfer.exe
# To rebuild:  .\compile.ps1
#
# NOTE: torch / torchvision are intentionally excluded.  They are only
#       required for custom style *training* (a developer workflow).
#       Inference runs entirely through ONNX Runtime and needs no torch.

from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs

block_cipher = None

# ── Runtime files needed by onnxruntime (provider configs, native DLLs) ──
ort_datas    = collect_data_files("onnxruntime", include_py_files=False)
ort_binaries = collect_dynamic_libs("onnxruntime")

a = Analysis(
    ["src/app.py"],
    pathex=["."],
    binaries=ort_binaries,
    datas=[
        # ── Application data bundled into the exe ────────────────────────
        # These land at sys._MEIPASS/styles/ and sys._MEIPASS/sample_images/
        # respectively — matching the paths used by _project_root() in app.py.
        ("styles",        "styles"),
        ("sample_images", "sample_images"),
        # ── onnxruntime provider config / XML schemas ────────────────────
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

# ── Single-file executable ────────────────────────────────────────────────
# All binaries, data and zipped sources are packed into one exe.
# On first launch Windows extracts them to %TEMP%\onefile_<pid>\ then runs.
exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name="StyleTransfer",
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
)
