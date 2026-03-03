"""Download pre-trained weights, export to ONNX, and generate preview thumbnails.

Downloads .pth files from yakhyo/fast-neural-style-transfer (GitHub release v1.0),
exports each to an ONNX model, then runs a small content image through each model
to produce a preview.jpg thumbnail shown in the gallery.

Usage::

    python scripts/setup_models.py            # download + export + preview all
    python scripts/setup_models.py --dry-run  # print what would happen
"""
from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
STYLES_ROOT: Path = PROJECT_ROOT / "styles"
TMP_DIR: Path = PROJECT_ROOT / ".model_cache"

# Sample content image used to render the style previews
# https://picsum.photos uses the same seed → same image every time (CC0)
PREVIEW_CONTENT_URL: str = "https://picsum.photos/seed/style42/512/512"
PREVIEW_CONTENT_PATH: Path = TMP_DIR / "preview_content.jpg"
PREVIEW_SIZE: int = 256  # size of generated preview thumbnail

# Ensure project root is on sys.path so `src` is importable
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Model catalogue (id → release asset name)
# ---------------------------------------------------------------------------
YAKHYO_BASE: str = (
    "https://github.com/yakhyo/fast-neural-style-transfer/releases/download/v1.0"
)

MODELS: list[dict[str, str]] = [
    {"id": "candy",         "pth": "candy.pth"},
    {"id": "mosaic",        "pth": "mosaic.pth"},
    {"id": "rain_princess", "pth": "rain-princess.pth"},
    {"id": "abstract",      "pth": "udnie.pth"},   # GitHub release artifact: udnie.pth
]


def _download(url: str, dest: Path, *, dry_run: bool) -> None:
    if dest.exists():
        print(f"  [skip]     {dest.name} already cached")
        return
    print(f"  [download] {url}")
    if dry_run:
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, dest)
    print(f"  [ok]       saved to {dest}")


def _export(pth_path: Path, onnx_path: Path, *, dry_run: bool) -> None:
    if onnx_path.exists():
        print(f"  [skip]     {onnx_path} already exists")
        return
    print(f"  [export]   {pth_path.name} → {onnx_path}")
    if dry_run:
        return

    import torch  # noqa: PLC0415
    import torch.nn as nn  # noqa: PLC0415

    # ---------------------------------------------------------------------------
    # Model definition matching yakhyo/fast-neural-style-transfer checkpoints.
    # Must match exactly so state_dict keys align.
    # ---------------------------------------------------------------------------
    class ConvLayer(nn.Module):
        def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                     stride: int, upsample: int | None = None) -> None:
            super().__init__()
            self.upsample = upsample
            self.reflection_pad = nn.ReflectionPad2d(kernel_size // 2)
            self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            if self.upsample is not None:
                x = nn.functional.interpolate(x, mode="nearest", scale_factor=self.upsample)
            return self.conv2d(self.reflection_pad(x))

    class ResidualBlock(nn.Module):
        def __init__(self, planes: int) -> None:
            super().__init__()
            self.conv1 = ConvLayer(planes, planes, 3, 1)
            self.in1 = nn.InstanceNorm2d(planes, affine=True)
            self.conv2 = ConvLayer(planes, planes, 3, 1)
            self.in2 = nn.InstanceNorm2d(planes, affine=True)
            self.relu = nn.ReLU()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            residual = x
            out = self.relu(self.in1(self.conv1(x)))
            out = self.in2(self.conv2(out))
            return out + residual

    class TransformerNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            f = [3, 32, 64, 128]
            self.conv1 = ConvLayer(f[0], f[1], 9, 1);  self.in1 = nn.InstanceNorm2d(f[1], affine=True)
            self.conv2 = ConvLayer(f[1], f[2], 3, 2);  self.in2 = nn.InstanceNorm2d(f[2], affine=True)
            self.conv3 = ConvLayer(f[2], f[3], 3, 2);  self.in3 = nn.InstanceNorm2d(f[3], affine=True)
            self.res1 = ResidualBlock(f[3]); self.res2 = ResidualBlock(f[3])
            self.res3 = ResidualBlock(f[3]); self.res4 = ResidualBlock(f[3])
            self.res5 = ResidualBlock(f[3])
            self.upsample_conv1 = ConvLayer(f[3], f[2], 3, 1, upsample=2); self.in4 = nn.InstanceNorm2d(f[2], affine=True)
            self.upsample_conv2 = ConvLayer(f[2], f[1], 3, 1, upsample=2); self.in5 = nn.InstanceNorm2d(f[1], affine=True)
            self.upsample_conv3 = ConvLayer(f[1], f[0], 9, 1)
            self.relu = nn.ReLU()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            y = self.relu(self.in1(self.conv1(x)))
            y = self.relu(self.in2(self.conv2(y)))
            y = self.relu(self.in3(self.conv3(y)))
            y = self.res1(y); y = self.res2(y); y = self.res3(y); y = self.res4(y); y = self.res5(y)
            y = self.relu(self.in4(self.upsample_conv1(y)))
            y = self.relu(self.in5(self.upsample_conv2(y)))
            return self.upsample_conv3(y)

    # ---------------------------------------------------------------------------
    # Load weights and export to ONNX
    # ---------------------------------------------------------------------------
    device = torch.device("cpu")
    net = TransformerNet().to(device)
    ckpt = torch.load(str(pth_path), map_location=device, weights_only=False)
    state = ckpt.get("model_state", ckpt)
    net.load_state_dict(state)
    net.eval()

    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.zeros(1, 3, 256, 256, device=device)
    torch.onnx.export(
        net, dummy, str(onnx_path),
        opset_version=11,
        input_names=["input"], output_names=["output"],
        dynamic_axes={"input": {0: "batch", 2: "height", 3: "width"},
                      "output": {0: "batch", 2: "height", 3: "width"}},
        do_constant_folding=True,
        dynamo=False,  # use stable legacy TorchScript exporter
    )
    print(f"  [ok]       {onnx_path}")


def _generate_preview(onnx_path: Path, preview_path: Path, content_path: Path, *, dry_run: bool) -> None:
    """Run *onnx_path* on *content_path* and save a thumbnail to *preview_path*."""
    if preview_path.exists():
        print(f"  [skip]     {preview_path.name} already exists")
        return
    if not onnx_path.exists():
        print(f"  [skip]     ONNX model not found; cannot generate preview")
        return
    print(f"  [preview]  {onnx_path.name} → {preview_path.name}")
    if dry_run:
        return

    import numpy as np  # noqa: PLC0415
    import onnxruntime as ort  # noqa: PLC0415
    from PIL import Image  # noqa: PLC0415

    # Load and pre-process content image
    img = Image.open(content_path).convert("RGB").resize((PREVIEW_SIZE, PREVIEW_SIZE))
    arr = np.array(img, dtype=np.float32)          # HWC, [0,255]
    arr = arr.transpose(2, 0, 1)[np.newaxis]       # 1CHW, [0,255]

    # Run inference
    sess = ort.InferenceSession(
        str(onnx_path),
        providers=["DmlExecutionProvider", "CPUExecutionProvider"],
    )
    out = sess.run(None, {sess.get_inputs()[0].name: arr})[0]  # 1CHW

    # Post-process and save
    out = np.clip(out[0].transpose(1, 2, 0), 0, 255).astype(np.uint8)  # HWC
    preview_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(out).save(str(preview_path), quality=85)
    print(f"  [ok]       {preview_path}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print what would be done without downloading or exporting.",
    )
    args = parser.parse_args(argv)

    print(f"Project root : {PROJECT_ROOT}")
    print(f"Styles root  : {STYLES_ROOT}")
    print(f"Cache dir    : {TMP_DIR}")
    if args.dry_run:
        print("** DRY RUN — no files will be written **\n")

    # Download shared content image for previews once
    _download(PREVIEW_CONTENT_URL, PREVIEW_CONTENT_PATH, dry_run=args.dry_run)

    for model in MODELS:
        style_id: str = model["id"]
        pth_name: str = model["pth"]
        url: str = f"{YAKHYO_BASE}/{pth_name}"
        pth_path: Path = TMP_DIR / pth_name
        onnx_path: Path = STYLES_ROOT / style_id / "model.onnx"
        preview_path: Path = STYLES_ROOT / style_id / "preview.jpg"

        print(f"\n[{style_id}]")
        _download(url, pth_path, dry_run=args.dry_run)
        _export(pth_path, onnx_path, dry_run=args.dry_run)
        _generate_preview(onnx_path, preview_path, PREVIEW_CONTENT_PATH, dry_run=args.dry_run)

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
