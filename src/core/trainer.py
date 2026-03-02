"""StyleTrainer — trains a TransformerNet for a single style image.

Training loop (Johnson 2016):
  1. Load style reference image; precompute VGG Gram matrices once.
  2. For each content image batch from MS-COCO:
       a. Forward pass through TransformerNet.
       b. Compute perceptual loss (content + style) via frozen VGG-16.
       c. Backprop, step Adam optimiser, checkpoint periodically.
  3. Export checkpoint to ONNX after training.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.ml.train_utils import CocoImageDataset, load_style_tensor
from src.ml.transformer_net import TransformerNet
from src.ml.vgg_loss import VGGPerceptualLoss

logger: logging.Logger = logging.getLogger(__name__)


def _resolve_device(device_str: str) -> torch.device:
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(device_str)


class StyleTrainer:
    """Trains one TransformerNet for a given style reference image.

    Example::

        trainer = StyleTrainer()
        trainer.train(
            style_images=[Path("my_style.jpg")],
            coco_dataset_path=Path("data/train2017"),
            output_model_path=Path("styles/my_style/model.pth"),
        )
        trainer.export_onnx(
            Path("styles/my_style/model.pth"),
            Path("styles/my_style/model.onnx"),
        )
    """

    def __init__(self, device: str = "auto") -> None:
        self._device: torch.device = _resolve_device(device)
        logger.info("StyleTrainer using device: %s", self._device)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        style_images: list[Path],
        coco_dataset_path: Path,
        output_model_path: Path,
        epochs: int = 2,
        batch_size: int = 4,
        image_size: int = 256,
        style_size: int | None = None,
        style_weight: float = 1e8,
        content_weight: float = 1e5,
        learning_rate: float = 1e-3,
        checkpoint_interval: int = 2000,
        checkpoint_path: Path | None = None,
        progress_callback: Callable[[int, int, float], None] | None = None,
    ) -> Path:
        """Train TransformerNet and save the final PyTorch checkpoint.

        Args:
            style_images:          One or more style reference image paths.
                                   If multiple, the first is used; future
                                   versions may blend multiple styles.
            coco_dataset_path:     Root of the MS-COCO images folder.
            output_model_path:     Where to save the trained .pth file.
            epochs:                Number of full passes over the dataset.
            batch_size:            Training batch size.
            image_size:            Resize content images to this (square) size.
            style_size:            Resize style image to this; None = original.
            style_weight:          Weight for style loss term.
            content_weight:        Weight for content loss term.
            learning_rate:         Adam learning rate.
            checkpoint_interval:   Save a checkpoint every N images processed.
            checkpoint_path:       Resume from this .pth checkpoint if given.
            progress_callback:     Called as callback(images_done, total, loss).

        Returns:
            Path to the saved .pth model.
        """
        device = self._device
        output_model_path.parent.mkdir(parents=True, exist_ok=True)

        # --- Build model ---
        net = TransformerNet().to(device)
        optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
        start_epoch: int = 0

        if checkpoint_path is not None and checkpoint_path.exists():
            ckpt: dict = torch.load(str(checkpoint_path), map_location=device)
            net.load_state_dict(ckpt["model_state"])
            optimizer.load_state_dict(ckpt["optimizer_state"])
            start_epoch = ckpt.get("epoch", 0)
            logger.info("Resumed from checkpoint %s (epoch %d)", checkpoint_path, start_epoch)

        # --- Loss ---
        loss_fn = VGGPerceptualLoss().to(device)
        loss_fn.eval()

        # --- Precompute style Grams ---
        style_tensor = load_style_tensor(style_images[0], size=style_size).to(device)
        style_grams = loss_fn.compute_style_grams(style_tensor)

        # --- Dataset ---
        dataset = CocoImageDataset(coco_dataset_path, image_size=image_size)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=(device.type == "cuda"),
        )
        total_images: int = len(dataset) * epochs

        # --- Training loop ---
        images_done: int = 0
        for epoch in range(start_epoch, epochs):
            net.train()
            for batch in loader:
                content: torch.Tensor = batch.to(device)
                output = net(content)
                c_loss, s_loss = loss_fn(output, content, style_grams)
                loss: torch.Tensor = content_weight * c_loss + style_weight * s_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                images_done += content.size(0)
                if progress_callback is not None:
                    progress_callback(images_done, total_images, loss.item())

                # Periodic checkpoint
                if checkpoint_interval > 0 and images_done % checkpoint_interval < batch_size:
                    ckpt_file = output_model_path.with_suffix(
                        f".ckpt_{images_done}.pth"
                    )
                    torch.save({
                        "model_state": net.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "epoch": epoch,
                        "images_done": images_done,
                    }, str(ckpt_file))
                    logger.info(
                        "Checkpoint saved: %s  loss=%.4f", ckpt_file, loss.item()
                    )

        # --- Save final model (state dict only, no optimizer) ---
        torch.save({"model_state": net.state_dict()}, str(output_model_path))
        logger.info("Training complete. Model saved to %s", output_model_path)
        return output_model_path

    # ------------------------------------------------------------------
    # ONNX export
    # ------------------------------------------------------------------

    def export_onnx(
        self,
        pth_path: Path,
        onnx_path: Path,
        image_size: int = 256,
    ) -> Path:
        """Export a trained .pth checkpoint to ONNX.

        Args:
            pth_path:   Path to the saved PyTorch state dict.
            onnx_path:  Destination .onnx path.
            image_size: Reference image size used for tracing (actual size
                        is variable thanks to dynamic_axes).

        Returns:
            Path to the written .onnx file.
        """
        onnx_path.parent.mkdir(parents=True, exist_ok=True)
        device = torch.device("cpu")  # always export from CPU

        net = TransformerNet().to(device)
        ckpt: dict = torch.load(str(pth_path), map_location=device)
        state: dict = ckpt.get("model_state", ckpt)  # handle bare state dicts
        net.load_state_dict(state)
        net.eval()

        dummy = torch.zeros(1, 3, image_size, image_size, device=device)
        torch.onnx.export(
            net,
            dummy,
            str(onnx_path),
            opset_version=11,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={
                "input":  {0: "batch", 2: "height", 3: "width"},
                "output": {0: "batch", 2: "height", 3: "width"},
            },
            do_constant_folding=True,
            dynamo=False,  # type: ignore[call-arg]  # use the stable legacy exporter; stubs lag behind
        )
        logger.info("ONNX model exported to %s", onnx_path)
        return onnx_path
