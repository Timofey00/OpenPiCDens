"""
U-Net tile prediction.
"""

from __future__ import annotations

import cv2
import numpy as np
import segmentation_models_pytorch as smp
import torch


def make_predictions(
    model: smp.Unet,
    img_path: str | None,
    tile_size: int,
    device: str,
    threshold: float = 0.5,
    image: np.ndarray | None = None,
) -> np.ndarray:
    """Predict a segmentation mask for a single tile.

    Either *img_path* or *image* must be provided — not both.

    Parameters
    ----------
    model : smp.Unet
        Trained segmentation model.
    img_path : str | None
        Path to a source image. When provided, *image* is ignored.
    tile_size : int
        Height and width the tile is resized to before inference.
        Must match the resolution the model was trained on.
    device : str
        PyTorch device string: ``"cuda"`` or ``"cpu"``.
    threshold : float
        Minimum sigmoid probability to classify a pixel as foreground
        (default: ``0.5``).
    image : np.ndarray | None
        Pre-loaded grayscale tile. Used when *img_path* is ``None``.

    Returns
    -------
    np.ndarray
        Predicted binary mask, shape ``(tile_size, tile_size, 3)``,
        dtype uint8, values 0 or 255.

    Notes
    -----
    Supports both current checkpoints (``classes=1``) and legacy
    checkpoints (``classes=3``, trained before the U-Net was switched
    to single-channel output). For legacy checkpoints the 3 output
    logits are averaged before applying sigmoid.

    Raises
    ------
    ValueError
        If neither *img_path* nor *image* is provided, or if the model
        produces an unsupported number of output channels.
    """
    if img_path is None and image is None:
        raise ValueError("Provide either img_path or image.")

    model.eval()
    with torch.no_grad():
        # Load or convert to grayscale float32
        if img_path is not None:
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            # image arrives as a 2-D grayscale tile from cut_and_pad
            img = image

        img = img.astype("float32") / 255.0
        img = cv2.resize(img, (tile_size, tile_size))

        # (H, W) → (1, 1, H, W) batch tensor
        tensor = torch.from_numpy(img[np.newaxis, np.newaxis]).to(device)

        raw = model(tensor)  # (1, C, H, W)
        n_channels = raw.shape[1]

        if n_channels == 1:
            # Current architecture: binary segmentation, single output channel.
            logits = raw.squeeze(0).squeeze(0)      # (H, W)
        elif n_channels == 3:
            # Legacy checkpoints trained before the U-Net was fixed to use
            # classes=1: the mask was duplicated into 3 identical channels
            # via cv2.cvtColor(mask, COLOR_GRAY2RGB) during training, so the
            # 3 output channels are (approximately) redundant copies of the
            # same prediction. Averaging the logits before the sigmoid is
            # equivalent to averaging 3 near-identical probability maps and
            # is the standard way to collapse such a checkpoint back to one
            # channel without retraining.
            logits = raw.squeeze(0).mean(dim=0)     # (3, H, W) -> (H, W)
        else:
            raise ValueError(
                f"Model output has {n_channels} channels; only 1 (current "
                "architecture) or 3 (legacy checkpoint) are supported. "
                "Re-train the model with classes=1, or load a compatible "
                "checkpoint."
            )

        pred = torch.sigmoid(logits)
        pred = pred.cpu().numpy()               # (H, W)
        pred = (pred > threshold).astype(np.uint8) * 255
        # Return as (H, W, 3) RGB so callers can use cv2.cvtColor on the result
        pred = np.stack([pred, pred, pred], axis=-1)

    return pred