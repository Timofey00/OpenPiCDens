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

    Raises
    ------
    ValueError
        If neither *img_path* nor *image* is provided.
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

        pred = model(tensor).squeeze()          # (3, H, W)
        pred = torch.sigmoid(pred)
        pred = pred.cpu().numpy().transpose(1, 2, 0)  # (H, W, 3)
        pred = (pred > threshold).astype(np.uint8) * 255

    return pred