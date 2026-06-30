"""
U-Net image preprocessing and prediction utilities.
"""

from __future__ import annotations

import os
import random
import string
from typing import Tuple

import cv2
import numpy as np
import torch
import segmentation_models_pytorch as smp

from unet.predict import make_predictions


# ---------------------------------------------------------------------------
# Image slicing
# ---------------------------------------------------------------------------

def cutAndPadImg(
    new_name_prefix: str | None,
    img_path: str,
    save_path: str | None,
    size: int,
    is_mask: bool = False,
    scale: float | int = 1.0,
) -> Tuple[list[np.ndarray], int, int, int, int]:
    """Divide *img_path* into square tiles of *size* × *size*.

    Tiles that fall short of *size* are zero-padded on the right/bottom.
    Optionally saves the tiles to *save_path*.

    Parameters
    ----------
    new_name_prefix : str | None
        Filename prefix for saved tiles.  Pass ``None`` to skip saving.
    img_path : str
        Source image path.
    save_path : str | None
        Destination directory for tile files; ``None`` to skip saving.
    size : int
        Tile side length in pixels.
    is_mask : bool
        If ``True``, binarise the image with Otsu's method (default: ``False``).
    scale : float | int
        Scaling factor applied before tiling (default: ``1.0``).

    Returns
    -------
    tiles : list[np.ndarray]
        Cropped tiles in row-major order.
    orig_h : int
        Original image height after scaling.
    orig_w : int
        Original image width after scaling.
    pad_right : int
        Number of columns added by padding.
    pad_bottom : int
        Number of rows added by padding.
    """
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    if is_mask:
        _, img = cv2.threshold(img, 230, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    else:
        img = cv2.medianBlur(img, 7)

    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=interpolation)

    orig_h, orig_w = img.shape
    # Pad up to the next multiple of `size`. Using -orig_w % size (instead of
    # (orig_w // size + 1) * size - orig_w) avoids adding a whole extra row/
    # column of tiles when orig_w/orig_h is already an exact multiple of size.
    pad_right = (-orig_w) % size
    pad_bottom = (-orig_h) % size
    padded = np.pad(img, ((0, pad_bottom), (0, pad_right)), constant_values=0)

    tiles: list[np.ndarray] = []
    tile_idx = 0
    for row in range(0, padded.shape[0], size):
        for col in range(0, padded.shape[1], size):
            tile = padded[row: row + size, col: col + size]
            if is_mask:
                _, tile = cv2.threshold(
                    tile, 230, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU
                )
            if save_path and new_name_prefix is not None:
                cv2.imwrite(
                    os.path.join(save_path, f"{new_name_prefix}_{tile_idx}.jpg"),
                    tile,
                )
            tiles.append(tile)
            tile_idx += 1

    return tiles, orig_h, orig_w, pad_right, pad_bottom


def cutAndPadImgsAndMasks(
    img_root: str,
    mask_root: str,
    save_path_img: str,
    save_path_mask: str,
    size: int,
    scale: float | int = 1.0,
) -> None:
    """Tile all image–mask pairs found in *img_root* / *mask_root*.

    Pairs are matched by filename stem.  Tiles are saved to
    *save_path_img* and *save_path_mask* with random name prefixes so
    tiles from different images do not overwrite each other.

    Parameters
    ----------
    img_root : str
        Directory containing source images (.jpg).
    mask_root : str
        Directory containing mask images (.png).
    save_path_img : str
        Output directory for image tiles.
    save_path_mask : str
        Output directory for mask tiles.
    size : int
        Tile side length in pixels.
    scale : float | int
        Scaling factor applied before tiling (default: ``1.0``).
    """
    for mask_name in os.listdir(mask_root):
        stem = mask_name.split(".")[0]
        prefix = "".join(
            random.choices(string.ascii_uppercase + string.digits, k=10)
        )
        cutAndPadImg(
            new_name_prefix=prefix,
            img_path=os.path.join(img_root, f"{stem}.jpg"),
            save_path=save_path_img,
            size=size,
            is_mask=False,
            scale=scale,
        )
        cutAndPadImg(
            new_name_prefix=prefix,
            img_path=os.path.join(mask_root, f"{stem}.png"),
            save_path=save_path_mask,
            size=size,
            is_mask=True,
            scale=scale,
        )


# ---------------------------------------------------------------------------
# Image splitting helpers
# ---------------------------------------------------------------------------

def moveImgs(
    root: str,
    to_path: str,
    max_size_y: int = 512,
    max_size_x: int = 512,
) -> None:
    """Recursively move and split images from *root* into *to_path*.

    Any image taller than *max_size_y* or wider than *max_size_x* is split
    into smaller sub-images before being written to *to_path*.

    Parameters
    ----------
    root : str
        Source root directory.
    to_path : str
        Flat destination directory for all output images.
    max_size_y : int
        Maximum output image height (default: ``512``).
    max_size_x : int
        Maximum output image width (default: ``512``).
    """
    for sub_name in os.listdir(root):
        sub_dir = os.path.join(root, sub_name)
        for img_name in os.listdir(sub_dir):
            img = cv2.imread(os.path.join(sub_dir, img_name))
            h, w = img.shape[:2]
            if h <= max_size_y:
                continue  # no splitting needed
            print(img.shape)
            for row in range(h // max_size_y + 1):
                for col in range(w // max_size_x + 1):
                    y0, y1 = row * max_size_y, min((row + 1) * max_size_y, h)
                    x0, x1 = col * max_size_x, min((col + 1) * max_size_x, w)
                    tile = img[y0:y1, x0:x1]
                    new_name = (
                        "".join(random.choices(string.ascii_uppercase + string.digits, k=5))
                        + ".jpg"
                    )
                    cv2.imwrite(os.path.join(to_path, new_name), tile)


def joinDivideImgs(
    tiles: list[np.ndarray],
    orig_h: int,
    orig_w: int,
    pad_right: int,
    pad_bottom: int,
) -> np.ndarray:
    """Reassemble tiles produced by :func:`cutAndPadImg`.

    Parameters
    ----------
    tiles : list[np.ndarray]
        Tiles in row-major order (as returned by :func:`cutAndPadImg`).
    orig_h : int
        Original image height (before padding).
    orig_w : int
        Original image width (before padding).
    pad_right : int
        Columns that were added by padding.
    pad_bottom : int
        Rows that were added by padding.

    Returns
    -------
    np.ndarray
        Reconstructed image cropped to ``(orig_h, orig_w)``.
    """
    if not tiles:
        raise ValueError(
            "joinDivideImgs received an empty tile list — nothing to "
            "reassemble. Check that cutAndPadImg / make_predictions "
            "produced at least one tile."
        )

    size = tiles[0].shape[0]
    padded_w = orig_w + pad_right
    tiles_per_row = padded_w // size

    if tiles_per_row == 0 or len(tiles) % tiles_per_row != 0:
        raise ValueError(
            f"Tile count ({len(tiles)}) is not consistent with "
            f"tiles_per_row ({tiles_per_row}) computed from orig_w="
            f"{orig_w}, pad_right={pad_right}, tile size={size}. This "
            "usually means the padding/tiling parameters used to "
            "produce `tiles` do not match orig_w/pad_right passed here."
        )

    rows: list[np.ndarray] = []
    for row_idx in range(len(tiles) // tiles_per_row):
        row_tiles = tiles[row_idx * tiles_per_row: (row_idx + 1) * tiles_per_row]
        row_img = np.concatenate(row_tiles, axis=1)
        rows.append(row_img[:, :orig_w])

    img = np.concatenate(rows, axis=0)
    return img[:orig_h]


# ---------------------------------------------------------------------------
# Full-image prediction
# ---------------------------------------------------------------------------

def predictImgMask(
    img_path: str,
    save_mask_path: str | None,
    divide_size: int,
    model: smp.Unet,
    scale: float | int = 1.0,
) -> np.ndarray:
    """Predict a segmentation mask for a large image.

    The image is split into tiles of *divide_size* × *divide_size*, each
    tile is passed through *model*, and the resulting masks are stitched
    back together.

    Parameters
    ----------
    img_path : str
        Path to the source image.
    save_mask_path : str | None
        If not ``None``, the predicted mask is written here.
    divide_size : int
        Tile side length for splitting.
    model : smp.Unet
        Trained segmentation model.
    scale : float | int
        Scaling factor applied before splitting (default: ``1.0``).

    Returns
    -------
    np.ndarray
        Predicted mask at the original (pre-scale) resolution.
    """
    tiles, orig_h, orig_w, pad_right, pad_bottom = cutAndPadImg(
        new_name_prefix=None,
        img_path=img_path,
        save_path=None,
        size=divide_size,
        is_mask=False,
        scale=scale,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("[INFO] Starting prediction…")

    pred_tiles = [
        make_predictions(
            model=model,
            img_path=None,
            tile_size=divide_size,
            device=device,
            threshold=0.5,
            image=tile,
        )
        for tile in tiles
    ]

    mask = joinDivideImgs(pred_tiles, orig_h, orig_w, pad_right, pad_bottom)

    # Invert scale for the output
    inv_interp = cv2.INTER_AREA if scale > 1 else cv2.INTER_CUBIC
    mask = cv2.resize(mask, None, fx=1 / scale, fy=1 / scale, interpolation=inv_interp)

    if save_mask_path:
        cv2.imwrite(save_mask_path, mask)

    return mask