"""
Tile-based image preprocessing for U-Net inference.

Split large micrograph images into overlapping tiles before prediction,
then stitch the predicted masks back together.
"""

from __future__ import annotations

import os
import random
import string

import cv2
import numpy as np


def cut_and_pad(
    img_path: str,
    size: int,
    is_mask: bool = False,
    scale: float | int = 1.0,
    save_path: str | None = None,
    name_prefix: str | None = None,
) -> tuple[list[np.ndarray], int, int, int, int]:
    """Divide an image into square tiles of *size* × *size*.

    Tiles that fall short of *size* are zero-padded on the right/bottom.

    Parameters
    ----------
    img_path : str
        Source image path.
    size : int
        Tile side length in pixels.
    is_mask : bool
        If ``True``, binarise the image with Otsu before tiling.
    scale : float | int
        Scaling factor applied before tiling.
    save_path : str | None
        If provided, tiles are written here as JPEG files.
    name_prefix : str | None
        Filename prefix for saved tiles (required when *save_path* is set).

    Returns
    -------
    tiles : list[np.ndarray]
        Tiles in row-major order.
    orig_h, orig_w : int
        Image dimensions after scaling (before padding).
    pad_right, pad_bottom : int
        Columns / rows added by padding.
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
    pad_right  = (orig_w // size + 1) * size - orig_w
    pad_bottom = (orig_h // size + 1) * size - orig_h
    padded = np.pad(img, ((0, pad_bottom), (0, pad_right)), constant_values=0)

    tiles: list[np.ndarray] = []
    for idx, (row, col) in enumerate(
        (r, c)
        for r in range(0, padded.shape[0], size)
        for c in range(0, padded.shape[1], size)
    ):
        tile = padded[row: row + size, col: col + size]
        if is_mask:
            _, tile = cv2.threshold(
                tile, 230, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU
            )
        if save_path and name_prefix:
            cv2.imwrite(os.path.join(save_path, f"{name_prefix}_{idx}.jpg"), tile)
        tiles.append(tile)

    return tiles, orig_h, orig_w, pad_right, pad_bottom


def join_tiles(
    tiles: list[np.ndarray],
    orig_h: int,
    orig_w: int,
    pad_right: int,
    pad_bottom: int,
) -> np.ndarray:
    """Reassemble tiles produced by :func:`cut_and_pad`.

    Parameters
    ----------
    tiles : list[np.ndarray]
        Tiles in row-major order.
    orig_h, orig_w : int
        Original image dimensions (before padding).
    pad_right, pad_bottom : int
        Padding added during cutting.

    Returns
    -------
    np.ndarray
        Reconstructed image cropped to ``(orig_h, orig_w)``.
    """
    size = tiles[0].shape[0]
    tiles_per_row = (orig_w + pad_right) // size

    rows = [
        np.concatenate(
            [t[:, :orig_w] for t in tiles[i * tiles_per_row: (i + 1) * tiles_per_row]],
            axis=1,
        )
        for i in range(len(tiles) // tiles_per_row)
    ]
    return np.concatenate(rows, axis=0)[:orig_h]


def tile_image_mask_pairs(
    img_root: str,
    mask_root: str,
    save_img_path: str,
    save_mask_path: str,
    size: int,
    scale: float | int = 1.0,
) -> None:
    """Tile all image–mask pairs in *img_root* / *mask_root*.

    Output tiles are saved with random name prefixes to avoid collisions.
    """
    for mask_name in os.listdir(mask_root):
        stem   = mask_name.split(".")[0]
        prefix = "".join(random.choices(string.ascii_uppercase + string.digits, k=10))
        cut_and_pad(
            os.path.join(img_root,  f"{stem}.jpg"), size,
            is_mask=False, scale=scale,
            save_path=save_img_path, name_prefix=prefix,
        )
        cut_and_pad(
            os.path.join(mask_root, f"{stem}.png"), size,
            is_mask=True, scale=scale,
            save_path=save_mask_path, name_prefix=prefix,
        )