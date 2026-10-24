"""
U-Net image preprocessing and prediction utilities.
"""

from __future__ import annotations

import logging
import os
import random
import string
from typing import Tuple

import cv2
import numpy as np
import torch
import segmentation_models_pytorch as smp

logger = logging.getLogger(__name__)


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
    overlap: int = 0,
    blur: str = "median",
    blur_ksize: int = 5,
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
    overlap : int
        Pixels of overlap between adjacent tiles (default: ``0``, i.e. the
        original non-overlapping behaviour). ``overlap // 2`` pixels of
        context are added around every tile's non-overlapping "core";
        pass the *same* value to :func:`joinDivideImgs` to reconstruct the
        image — it crops that context back off before stitching, which
        removes the hard seams plain edge-to-edge tiling produces. Must
        be smaller than *size*.
    blur, blur_ksize : str, int
        Smoothing of an image (not a mask) before scaling, as
        :func:`apply_blur`: ``"median"`` (default), ``"gaussian"``,
        ``"bilateral"`` or ``"none"``, odd kernel in px (default ``5``).
        Train on such tiles with the same values (``unet.train.train``'s
        *blur* / *blur_ksize*) so the model records what it was trained on.

    Returns
    -------
    tiles : list[np.ndarray]
        Cropped tiles in row-major order.
    orig_h : int
        Original image height after scaling.
    orig_w : int
        Original image width after scaling.
    pad_right : int
        Number of columns added by padding beyond the left/top context
        margin (``overlap // 2``).
    pad_bottom : int
        Number of rows added by padding beyond the left/top context
        margin (``overlap // 2``).
    """
    if not 0 <= overlap < size:
        raise ValueError(f"overlap ({overlap}) must be in [0, size) — size={size}")

    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    if is_mask:
        _, img = cv2.threshold(img, 230, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    else:
        img = apply_blur(img, *normalize_blur(blur, blur_ksize))

    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=interpolation)

    orig_h, orig_w = img.shape

    # `size` = `stride` (the non-overlapping core every tile contributes to
    # the reconstruction) plus `margin` context pixels on each side. With
    # overlap=0, margin=0 and stride=size, reducing exactly to the old
    # non-overlapping tiling.
    margin = overlap // 2
    stride = size - 2 * margin

    n_h = -(-orig_h // stride)  # ceil division: core blocks needed to cover orig_h
    n_w = -(-orig_w // stride)

    pad_top = pad_left = margin
    # Right/bottom padding covers both the trailing context margin and
    # whatever is needed to fill out the last row/column of core blocks.
    pad_bottom = n_h * stride + margin - orig_h
    pad_right = n_w * stride + margin - orig_w
    padded = np.pad(
        img, ((pad_top, pad_bottom), (pad_left, pad_right)), constant_values=0
    )

    tiles: list[np.ndarray] = []
    tile_idx = 0
    for row in range(0, n_h * stride, stride):
        for col in range(0, n_w * stride, stride):
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
    blur: str = "median",
    blur_ksize: int = 5,
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
    blur, blur_ksize : str, int
        Smoothing of the images (not the masks), see :func:`cutAndPadImg`.
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
            blur=blur,
            blur_ksize=blur_ksize,
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
            logger.debug("%s: %s, splitting", img_name, img.shape)
            # Ceiling division: `h // max_size_y + 1` over-counts by one row
            # whenever h is an exact multiple of max_size_y, producing a
            # trailing 0-height tile that crashes cv2.imwrite (same for w).
            n_rows = -(-h // max_size_y)
            n_cols = -(-w // max_size_x)
            for row in range(n_rows):
                for col in range(n_cols):
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
    overlap: int = 0,
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
        Columns that were added by padding, as returned by
        :func:`cutAndPadImg` (used only to sanity-check *overlap* matches
        what tiles were actually cut with).
    pad_bottom : int
        Rows that were added by padding, as returned by
        :func:`cutAndPadImg` (same sanity-check role as *pad_right*).
    overlap : int
        Must match the *overlap* passed to :func:`cutAndPadImg`. When
        greater than zero, the outer ``overlap // 2`` pixels of every tile
        are context predicted with less information near the tile's own
        edge; they are cropped away and only each tile's non-overlapping
        core is stitched, which avoids the hard seams plain edge-to-edge
        concatenation (``overlap=0``) leaves at tile boundaries.

    Returns
    -------
    np.ndarray
        Reconstructed image cropped to ``(orig_h, orig_w)``.
    """
    if not tiles:
        raise ValueError(
            "joinDivideImgs received an empty tile list — nothing to "
            "reassemble. Check that cutAndPadImg (and the model run on "
            "its tiles) produced at least one tile."
        )

    size = tiles[0].shape[0]
    margin = overlap // 2
    stride = size - 2 * margin
    if stride <= 0:
        raise ValueError(
            f"overlap ({overlap}) must be smaller than tile size ({size})"
        )

    n_h = -(-orig_h // stride)  # ceil division, mirrors cutAndPadImg
    n_w = -(-orig_w // stride)

    expected_pad_right = n_w * stride + margin - orig_w
    expected_pad_bottom = n_h * stride + margin - orig_h
    if pad_right != expected_pad_right or pad_bottom != expected_pad_bottom:
        raise ValueError(
            f"pad_right/pad_bottom ({pad_right}, {pad_bottom}) don't match "
            f"what overlap={overlap} implies ({expected_pad_right}, "
            f"{expected_pad_bottom}) for orig_h={orig_h}, orig_w={orig_w}, "
            f"tile size={size}. This usually means `overlap` here doesn't "
            "match the overlap cutAndPadImg was called with."
        )

    if len(tiles) != n_h * n_w:
        raise ValueError(
            f"Tile count ({len(tiles)}) does not match the {n_h}x{n_w} "
            f"grid computed from orig_h={orig_h}, orig_w={orig_w}, "
            f"size={size}, overlap={overlap}. This usually means the "
            "padding/tiling parameters used to produce `tiles` do not "
            "match the arguments passed here."
        )

    # Keep only each tile's non-overlapping stride×stride core (dropping
    # the `margin`-pixel context border on every side) and place the cores
    # edge to edge — that context was there purely to give the model
    # neighbourhood information near the tile's own boundary; stitching by
    # core instead of by full (overlapping) tile is what removes seams.
    rows: list[np.ndarray] = []
    for row_idx in range(n_h):
        row_tiles = tiles[row_idx * n_w: (row_idx + 1) * n_w]
        cores = [t[margin: margin + stride, margin: margin + stride] for t in row_tiles]
        rows.append(np.concatenate(cores, axis=1))

    img = np.concatenate(rows, axis=0)
    return img[:orig_h, :orig_w]


# ---------------------------------------------------------------------------
# Full-image prediction
# ---------------------------------------------------------------------------

def predictImgMask(
    img_path: str,
    save_mask_path: str | None,
    divide_size: int,
    model: smp.Unet,
    scale: float | int = 1.0,
    overlap: int = 0,
    input_size: int | None = None,
    threshold: float = 0.5,
    in_channels: int | None = None,
    blur: str | None = None,
    blur_ksize: int | None = None,
) -> np.ndarray:
    """Predict a segmentation mask for a large image.

    The image is split into tiles of *divide_size* × *divide_size*, each
    tile is resized to the network input *input_size* (default: the tile
    size itself), passed through *model*, resized back and the resulting
    masks are stitched together.

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
    overlap : int
        Pixels of overlap between adjacent tiles before stitching
        (default: ``0``, i.e. plain edge-to-edge tiling). Increasing this
        gives the model neighbourhood context near every tile boundary
        and removes stitching seams there (see Notes). Most useful where the feature scale is
        small relative to *divide_size* (e.g. dense latewood), since more
        tile boundaries then cross real structure.
    input_size : int | None
        Side the model was trained on, when tiles were resized for training
        (e.g. 256 px tiles fed to the network as 128 px).
    threshold : float
        Lumen probability above which a pixel is white (default ``0.5``).
    in_channels : int | None
        1 (grayscale + median blur) or 3 (RGB); default: the model's
        ``picdens_meta["in_channels"]``, 1 for models without it. See
        :func:`preprocess_image`.
    blur, blur_ksize : str | None, int | None
        Smoothing before the network (:data:`BLUR_TYPES`, odd kernel in px);
        default (*blur* ``None``): what the model was trained with
        (:func:`model_blur`). Should match training.

    Returns
    -------
    np.ndarray
        Predicted mask ``(H, W, 3)``, 0/255, at the original (pre-scale)
        resolution.

    Notes
    -----
    Unlike :func:`cutAndPadImg`, the image is not zero-padded up to a whole
    number of tiles: the last tile of every row/column is shifted back to end
    at the image edge. Padding made the network process up to 4× more pixels
    than the image has (a 512 px image with 512 px tiles and an overlap became
    a 2×2 grid) and the black field degraded predictions near the edges. Only
    an image smaller than one tile is padded. Each tile contributes its core
    (``overlap // 2`` px are dropped on every side facing another tile); the
    probabilities are stitched and thresholded once, after scaling back.
    """
    input_size = input_size or divide_size
    if not 0 <= overlap < divide_size:
        raise ValueError(f"overlap ({overlap}) must be in [0, size) — size={divide_size}")

    if in_channels is None:
        in_channels = int((getattr(model, "picdens_meta", None) or {}).get("in_channels", 1))
    if blur is None:
        blur, blur_ksize = model_blur(model, in_channels)
    # Same preprocessing as in training (unet.train.train).
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(img_path)
    src_h, src_w = img.shape[:2]
    img = preprocess_image(img, scale, in_channels, blur, blur_ksize)
    h, w = img.shape[:2]

    size, margin = divide_size, overlap // 2
    pad = ((0, max(0, size - h)), (0, max(0, size - w))) + ((0, 0),) * (img.ndim - 2)
    img = np.pad(img, pad)  # only an image < one tile
    padded_h, padded_w = img.shape[:2]
    img = img.astype(np.float32) / 255.0

    device = next(model.parameters()).device
    model.eval()
    logger.debug("Prediction: %s, %dx%d px, tile %d, overlap %d", img_path, w, h, size, overlap)
    prob = np.zeros((padded_h, padded_w), np.float32)
    with torch.no_grad():
        for y in _tile_starts(padded_h, size, size - 2 * margin):
            for x in _tile_starts(padded_w, size, size - 2 * margin):
                tile = img[y: y + size, x: x + size]
                if input_size != size:
                    tile = cv2.resize(tile, (input_size, input_size))
                out = _tile_probability(model, tile, device)
                if input_size != size:
                    out = cv2.resize(out, (size, size))
                # this tile's core; at the image edges - up to the edge
                y0 = 0 if y == 0 else y + margin
                x0 = 0 if x == 0 else x + margin
                y1 = padded_h if y + size >= padded_h else y + size - margin
                x1 = padded_w if x + size >= padded_w else x + size - margin
                prob[y0:y1, x0:x1] = out[y0 - y: y1 - y, x0 - x: x1 - x]

    prob = prob[:h, :w]
    if prob.shape != (src_h, src_w):  # invert scale
        prob = cv2.resize(prob, (src_w, src_h), interpolation=cv2.INTER_LINEAR)
    mask = (prob > threshold).astype(np.uint8) * 255
    # (H, W, 3), as before: callers convert it with cv2.cvtColor
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)

    if save_mask_path:
        cv2.imwrite(save_mask_path, mask)

    return mask


def _tile_starts(n: int, size: int, step: int) -> list[int]:
    """Offsets of tiles of *size* along a side of *n* >= *size* px: every
    *step* px, the last one shifted back to end exactly at the edge."""
    starts = list(range(0, n - size + 1, step))
    if starts[-1] + size < n:
        starts.append(n - size)
    return starts


#: Smoothing a U-Net model's input can be trained with (see :func:`apply_blur`).
BLUR_TYPES = ("none", "median", "gaussian", "bilateral")


def default_blur(in_channels: int) -> tuple[str, int]:
    """Blur of models that do not record it: a 5 px median for grayscale (what
    :func:`cutAndPadImg` always applied), none for RGB."""
    return ("median", 5) if in_channels == 1 else ("none", 1)


def normalize_blur(blur: str | None, ksize: int | None, in_channels: int = 1) -> tuple[str, int]:
    """``(type, kernel)`` with defaults filled in: *blur* ``None`` - :func:`default_blur`;
    *ksize* ``None`` - 5 px; a kernel of 0/1 or type ``"none"`` -> ``("none", 1)``.
    Raises ``ValueError`` for an unknown type or an even kernel."""
    if blur is None:
        default_type, default_ksize = default_blur(in_channels)
        blur, ksize = default_type, default_ksize if ksize is None else ksize
    blur = str(blur).lower()
    if blur not in BLUR_TYPES:
        raise ValueError(f"unknown blur {blur!r}; known: {BLUR_TYPES}")
    ksize = 5 if ksize is None else int(ksize)
    if blur == "none" or ksize <= 1:
        return "none", 1
    if ksize % 2 == 0:
        raise ValueError(f"blur kernel must be odd, not {ksize}")
    return blur, ksize


def model_blur(model, in_channels: int) -> tuple[str, int]:
    """The blur *model* was trained with: ``picdens_meta["blur"]`` / ``["blur_ksize"]``, a
    median of ``["median_ksize"]`` px (models of an earlier version), else :func:`default_blur`."""
    meta = getattr(model, "picdens_meta", None) or {}
    if meta.get("blur") is not None:
        return normalize_blur(meta["blur"], meta.get("blur_ksize"), in_channels)
    if meta.get("median_ksize") is not None:
        return normalize_blur("median", meta["median_ksize"], in_channels)
    return default_blur(in_channels)


def blur_text(blur: str, ksize: int) -> str:
    """Human-readable blur, e.g. ``"медианный 5×5"`` or ``"нет"``."""
    names = {"median": "медианный", "gaussian": "гауссов", "bilateral": "билатеральный"}
    return f"{names[blur]} {ksize}×{ksize}" if blur != "none" else "нет"


def apply_blur(img: np.ndarray, blur: str, ksize: int) -> np.ndarray:
    """Smooth a uint8 image (grayscale or RGB) with a *ksize* px kernel:

    * ``"median"`` - removes speckle, keeps edges, but erases bright details narrower than
      about half the kernel (a 5 px median - lumens under ~3 px);
    * ``"gaussian"`` - blurs everything evenly, edges and narrow lumens included (sigma from
      the kernel size);
    * ``"bilateral"`` - averages only pixels of similar brightness, so it smooths walls and
      lumens but keeps the wall/lumen edge (*ksize* = neighbourhood diameter, colour sigma 30,
      space sigma *ksize*/2); slower than the others;
    * ``"none"`` - unchanged.
    """
    if blur == "none" or ksize <= 1:
        return img
    if blur == "median":
        return cv2.medianBlur(img, ksize)
    if blur == "gaussian":
        return cv2.GaussianBlur(img, (ksize, ksize), 0)
    if blur == "bilateral":
        return cv2.bilateralFilter(img, ksize, 30, ksize / 2)
    raise ValueError(f"unknown blur {blur!r}; known: {BLUR_TYPES}")


def preprocess_image(
    img: np.ndarray, scale: float | int = 1.0, in_channels: int = 1,
    blur: str | None = None, blur_ksize: int | None = None,
) -> np.ndarray:
    """A BGR image (as read by ``cv2.imread``) the way U-Net models see it: ``(H, W)`` grayscale
    for ``in_channels=1``, ``(H, W, 3)`` RGB for ``in_channels=3``; smoothed by :func:`apply_blur`
    (*blur* ``None``: :func:`default_blur` - a 5 px median for grayscale, as :func:`cutAndPadImg`,
    none for RGB); scaled by *scale*, uint8. Training and binarization must use the same blur."""
    if in_channels == 1:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if img.ndim == 3 else img
    elif in_channels == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    else:
        raise ValueError(f"in_channels must be 1 or 3, not {in_channels}")
    img = apply_blur(img, *normalize_blur(blur, blur_ksize, in_channels))
    if scale != 1:
        interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=interpolation)
    return img


def _tile_probability(model: torch.nn.Module, tile: np.ndarray, device) -> np.ndarray:
    """Lumen probability ``(H, W)`` for one float tile in [0, 1], ``(H, W)`` or ``(H, W, C)``."""
    x = tile[np.newaxis, np.newaxis] if tile.ndim == 2 else tile.transpose(2, 0, 1)[np.newaxis]
    raw = model(torch.from_numpy(np.ascontiguousarray(x)).to(device))  # (1, C, H, W)
    if raw.shape[1] == 1:
        logits = raw[0, 0]
    elif raw.shape[1] == 3:
        # legacy checkpoints (trained before classes=1) output the mask duplicated into
        # 3 near-identical channels: averaging their logits collapses them back to one
        logits = raw[0].mean(dim=0)
    else:
        raise ValueError(
            f"Model output has {raw.shape[1]} channels; only 1 (current architecture) or 3 "
            "(legacy checkpoint) are supported."
        )
    return torch.sigmoid(logits).cpu().numpy()
