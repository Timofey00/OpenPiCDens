"""
Image binarisation for wood micrograph analysis.

The single public class :class:`Binarizer` converts a grayscale or
colour micrograph to a binary (black/white) image ready for porosity
profiling.  Four thresholding strategies are supported, each with an
optional pre-blur step.

Typical usage::

    bi = Binarizer(method="Otsu", blur="Median", ksize=3)
    binary = bi.binarize("path/to/ring.jpg")

    # or binarise a whole directory tree
    bi.binarize_root(src_root="imgs/", dst_root="bi_imgs/")
"""

from __future__ import annotations

import os

import cv2
import numpy as np
import torch

from .utils import initPath

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_BLUR_TYPES = frozenset({"Gaussian", "Median", "NBF", "Bilateral"})
_BI_METHODS = frozenset({"Otsu", "Mean", "Gaussian", "Triangle", "UNET"})


# ---------------------------------------------------------------------------
# Public class
# ---------------------------------------------------------------------------

class Binarizer:
    """Convert micrograph images to binary (black/white) format.

    Parameters
    ----------
    method : str
        Thresholding method. One of ``"Otsu"`` (default), ``"Mean"``,
        ``"Gaussian"``, ``"Triangle"``, ``"UNET"``.
    blur : str
        Blurring method applied before thresholding. One of
        ``"Gaussian"``, ``"Median"`` (default), ``"NBF"``,
        ``"Bilateral"``, or ``""`` / ``None`` for no blur.
    ksize : int
        Kernel size for blurring; must be odd (default: ``3``).
    gamma_equalisation : bool
        Apply histogram equalisation before blurring (default: ``False``).
    constant_th : int
        Fixed threshold used when *method* is not a named method
        (fallback path, default: ``235``).
    model_path : str | None
        Path to a saved PyTorch U-Net model. Required when
        ``method="UNET"`` (default: ``None``).
    scale : float | int
        Scaling factor applied to images before processing
        (default: ``1.0``).

    Raises
    ------
    ValueError
        If ``method="UNET"`` but *model_path* is ``None``.
    """

    def __init__(
        self,
        method: str = "Otsu",
        blur: str = "Median",
        ksize: int = 3,
        gamma_equalisation: bool = False,
        constant_th: int = 235,
        model_path: str | None = None,
        scale: float | int = 1.0,
    ) -> None:
        if method == "UNET" and model_path is None:
            raise ValueError("model_path is required when method='UNET'")

        self.method = method
        self.blur = blur
        self.ksize = ksize
        self.gamma_equalisation = gamma_equalisation
        self.constant_th = constant_th
        self.scale = scale
        self._model: object | None = None

        if method == "UNET":
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self._model = torch.load(model_path).to(device)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def binarize(self, img_path: str) -> np.ndarray:
        """Load *img_path*, apply blur and threshold, return binary image.

        Parameters
        ----------
        img_path : str
            Path to the source image.

        Returns
        -------
        np.ndarray
            Binary image; pixel values are either ``0`` or ``255``.
        """
        img = cv2.imread(img_path)
        if self.scale != 1.0:
            img = cv2.resize(
                img, None, fx=self.scale, fy=self.scale,
                interpolation=cv2.INTER_CUBIC,
            )
        # Convert to grayscale before any single-channel operations
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if self.gamma_equalisation:
            img = cv2.equalizeHist(img)

        img = self._apply_blur(img)
        return self._apply_threshold(img, img_path)

    def binarize_root(self, src_root: str, dst_root: str) -> None:
        """Binarise every image under *src_root* and mirror results to *dst_root*.

        The directory layout ``src_root/<id>/<img>`` is preserved under
        *dst_root*.  Sub-directories must be named with integer IDs;
        image filenames must have integer stems (used for sorting).

        Parameters
        ----------
        src_root : str
            Source root directory.
        dst_root : str
            Destination root directory.
        """
        sub_dirs = sorted(os.listdir(src_root), key=int)
        for sub in sub_dirs:
            src_dir = os.path.join(src_root, sub)
            img_names = sorted(
                os.listdir(src_dir), key=lambda f: int(f.split(".")[0])
            )
            dst_dir = os.path.join(dst_root, sub)
            initPath(dst_dir)
            for img_name in img_names:
                bi_img = self.binarize(os.path.join(src_dir, img_name))
                cv2.imwrite(os.path.join(dst_dir, img_name), bi_img)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _apply_blur(self, img: np.ndarray) -> np.ndarray:
        """Apply the configured blur to *img* (expects grayscale input)."""
        if self.blur == "Gaussian":
            return cv2.GaussianBlur(img, (self.ksize, self.ksize), 0)
        if self.blur == "Median":
            return cv2.medianBlur(img, self.ksize)
        if self.blur == "NBF":
            return cv2.blur(img, (self.ksize, self.ksize))
        if self.blur == "Bilateral":
            # bilateralFilter accepts single-channel images
            return cv2.bilateralFilter(img, 11, 41, 21)
        return img

    def _apply_threshold(self, img: np.ndarray, img_path: str) -> np.ndarray:
        """Apply the configured thresholding method to *img* (expects grayscale)."""
        # Ensure single-channel at threshold stage
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        if self.method == "Otsu":
            _, bi = cv2.threshold(
                img, 230, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU
            )
            return bi

        if self.method in ("Mean", "Gaussian"):
            return cv2.adaptiveThreshold(
                img, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 11, 2,
            )

        if self.method == "Triangle":
            from skimage import filters as skfilters
            thresh = skfilters.threshold_triangle(img)
            return (img > thresh).astype(np.uint8) * 255

        if self.method == "UNET":
            from unet.utills import predictImgMask
            bi = predictImgMask(
                img_path=img_path,
                save_mask_path=None,
                divide_size=128,
                model=self._model,
                scale=self.scale,
            )
            bi = cv2.cvtColor(bi, cv2.COLOR_RGB2GRAY)
            _, bi = cv2.threshold(
                bi, 200, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU
            )
            return bi

        # Fallback: fixed threshold
        _, bi = cv2.threshold(
            img, self.constant_th, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU
        )
        return bi