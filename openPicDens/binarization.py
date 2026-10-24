"""
Image binarisation for wood micrograph analysis.

The single public class :class:`Binarizer` converts a grayscale or
colour micrograph to a binary (black/white) image ready for porosity
profiling.  Global thresholds (OpenCV's Otsu and scikit-image's
Isodata, Li, Yen, Minimum, Mean, Triangle, multi-Otsu), local ones
(OpenCV adaptive mean/Gaussian, scikit-image ``threshold_local``,
Niblack, Sauvola), hysteresis, each with an optional pre-blur step,
plus two learned methods: a U-Net and a random-forest pixel classifier
(``"RF"``, see :mod:`openPicDens.rf_binarization`). :data:`BI_METHODS`
lists them all.

Typical usage::

    bi = Binarizer(method="Otsu", blur="Median", ksize=3)
    binary = bi.binarize("path/to/ring.jpg")

    # or binarise a whole directory tree
    bi.binarize_root(src_root="imgs/", dst_root="bi_imgs/")

    # compare every method: dst/<method>/<image name>
    binarize_image_all_methods("path/to/ring.jpg", "compare/")
    binarize_dir_all_methods("imgs/", "compare/")
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np
import torch

from .utils import initPath

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_BLUR_TYPES = frozenset({"Gaussian", "Median", "NBF", "Bilateral"})
#: Every method :class:`Binarizer` knows, in the order the ``*_all_methods``
#: functions run them. ``"Mean"``/``"Gaussian"`` are OpenCV's adaptive
#: thresholds; ``"GlobalMean"`` and ``"Local"`` are scikit-image's
#: ``threshold_mean`` / ``threshold_local``.
BI_METHODS = (
    "Otsu", "Isodata", "Li", "Yen", "Minimum", "GlobalMean", "Triangle", "MultiOtsu",
    "Mean", "Gaussian", "Local", "Niblack", "Sauvola", "Hysteresis",
    "UNET", "RF",
)
#: Methods that need a trained model file (``model_path``).
LEARNED_METHODS = frozenset({"UNET", "RF"})

#: :class:`Binarizer` arguments of :meth:`Binarizer.preprocess` — used by
#: every method except the learned ones.
PREPROCESS_PARAMETERS = ("scale", "blur", "ksize", "gamma_equalisation")
#: Further :class:`Binarizer` arguments each method uses (see :func:`method_parameters`).
_OWN_PARAMETERS = {
    "Mean": ("adaptive_block_size", "adaptive_c"),
    "Gaussian": ("adaptive_block_size", "adaptive_c"),
    "MultiOtsu": ("multiotsu_classes",),
    "Local": ("local_block_size", "local_method", "local_offset"),
    "Niblack": ("niblack_window", "niblack_k"),
    "Sauvola": ("sauvola_window", "sauvola_k"),
    "Hysteresis": ("hysteresis_low", "hysteresis_high"),
    "UNET": ("scale", "unet_tile_size", "unet_input_size", "unet_overlap", "unet_blur", "unet_blur_ksize",
             "min_lumen_area", "max_hole_area"),
    "RF": ("scale",),
}


def method_parameters(method: str) -> tuple[str, ...]:
    """The :class:`Binarizer` arguments that affect *method* (besides ``model_path``)."""
    own = _OWN_PARAMETERS.get(method, ())
    return own if method in LEARNED_METHODS else PREPROCESS_PARAMETERS + own
_BI_METHODS = frozenset(BI_METHODS)

#: ``"UNET"`` tile side when ``unet_tile_size`` is not given. Independent of the
#: tile size the model was trained on: the network is fully convolutional, and
#: on PiSy 512 px tiles beat 128 px ones both in quality (fewer cells cut by a
#: tile border) and speed (less overlap per pixel).
UNET_TILE_SIZE = 512

#: Files the directory functions treat as images.
IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"})


# ---------------------------------------------------------------------------
# Public class
# ---------------------------------------------------------------------------

class Binarizer:
    """Convert micrograph images to binary (black/white) format.

    Parameters
    ----------
    method : str
        Thresholding method, one of :data:`BI_METHODS` (default ``"Otsu"``):

        * global — ``"Otsu"`` (OpenCV), ``"Isodata"``, ``"Li"``, ``"Yen"``,
          ``"Minimum"`` (fails on a histogram without two clear peaks),
          ``"GlobalMean"``, ``"Triangle"``, ``"MultiOtsu"`` (lumen = the
          brightest of *multiotsu_classes* classes);
        * local — ``"Mean"`` / ``"Gaussian"`` (OpenCV adaptive, 11×11
          window), ``"Local"`` (``skimage.filters.threshold_local``),
          ``"Niblack"``, ``"Sauvola"``;
        * ``"Hysteresis"`` — pixels above the high threshold are lumen, and
          so are pixels above the low one that connect to them;
        * learned — ``"UNET"``, ``"RF"`` (need *model_path*).
    blur : str
        Blurring method applied before thresholding (ignored by
        ``"RF"``, which does its own multi-scale filtering). One of
        ``"Gaussian"``, ``"Median"`` (default), ``"NBF"``,
        ``"Bilateral"``, or ``""`` / ``None`` for no blur.
    ksize : int
        Kernel size for blurring; must be odd (default: ``3``).
    gamma_equalisation : bool
        Apply histogram equalisation before blurring (default: ``False``).
    constant_th : int
        Fixed threshold used when *method* is not a named method
        (fallback path, default: ``235``).
    sauvola_window : int
        Local window size for Sauvola thresholding; must be odd
        (default: ``25``). Only used when ``method="Sauvola"``. Larger
        windows average over more neighbourhood context (smoother, less
        sensitive to per-pixel noise) but adapt more slowly to sharp
        local density changes, e.g. the earlywood/latewood boundary.
    sauvola_k : float
        Sauvola sensitivity constant (default: ``0.2``, the value from
        the original paper). Higher values push the threshold down
        (more pixels classified as foreground/wall) in high-variance
        neighbourhoods — raise it if latewood cell walls come out too
        thin/broken, lower it if lumens are getting swallowed.
    adaptive_block_size, adaptive_c :
        Window (odd, default ``11``) and constant subtracted from the
        local mean (default ``2``) of OpenCV's adaptive ``"Mean"`` /
        ``"Gaussian"`` thresholds.
    niblack_window, niblack_k :
        Window (odd) and constant of Niblack's ``T = mean − k·std``
        (defaults ``25``, ``0.2``). Only used when ``method="Niblack"``.
    local_block_size, local_method, local_offset :
        Arguments of ``skimage.filters.threshold_local``: odd window size
        (default ``35``), ``"gaussian"`` (default), ``"mean"`` or
        ``"median"``, and a constant subtracted from the local threshold
        (default ``0``). Only used when ``method="Local"``.
    multiotsu_classes : int
        Number of classes for ``"MultiOtsu"`` (default ``3``); the
        brightest one is the lumen.
    hysteresis_low, hysteresis_high : float | None
        Thresholds for ``"Hysteresis"``. ``None`` (default) takes both
        from a three-class multi-Otsu split of the image.
    unet_tile_size, unet_input_size : int | None
        ``"UNET"``: side of the tiles the (scaled) image is cut into, and
        the side each tile is resized to for the network. ``None``
        (default): tiles of :data:`UNET_TILE_SIZE` (512) px, fed to the
        network with the input/tile ratio the model was trained with
        (``unet.train.model_metadata``; 1, i.e. no resizing, for models
        without metadata), so cells keep the size in pixels they had in
        training.
    unet_overlap : int
        ``"UNET"``: overlap of neighbouring tiles in px (default ``32``);
        removes seams at tile borders. Must be smaller than the tile.
    unet_blur, unet_blur_ksize : str | None, int | None
        ``"UNET"``: smoothing of the image before the network -
        ``"median"``, ``"gaussian"``, ``"bilateral"`` or ``"none"``
        (``unet.utils.apply_blur``) - and its odd kernel in px. ``None``
        (default): what the model was trained with (``picdens_meta["blur"]``
        / ``["blur_ksize"]``; a 5 px median for grayscale models without it,
        none for RGB); a kernel left ``None`` with a type given is the
        model's if the type is the same, else 5. A blur other than in
        training is logged as a warning.
    min_lumen_area, max_hole_area : int
        ``"UNET"`` post-processing: lumens smaller than *min_lumen_area* px
        (default ``10``) are removed and holes in lumens smaller than
        *max_hole_area* px (default ``30``) are filled; ``0`` turns either off.
    model_path : str | None
        Path to a saved PyTorch U-Net model (``method="UNET"``) or a
        random-forest model from
        :func:`~openPicDens.rf_binarization.train_rf_model`
        (``method="RF"``). Required for both (default: ``None``).
    scale : float | int
        Scaling factor applied to images before processing
        (default: ``1.0``; ``0.5`` compresses the image to half its size).
        For ``"UNET"`` it should match the scale the model was trained with
        (a mismatch is logged); the mask is returned at the source
        resolution. For ``"RF"`` it instead multiplies the
        model's working scale — use it to bring cells of another
        species/magnification to the size the model was trained on
        (e.g. ``2.0`` if cells are half as large in pixels); the mask is
        still returned at the source resolution.

    Raises
    ------
    ValueError
        If ``method`` is ``"UNET"`` or ``"RF"`` but *model_path* is ``None``.
    """

    def __init__(
        self,
        method: str = "Otsu",
        blur: str = "Median",
        ksize: int = 3,
        gamma_equalisation: bool = False,
        constant_th: int = 235,
        sauvola_window: int = 25,
        sauvola_k: float = 0.2,
        adaptive_block_size: int = 11,
        adaptive_c: float = 2,
        niblack_window: int = 25,
        niblack_k: float = 0.2,
        local_block_size: int = 35,
        local_method: str = "gaussian",
        local_offset: float = 0.0,
        multiotsu_classes: int = 3,
        hysteresis_low: float | None = None,
        hysteresis_high: float | None = None,
        unet_tile_size: int | None = None,
        unet_input_size: int | None = None,
        unet_overlap: int = 32,
        min_lumen_area: int = 10,
        max_hole_area: int = 30,
        unet_blur: str | None = None,
        unet_blur_ksize: int | None = None,
        model_path: str | None = None,
        scale: float | int = 1.0,
    ) -> None:
        if method in ("UNET", "RF") and model_path is None:
            raise ValueError(f"model_path is required when method={method!r}")

        self.method = method
        self.blur = blur
        self.ksize = ksize
        self.gamma_equalisation = gamma_equalisation
        self.constant_th = constant_th
        self.sauvola_window = sauvola_window
        self.sauvola_k = sauvola_k
        self.adaptive_block_size = adaptive_block_size
        self.adaptive_c = adaptive_c
        self.niblack_window = niblack_window
        self.niblack_k = niblack_k
        self.local_block_size = local_block_size
        self.local_method = local_method
        self.local_offset = local_offset
        self.multiotsu_classes = multiotsu_classes
        self.hysteresis_low = hysteresis_low
        self.hysteresis_high = hysteresis_high
        self.unet_tile_size = unet_tile_size
        self.unet_input_size = unet_input_size
        self.unet_overlap = unet_overlap
        self.min_lumen_area = min_lumen_area
        self.max_hole_area = max_hole_area
        self.unet_blur = unet_blur
        self.unet_blur_ksize = unet_blur_ksize
        self.scale = scale
        self.model_path = model_path
        self._model: object | None = None

        if method == "UNET":
            device = "cuda" if torch.cuda.is_available() else "cpu"
            # map_location: a model saved from a GPU still loads on a machine without one (and vice
            # versa). weights_only=False: these .pth files hold the whole pickled model, which
            # torch >= 2.6 refuses to load by default.
            self._model = torch.load(model_path, map_location=device, weights_only=False).to(device)
            gpu = f", {torch.cuda.get_device_name(0)}" if device == "cuda" else ""
            logger.info(
                "UNet: устройство %s%s (torch %s, CUDA в сборке: %s)",
                device, gpu, torch.__version__, torch.version.cuda or "нет",
            )
            self._resolve_unet_tiling()
        elif method == "RF":
            from .rf_binarization import RFModel
            self._model = RFModel.load(model_path)

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
        return self.binarize_image(_read_image(img_path), img_path)

    def binarize_image(self, img: np.ndarray, img_path: str) -> np.ndarray:
        """Like :meth:`binarize` for an image already read from *img_path*.

        *img_path* is still needed by ``"UNET"``, which reads the file itself.
        """
        if self.method == "RF":
            # The classifier was trained on raw grayscale; blur/equalisation
            # would shift its features away from the training distribution.
            # `scale` is folded into the model's own working scale so the
            # mask comes back at the source resolution.
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
            return self._model.binarize(gray, scale=self.scale)
        if self.method == "UNET":
            # predictImgMask reads and scales the file itself.
            return self._apply_threshold(img, img_path)
        return self._apply_threshold(self.preprocess(img), img_path)

    def _resolve_unet_tiling(self) -> None:
        """Fill ``unet_tile_size`` / ``unet_input_size`` left as ``None``: tiles
        of :data:`UNET_TILE_SIZE` px, resized for the network by the same
        input/tile ratio as in training (``unet.train.model_metadata``; models
        without metadata were trained on unresized 128 px tiles, ratio 1)."""
        meta = dict(getattr(self._model, "picdens_meta", None) or {})
        if meta:
            logger.info("UNet: модель обучена с параметрами %s", meta)
        # 1 - grayscale, 3 - RGB (unet.utils.preprocess_image)
        self.unet_in_channels = int(meta.get("in_channels", 1))
        from unet.utils import blur_text, model_blur, normalize_blur
        trained = model_blur(self._model, self.unet_in_channels)
        if self.unet_blur is None:
            self.unet_blur, self.unet_blur_ksize = trained
        else:
            if self.unet_blur_ksize is None:  # the size from the model when the type is the same
                self.unet_blur_ksize = trained[1] if self.unet_blur == trained[0] else 5
            self.unet_blur, self.unet_blur_ksize = normalize_blur(self.unet_blur, self.unet_blur_ksize)
            if (self.unet_blur, self.unet_blur_ksize) != trained:
                logger.warning(
                    "UNet: модель обучалась со сглаживанием «%s», а бинаризация идёт с «%s» - снимок выглядит "
                    "иначе, чем при обучении", blur_text(*trained), blur_text(self.unet_blur, self.unet_blur_ksize),
                )
        train_tile = int(meta.get("tile_size") or meta.get("input_size") or 128)
        train_input = int(meta.get("input_size") or train_tile)
        ratio = train_input / train_tile
        if self.unet_tile_size is None:
            self.unet_tile_size = UNET_TILE_SIZE
        if self.unet_input_size is None:
            # a multiple of 32, which every smp encoder accepts
            self.unet_input_size = max(32, round(self.unet_tile_size * ratio / 32) * 32)
        logger.info(
            "UNet: плитка %d px, вход сети %d px (при обучении %d -> %d px), сглаживание %s",
            self.unet_tile_size, self.unet_input_size, train_tile, train_input,
            blur_text(self.unet_blur, self.unet_blur_ksize),
        )
        if not 0 <= self.unet_overlap < self.unet_tile_size:
            raise ValueError(
                f"перекрытие плиток ({self.unet_overlap}) должно быть меньше размера плитки "
                f"({self.unet_tile_size})"
            )
        trained_scale = meta.get("scale")
        if trained_scale is not None and abs(float(trained_scale) - self.scale) > 1e-6:
            logger.warning(
                "UNet: модель обучалась на снимках с масштабом %s, а бинаризация идёт с масштабом %s "
                "- клетки будут другого размера, чем при обучении", trained_scale, self.scale,
            )

    def describe(self) -> str:
        """The settings that affect this binarizer's result, e.g.
        ``"blur='Median', ksize=3, sauvola_window=25, sauvola_k=0.2"``."""
        parts = [f"{name}={getattr(self, name)!r}" for name in method_parameters(self.method)]
        if self.model_path is not None:
            parts.append(f"model={os.path.basename(self.model_path)}")
        return ", ".join(parts)

    def preprocess_key(self) -> tuple:
        """Everything :meth:`preprocess` depends on: binarizers with equal keys
        can share one preprocessed image."""
        return (self.scale, self.blur or "", self.ksize, self.gamma_equalisation)

    def preprocess(self, img: np.ndarray) -> np.ndarray:
        """Scale, convert to grayscale, equalise and blur *img* — everything
        a thresholding method sees. Depends only on the preprocessing
        parameters (:meth:`preprocess_key`), not on *method*.
        """
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
        return self._apply_blur(img)

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
        sub_dirs = sorted(os.listdir(src_root))
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
            adaptive = (
                cv2.ADAPTIVE_THRESH_MEAN_C if self.method == "Mean"
                else cv2.ADAPTIVE_THRESH_GAUSSIAN_C
            )
            return cv2.adaptiveThreshold(
                img, 255, adaptive, cv2.THRESH_BINARY,
                self.adaptive_block_size | 1, self.adaptive_c,
            )

        from skimage import filters as skfilters

        global_thresholds = {
            "Isodata": skfilters.threshold_isodata,
            "Li": skfilters.threshold_li,
            "Yen": skfilters.threshold_yen,
            "Minimum": skfilters.threshold_minimum,
            "GlobalMean": skfilters.threshold_mean,
            "Triangle": skfilters.threshold_triangle,
        }
        if self.method in global_thresholds:
            return _to_mask(img > global_thresholds[self.method](img))

        if self.method == "MultiOtsu":
            thresholds = skfilters.threshold_multiotsu(img, classes=self.multiotsu_classes)
            return _to_mask(img > thresholds[-1])

        if self.method == "Local":
            thresh = skfilters.threshold_local(
                img, block_size=self.local_block_size | 1,
                method=self.local_method, offset=self.local_offset,
            )
            return _to_mask(img > thresh)

        if self.method == "Niblack":
            thresh = skfilters.threshold_niblack(
                img, window_size=self.niblack_window | 1, k=self.niblack_k
            )
            return _to_mask(img > thresh)

        if self.method == "Sauvola":
            thresh = skfilters.threshold_sauvola(
                img, window_size=self.sauvola_window, k=self.sauvola_k
            )
            return _to_mask(img > thresh)

        if self.method == "Hysteresis":
            low, high = self.hysteresis_low, self.hysteresis_high
            if low is None or high is None:
                auto_low, auto_high = skfilters.threshold_multiotsu(img, classes=3)
                low = auto_low if low is None else low
                high = auto_high if high is None else high
            return _to_mask(skfilters.apply_hysteresis_threshold(img, low, high))

        if self.method == "UNET":
            from unet.utils import predictImgMask
            bi = predictImgMask(
                img_path=img_path,
                save_mask_path=None,
                divide_size=self.unet_tile_size,
                model=self._model,
                scale=self.scale,
                overlap=self.unet_overlap,
                input_size=self.unet_input_size,
                in_channels=self.unet_in_channels,
                blur=self.unet_blur,
                blur_ksize=self.unet_blur_ksize,
            )
            bi = cv2.cvtColor(bi, cv2.COLOR_RGB2GRAY)
            _, bi = cv2.threshold(
                bi, 200, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU
            )
            if bi.shape[:2] != img.shape[:2]:
                # scaling back by 1/scale can miss the source size by a pixel
                bi = cv2.resize(bi, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
            return remove_specks(bi, self.min_lumen_area, self.max_hole_area)

        # Fallback: fixed threshold
        _, bi = cv2.threshold(
            img, self.constant_th, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU
        )
        return bi


# ---------------------------------------------------------------------------
# Every method at once
# ---------------------------------------------------------------------------

def binarize_image_all_methods(
    img_path: str | os.PathLike,
    dst_dir: str | os.PathLike,
    methods: list[str] | tuple[str, ...] | None = None,
    unet_model_path: str | None = None,
    rf_model_path: str | None = None,
    method_kwargs: dict[str, dict] | None = None,
    **binarizer_kwargs,
) -> dict[str, str]:
    """Binarise one image with every method, writing ``dst_dir/<method>/<image name>``.

    Parameters
    ----------
    img_path : str | os.PathLike
        Source image.
    dst_dir : str | os.PathLike
        Root of the results; one subdirectory per method is created.
    methods : sequence of str | None
        Methods to run (default: all of :data:`BI_METHODS`).
    unet_model_path, rf_model_path : str | None
        Models for ``"UNET"`` / ``"RF"``. A learned method without its
        model is skipped (logged), as is any method that fails on this
        image (e.g. ``"Minimum"`` on a histogram without two peaks).
    method_kwargs : dict[str, dict] | None
        Per-method :class:`Binarizer` arguments, e.g.
        ``{"Sauvola": {"sauvola_window": 51}, "Otsu": {"blur": ""}}``; they
        override *binarizer_kwargs* for that method.
    **binarizer_kwargs
        Any other :class:`Binarizer` argument (blur, ksize, sauvola_k, ...),
        shared by every method.

    Returns
    -------
    dict[str, str]
        Path of the written mask for every method that succeeded.
    """
    binarizers = _make_binarizers(methods, unet_model_path, rf_model_path, binarizer_kwargs, method_kwargs)
    img_path = Path(img_path)
    written, failed = _binarize_with_all(img_path, Path(dst_dir), Path(img_path.name), binarizers)
    logger.info(
        "%s: бинаризовано методами %d/%d%s", img_path, len(written), len(binarizers),
        f", ошибки: {', '.join(failed)}" if failed else "",
    )
    return written


def binarize_dir_all_methods(
    src_dir: str | os.PathLike,
    dst_dir: str | os.PathLike,
    methods: list[str] | tuple[str, ...] | None = None,
    unet_model_path: str | None = None,
    rf_model_path: str | None = None,
    recursive: bool = True,
    progress: Callable[[int, int, Path], bool | None] | None = None,
    method_kwargs: dict[str, dict] | None = None,
    **binarizer_kwargs,
) -> dict[str, list[str]]:
    """Binarise every image in *src_dir* with every method.

    The result of a method goes to ``dst_dir/<method>/``, keeping each
    image's path relative to *src_dir* — so a project layout
    ``src_dir/<tree>/<year>.jpg`` becomes ``dst_dir/<method>/<tree>/<year>.jpg``
    and each method subdirectory can be used as a binarized-images folder
    as is. Models are loaded once for the whole directory, and each image is
    read once and preprocessed once per distinct preprocessing setting.

    Parameters
    ----------
    src_dir : str | os.PathLike
        Directory with source images (:data:`IMAGE_EXTENSIONS`).
    dst_dir : str | os.PathLike
        Root of the results. If it lies inside *src_dir*, its files are
        not treated as sources.
    recursive : bool
        Also process subdirectories (default: ``True``).
    progress : callable | None
        Called before each image as ``progress(done, total, path)``; a
        truthy return value stops the run (the images done so far are kept).
    methods, unet_model_path, rf_model_path, method_kwargs, **binarizer_kwargs :
        As in :func:`binarize_image_all_methods`.

    Returns
    -------
    dict[str, list[str]]
        Written mask paths per method. Failures (per image and method) are
        logged and do not stop the run.
    """
    src_dir, dst_dir = Path(src_dir), Path(dst_dir)
    images = list_images(src_dir, recursive=recursive, exclude=dst_dir)
    if not images:
        logger.warning("%s: снимков не найдено", src_dir)
        return {}

    binarizers = _make_binarizers(methods, unet_model_path, rf_model_path, binarizer_kwargs, method_kwargs)
    results: dict[str, list[str]] = {method: [] for method in binarizers}
    n_failed = 0
    for i, img_path in enumerate(images, 1):
        if progress is not None and progress(i - 1, len(images), img_path):
            logger.info("%s: остановлено после %d/%d снимков", src_dir, i - 1, len(images))
            break
        logger.info("Бинаризация всеми методами %d/%d: %s", i, len(images), img_path)
        try:
            written, failed = _binarize_with_all(
                img_path, dst_dir, img_path.relative_to(src_dir), binarizers
            )
        except Exception:
            logger.exception("%s: не удалось прочитать", img_path)
            n_failed += len(binarizers)
            continue
        for method, path in written.items():
            results[method].append(path)
        n_failed += len(failed)

    logger.info(
        "%s: %d снимков × %d методов, ошибок: %d", src_dir, len(images), len(binarizers), n_failed
    )
    return results


def list_images(
    src_dir: str | os.PathLike, recursive: bool = True, exclude: str | os.PathLike | None = None,
) -> list[Path]:
    """Sorted image files (:data:`IMAGE_EXTENSIONS`) in *src_dir*, skipping
    anything inside *exclude* (e.g. an output folder placed in *src_dir*)."""
    src_dir = Path(src_dir)
    if not src_dir.is_dir():
        raise NotADirectoryError(f"Нет такой папки: {src_dir}")
    excluded = Path(exclude).resolve() if exclude is not None else None
    pattern = "**/*" if recursive else "*"
    return sorted(
        p for p in src_dir.glob(pattern)
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        and (excluded is None or excluded not in p.resolve().parents)
    )


def _make_binarizers(
    methods: list[str] | tuple[str, ...] | None,
    unet_model_path: str | None,
    rf_model_path: str | None,
    binarizer_kwargs: dict,
    method_kwargs: dict[str, dict] | None = None,
) -> dict[str, Binarizer]:
    """One :class:`Binarizer` per method that can run (models loaded once);
    *method_kwargs* overrides *binarizer_kwargs* per method."""
    method_kwargs = method_kwargs or {}
    for kwargs in (binarizer_kwargs, *method_kwargs.values()):
        if "method" in kwargs or "model_path" in kwargs:
            raise TypeError("method/model_path are chosen per method; use unet_model_path/rf_model_path")
    methods = BI_METHODS if methods is None else tuple(methods)
    unknown = [m for m in methods if m not in _BI_METHODS]
    if unknown:
        raise ValueError(f"unknown binarization method(s): {unknown}; known: {BI_METHODS}")
    model_paths = {"UNET": unet_model_path, "RF": rf_model_path}

    binarizers: dict[str, Binarizer] = {}
    for method in methods:
        model_path = model_paths.get(method)
        if method in LEARNED_METHODS and model_path is None:
            logger.info("%s пропущен: не задан файл модели", method)
            continue
        try:
            kwargs = {**binarizer_kwargs, **method_kwargs.get(method, {})}
            binarizers[method] = Binarizer(method=method, model_path=model_path, **kwargs)
        except Exception:
            logger.exception("%s пропущен: не удалось создать бинаризатор", method)
    if not binarizers:
        raise ValueError("no binarization method can run")
    return binarizers


def _binarize_with_all(
    img_path: Path, dst_root: Path, rel_path: Path, binarizers: dict[str, Binarizer],
    on_mask: Callable[[str, np.ndarray, float], None] | None = None,
) -> tuple[dict[str, str], list[str]]:
    """Run every binarizer on *img_path*, writing ``dst_root/<method>/<rel_path>``.

    *on_mask* (if given) gets every successful ``(method, mask, seconds)``
    before it is written. Returns the written paths and the methods that
    failed. Reading the image is the only error that propagates.
    """
    img = _read_image(str(img_path))
    # Thresholding methods with the same preprocessing settings share one
    # preprocessed copy (its time is counted to the first of them).
    prepared: dict[tuple, np.ndarray] = {}
    written: dict[str, str] = {}
    failed: list[str] = []
    for method, binarizer in binarizers.items():
        try:
            started = time.perf_counter()
            if method in LEARNED_METHODS:
                mask = binarizer.binarize_image(img, str(img_path))
            else:
                key = binarizer.preprocess_key()
                if key not in prepared:
                    prepared[key] = binarizer.preprocess(img)
                mask = binarizer._apply_threshold(prepared[key], str(img_path))
            if on_mask is not None:
                on_mask(method, mask, time.perf_counter() - started)
            dst_path = dst_root / method / rel_path
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(dst_path), mask):
                raise OSError(f"не удалось записать {dst_path}")
            written[method] = str(dst_path)
        except Exception as exc:
            logger.warning("%s: метод %s не сработал: %s", img_path, method, exc)
            failed.append(method)
    return written, failed


def _read_image(img_path: str) -> np.ndarray:
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Не удалось прочитать изображение: {img_path}")
    return img


def _to_mask(lumen: np.ndarray) -> np.ndarray:
    """Boolean lumen array → ``uint8`` image, 255 = lumen."""
    return lumen.astype(np.uint8) * 255


def remove_specks(mask: np.ndarray, min_area: int = 10, max_hole: int = 30) -> np.ndarray:
    """Remove lumens smaller than *min_area* px and fill holes in lumens smaller
    than *max_hole* px (4-connectivity; ``0`` skips a step).

    *mask* is a ``uint8`` image, 255 (``> 127``) = lumen; returns the same.
    """
    lumen = mask > 127
    if min_area > 0:
        _, labels, stats, _ = cv2.connectedComponentsWithStats(lumen.astype(np.uint8), connectivity=4)
        keep = stats[:, cv2.CC_STAT_AREA] >= min_area
        keep[0] = False
        lumen = keep[labels]
    if max_hole > 0:
        _, labels, stats, _ = cv2.connectedComponentsWithStats((~lumen).astype(np.uint8), connectivity=4)
        hole = stats[:, cv2.CC_STAT_AREA] < max_hole
        hole[0] = False
        lumen |= hole[labels]
    return _to_mask(lumen)