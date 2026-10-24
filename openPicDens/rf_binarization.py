"""
Random-forest pixel classifier for wood micrograph binarisation.

A lightweight alternative to the U-Net: every pixel is described by
multi-scale intensity / edge / texture features
(:func:`skimage.feature.multiscale_basic_features`) and classified as
lumen or cell wall by a :class:`~sklearn.ensemble.RandomForestClassifier`.
Unlike a global threshold (Otsu), the classifier sees the neighbourhood
of each pixel, so it can tell a bright lumen from the equally bright
interior of a thick latewood cell wall.

Training needs only image/mask pairs — existing U-Net predictions work
as masks, so no manual annotation is required for a species that
already has a model. For a new species, *sparse* masks (a few strokes
of lumen / wall, everything else left unlabelled) are enough.

Typical usage::

    from openPicDens.rf_binarization import train_rf_model

    train_rf_model(
        img_paths=["2003.jpg"], mask_paths=["2003_bi.jpg"],
        save_path="pine_rf.joblib",
    )

    bi = Binarizer(method="RF", model_path="pine_rf.joblib")
    binary = bi.binarize("2002.jpg")
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import cv2
import joblib
import numpy as np
from skimage.feature import multiscale_basic_features
from sklearn.ensemble import RandomForestClassifier

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Pixel value marking unlabelled pixels in a *sparse* training mask.
UNLABELLED = 128

_DEFAULT_FEATURES = dict(
    intensity=True, edges=True, texture=True, sigma_min=1, sigma_max=16,
)

_TILE = 1024  # prediction tile side (px, at working scale) — bounds memory
_MIN_TRAIN_SIDE = 32  # training images thinner than this (working scale) are skipped


# ---------------------------------------------------------------------------
# Model container
# ---------------------------------------------------------------------------

@dataclass
class RFModel:
    """Trained classifier plus everything needed to reproduce its input.

    Attributes
    ----------
    classifier : RandomForestClassifier
        Fitted pixel classifier (class ``1`` = lumen).
    work_scale : float
        Images are resized by this factor before feature extraction —
        must be the same at training and prediction time.
    feature_kwargs : dict
        Arguments for :func:`multiscale_basic_features`.
    prob_threshold : float
        Lumen probability above which a pixel is white.
    min_radius : float
        Lumen components whose inscribed-circle radius (px, at working
        scale) is below this are discarded — removes thin slivers of
        wall interior while keeping small round latewood lumens.
    min_area : int
        Lumen components smaller than this (px, at working scale) are
        discarded.
    """

    classifier: RandomForestClassifier
    work_scale: float = 0.5
    feature_kwargs: dict = field(default_factory=lambda: dict(_DEFAULT_FEATURES))
    prob_threshold: float = 0.5
    min_radius: float = 2.0
    min_area: int = 15

    def save(self, path: str) -> None:
        """Write the model to *path* (joblib format)."""
        joblib.dump(self, path, compress=3)

    @staticmethod
    def load(path: str) -> "RFModel":
        """Load a model written by :meth:`save`."""
        model = joblib.load(path)
        if not isinstance(model, RFModel):
            raise TypeError(f"{path!r} does not contain an RFModel")
        return model

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict_proba(self, gray: np.ndarray, scale: float = 1.0) -> np.ndarray:
        """Lumen probability map for *gray* at the working scale.

        *scale* multiplies :attr:`work_scale` (see :meth:`binarize`).
        """
        work = _resize(gray, self.work_scale * scale)
        prob = np.empty(work.shape, dtype=np.float32)
        # Tiles overlap by the reach of the largest Gaussian (skimage truncates
        # it at 4 sigma) so features near tile edges match the untiled result.
        margin = int(np.ceil(4 * self.feature_kwargs.get("sigma_max", 16))) + 1
        h, w = work.shape
        for y in range(0, h, _TILE):
            for x in range(0, w, _TILE):
                y0, x0 = max(y - margin, 0), max(x - margin, 0)
                y1, x1 = min(y + _TILE + margin, h), min(x + _TILE + margin, w)
                feats = _features(work[y0:y1, x0:x1], self.feature_kwargs)
                p = self.classifier.predict_proba(
                    feats.reshape(-1, feats.shape[-1])
                )[:, 1].reshape(feats.shape[:2])
                prob[y: y + _TILE, x: x + _TILE] = p[
                    y - y0: y - y0 + _TILE, x - x0: x - x0 + _TILE
                ]
        return prob

    def binarize(self, gray: np.ndarray, scale: float = 1.0) -> np.ndarray:
        """Binary image for *gray*: ``255`` = lumen, ``0`` = wall.

        *scale* multiplies :attr:`work_scale` — pass ``2.0`` for images
        whose cells are half as large in pixels as in the training data.
        Post-processing runs at the working scale; the result is resized
        back to the shape of *gray*.
        """
        prob = self.predict_proba(gray, scale)
        mask = filter_components(
            prob > self.prob_threshold, self.min_radius, self.min_area
        )
        mask = mask.astype(np.uint8) * 255
        if mask.shape != gray.shape:
            mask = cv2.resize(
                mask, (gray.shape[1], gray.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
        return mask


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_rf_model(
    img_paths: list[str],
    mask_paths: list[str],
    save_path: str | None = None,
    work_scale: float = 0.5,
    sparse: bool = False,
    n_samples: int = 300_000,
    n_estimators: int = 60,
    max_depth: int = 18,
    min_radius: float = 2.0,
    min_area: int = 15,
    feature_kwargs: dict | None = None,
    random_state: int = 0,
) -> RFModel:
    """Train a lumen/wall pixel classifier from image–mask pairs.

    Parameters
    ----------
    img_paths : list[str]
        Source micrographs.
    mask_paths : list[str]
        Masks matching *img_paths*, same size: white (``>= 200``) = lumen,
        black (``<= 55``) = wall. Dense masks (e.g. U-Net output) are
        thresholded with Otsu instead, which tolerates JPEG artefacts and
        reads a mask coloured on black as "coloured = lumen".
    save_path : str | None
        If given, the trained model is saved here (see :meth:`RFModel.save`).
    work_scale : float
        Resize factor applied before feature extraction (default: ``0.5``).
        Pick it so cells are roughly the size they are in the training
        images; ``0.5`` keeps pine latewood walls resolvable while making
        prediction ~4× cheaper than full resolution.
    sparse : bool
        If ``True``, only pixels near pure black / pure white are used
        for training and everything else (e.g. the value ``128``,
        :data:`UNLABELLED`) is ignored — draw a few strokes of lumen and
        wall on a grey canvas instead of labelling the whole image.
    n_samples : int
        Total training pixels, drawn balanced between the two classes
        and spread across images in proportion to their area (default:
        ``300_000``). Images narrower than 32 px at the working scale are
        skipped.
    n_estimators, max_depth : int
        Random-forest size. Fewer / shallower trees predict faster.
    min_radius, min_area :
        Post-processing parameters stored in the model, see :class:`RFModel`.
    feature_kwargs : dict | None
        Override arguments for :func:`multiscale_basic_features`.
    random_state : int
        Seed for pixel sampling and the forest.

    Returns
    -------
    RFModel
        The trained model.
    """
    if len(img_paths) != len(mask_paths):
        raise ValueError("img_paths and mask_paths must have the same length")
    if not img_paths:
        raise ValueError("at least one image/mask pair is required")

    fk = dict(_DEFAULT_FEATURES, **(feature_kwargs or {}))
    rng = np.random.default_rng(random_state)
    # Sample each image in proportion to its area, so a set of small tiles
    # and a few large scans are weighted by how much wood they show.
    # Image headers are enough for the sizes (no full decode).
    areas = [w * h for w, h in (_image_size(p) for p in img_paths)]
    sample_rate = n_samples / sum(areas)

    xs, ys = [], []
    t0 = time.perf_counter()
    for img_path, mask_path, area in zip(img_paths, mask_paths, areas):
        per_class = max(1, round(sample_rate * area / 2))
        gray = _read_gray(img_path)
        mask = _read_gray(mask_path)
        if mask.shape != gray.shape:
            raise ValueError(
                f"mask {mask_path!r} {mask.shape} does not match image "
                f"{img_path!r} {gray.shape}"
            )
        gray = _resize(gray, work_scale)
        if min(gray.shape) < _MIN_TRAIN_SIDE:
            # Features of a sliver this thin are dominated by border effects.
            logger.warning("skipping %s: %s px at working scale is too small",
                           img_path, gray.shape)
            continue
        mask = cv2.resize(
            mask, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_NEAREST
        )
        if sparse:
            lumen, wall = mask >= 200, mask <= 55
        else:
            # Otsu, not a fixed 127: a mask drawn in colour on black (one
            # annotation colour per lumen, ~114 in grayscale) reads as
            # "coloured = lumen", like the U-Net tiling does; a plain
            # black/white mask gives the same as > 127.
            th, _ = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            lumen = mask > th
            wall = ~lumen

        feats = _features(gray, fk)
        feats = feats.reshape(-1, feats.shape[-1])
        for label, cls_mask in ((1, lumen), (0, wall)):
            idx = np.flatnonzero(cls_mask.ravel())
            if idx.size == 0:
                logger.warning("%s has no %s pixels", mask_path,
                               "lumen" if label else "wall")
                continue
            idx = rng.choice(idx, min(per_class, idx.size), replace=False)
            xs.append(feats[idx])
            ys.append(np.full(idx.size, label, dtype=np.uint8))
        del feats

    y = np.concatenate(ys)
    if np.unique(y).size < 2:
        raise ValueError("training masks must contain both lumen and wall pixels")

    clf = RandomForestClassifier(
        n_estimators=n_estimators, max_depth=max_depth, min_samples_leaf=5,
        n_jobs=-1, random_state=random_state,
    ).fit(np.concatenate(xs), y)
    logger.info("RF trained on %d pixels from %d image(s) in %.1fs",
                y.size, len(img_paths), time.perf_counter() - t0)

    model = RFModel(
        classifier=clf, work_scale=work_scale, feature_kwargs=fk,
        min_radius=min_radius, min_area=min_area,
    )
    if save_path is not None:
        model.save(save_path)
    return model


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def filter_components(
    mask: np.ndarray, min_radius: float, min_area: int
) -> np.ndarray:
    """Drop lumen components that are too thin or too small.

    A component is kept if its largest inscribed circle has radius
    ``>= min_radius`` *and* its area is ``>= min_area``. Unlike a
    morphological opening, kept components retain their full shape.
    """
    mask = mask.astype(np.uint8)
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    max_dist = np.zeros(n, dtype=np.float32)
    np.maximum.at(max_dist, labels.ravel(), dist.ravel())
    keep = (max_dist >= min_radius) & (stats[:, cv2.CC_STAT_AREA] >= min_area)
    keep[0] = False
    return keep[labels]


def _features(gray: np.ndarray, feature_kwargs: dict) -> np.ndarray:
    return multiscale_basic_features(
        gray, channel_axis=None, **feature_kwargs
    ).astype(np.float32, copy=False)


def _resize(gray: np.ndarray, scale: float) -> np.ndarray:
    if scale == 1.0:
        return gray
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    return cv2.resize(gray, None, fx=scale, fy=scale, interpolation=interp)


def _image_size(path: str) -> tuple[int, int]:
    """``(width, height)`` of *path*, read from the file header only."""
    from PIL import Image
    try:
        with Image.open(path) as img:
            return img.size
    except FileNotFoundError:
        raise FileNotFoundError(path) from None


def _read_gray(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    return img
