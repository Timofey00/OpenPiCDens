"""
Comparing binarization methods against reference (ground-truth) masks.

:func:`run_binarization_test` binarises one image or a directory with
every chosen method (:func:`openPicDens.binarization.binarize_dir_all_methods`
layout: ``out_dir/<method>/<image path>``), compares each result with the
reference mask of the same image and writes per-image and per-method
statistics.

Pixels are compared with lumen (white, ``> 127``) as the positive class:

* ``accuracy`` — share of pixels classified as in the reference;
* ``precision`` — share of predicted lumen that is lumen in the reference;
* ``recall`` (sensitivity) — share of reference lumen that was found;
* ``specificity`` — the same for the wall;
* ``f1`` (= Dice) and ``iou`` (= Jaccard) — overlap of the two lumen areas;
* ``balanced_accuracy`` — mean of recall and specificity;
* ``mcc`` — Matthews correlation, robust to an unbalanced lumen/wall ratio;
* ``lumen_frac_*`` — porosity of the whole image (lumen share), predicted,
  reference and their difference;
* ``n_cells_*`` — connected lumen components of at least ``min_cell_area``
  pixels; ``cells_ratio`` = predicted / reference (merged lumens push it
  below 1, broken walls or noise above 1).

The summary table has the mean of every metric over the images per method
plus ``f1_pooled`` / ``iou_pooled`` from the pixel counts summed over all
images (large images weigh more), sorted by mean F1.
"""

from __future__ import annotations

import logging
import math
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from .binarization import _binarize_with_all, _make_binarizers, list_images

logger = logging.getLogger(__name__)

#: Metrics in the order they appear in the tables (after the identifying columns).
METRIC_COLUMNS = (
    "f1", "iou", "precision", "recall", "specificity", "accuracy", "balanced_accuracy", "mcc",
    "lumen_frac_pred", "lumen_frac_ref", "lumen_frac_diff",
    "n_cells_pred", "n_cells_ref", "cells_ratio",
)
_COUNT_COLUMNS = ("tp", "fp", "fn", "tn")


# ---------------------------------------------------------------------------
# Metrics for one pair of masks
# ---------------------------------------------------------------------------

def mask_metrics(pred: np.ndarray, ref: np.ndarray, min_cell_area: int = 10) -> dict[str, float]:
    """Pixel and cell-count agreement of binary *pred* with reference *ref*.

    Both are grayscale images, white (``> 127``) = lumen. A *pred* of
    another size is resized to *ref* (nearest neighbour) and the result gets
    ``resized = True``. Returns the counts (``tp``/``fp``/``fn``/``tn``) and
    :data:`METRIC_COLUMNS`; a ratio with a zero denominator is NaN.
    """
    if pred.ndim == 3:
        pred = cv2.cvtColor(pred, cv2.COLOR_BGR2GRAY)
    if ref.ndim == 3:
        ref = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
    resized = pred.shape != ref.shape
    if resized:
        pred = cv2.resize(pred, (ref.shape[1], ref.shape[0]), interpolation=cv2.INTER_NEAREST)
    p, r = pred > 127, ref > 127

    tp = int(np.count_nonzero(p & r))
    fp = int(np.count_nonzero(p & ~r))
    fn = int(np.count_nonzero(~p & r))
    tn = int(p.size - tp - fp - fn)
    n_pred, n_ref = _count_cells(p, min_cell_area), _count_cells(r, min_cell_area)

    metrics = dict(tp=tp, fp=fp, fn=fn, tn=tn, **_ratios(tp, fp, fn, tn))
    metrics.update(
        lumen_frac_pred=(tp + fp) / p.size,
        lumen_frac_ref=(tp + fn) / p.size,
        n_cells_pred=n_pred,
        n_cells_ref=n_ref,
        cells_ratio=_div(n_pred, n_ref),
        resized=resized,
    )
    metrics["lumen_frac_diff"] = metrics["lumen_frac_pred"] - metrics["lumen_frac_ref"]
    return metrics


def _ratios(tp: float, fp: float, fn: float, tn: float) -> dict[str, float]:
    recall, specificity = _div(tp, tp + fn), _div(tn, tn + fp)
    # float() before multiplying: the products of pixel counts of a large
    # image overflow int64 in the MCC denominator.
    mcc_den = math.sqrt(float(tp + fp) * float(tp + fn) * float(tn + fp) * float(tn + fn))
    return dict(
        f1=_div(2 * tp, 2 * tp + fp + fn),
        iou=_div(tp, tp + fp + fn),
        precision=_div(tp, tp + fp),
        recall=recall,
        specificity=specificity,
        accuracy=_div(tp + tn, tp + fp + fn + tn),
        balanced_accuracy=(recall + specificity) / 2,
        mcc=_div(float(tp) * tn - float(fp) * fn, mcc_den),
    )


def _div(a: float, b: float) -> float:
    return a / b if b else float("nan")


def _count_cells(lumen: np.ndarray, min_area: int) -> int:
    _, _, stats, _ = cv2.connectedComponentsWithStats(lumen.astype(np.uint8), connectivity=8)
    return int(np.count_nonzero(stats[1:, cv2.CC_STAT_AREA] >= min_area))


def load_reference(path: str | os.PathLike) -> np.ndarray | None:
    """Reference mask as ``uint8`` 0/255 (255 = lumen), ``None`` if unreadable.

    Binarised with Otsu rather than a fixed ``> 127``, so that a mask drawn
    in colour on black - e.g. lumens filled with one annotation colour, which
    is only ~114 in grayscale - reads as "coloured = lumen", the same way
    ``unet.utils.cutAndPadImg`` reads masks for U-Net training. A plain
    black/white mask (with or without JPEG noise) gives the same result as
    ``> 127``.
    """
    gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        return None
    levels = np.unique(gray)
    if not np.isin(levels, (0, 255)).all() and len(levels) <= 16:
        logger.info(
            "%s: эталон не чёрно-белый (уровни яркости %s) - просветом считается всё, что "
            "светлее фона (порог Отсу)", path, levels.tolist(),
        )
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return binary


# ---------------------------------------------------------------------------
# Finding the reference of an image
# ---------------------------------------------------------------------------

def find_reference(
    rel_path: Path, ref_dir: Path, index: dict[str, list[Path]],
) -> Path | None:
    """Reference file for the image at *rel_path* (relative to its source root).

    Same stem (case-insensitive, any image extension) in the same relative
    subfolder of *ref_dir* first, otherwise the only file with that stem
    anywhere in *ref_dir*. ``None`` if there is none or the stem is ambiguous.
    *index* comes from :func:`build_stem_index`.
    """
    candidates = index.get(rel_path.stem.lower(), [])
    same_folder = [c for c in candidates if c.parent == ref_dir / rel_path.parent]
    if same_folder:
        return same_folder[0]
    if len(candidates) == 1:
        return candidates[0]
    if candidates:
        logger.warning(
            "%s: эталон неоднозначен (%s)", rel_path, ", ".join(str(c) for c in candidates)
        )
    return None


def build_stem_index(
    ref_dir: str | os.PathLike, exclude: str | os.PathLike | None = None,
) -> dict[str, list[Path]]:
    """Image files under *ref_dir* (recursively, nothing inside *exclude*) by lower-case stem."""
    index: dict[str, list[Path]] = {}
    for path in list_images(ref_dir, recursive=True, exclude=exclude):
        index.setdefault(path.stem.lower(), []).append(path)
    return index


def pair_with_references(
    src: str | os.PathLike, ref_dir: str | os.PathLike, recursive: bool = True,
    exclude: str | os.PathLike | None = None,
) -> tuple[list[tuple[Path, Path, Path]], list[Path]]:
    """Source images of *src* (a file or a directory) with their references.

    *exclude* (the test's output folder) is left out of both the sources and the references, so
    masks written by an earlier run inside *ref_dir* never count as references. An image whose
    only "reference" is the image file itself (*ref_dir* is the source folder) has none.

    Returns ``[(image, path relative to src, reference), ...]`` and the
    relative paths of images that have no reference.
    """
    src, ref_dir = Path(src), Path(ref_dir)
    if src.is_file():
        images, root = [src], src.parent
    else:
        images, root = list_images(src, recursive=recursive, exclude=exclude), src
    index = build_stem_index(ref_dir, exclude=exclude)
    pairs, missing = [], []
    n_self = 0
    for image in images:
        rel = image.relative_to(root)
        ref = find_reference(rel, ref_dir, index)
        if ref is not None and ref.resolve() == image.resolve():
            n_self += 1
            ref = None
        if ref is None:
            missing.append(rel)
        else:
            pairs.append((image, rel, ref))
    if n_self:
        logger.warning(
            "%d снимков нашли в %s самих себя вместо эталона - папка эталонов совпадает с папкой "
            "снимков? Они пропущены", n_self, ref_dir,
        )
    return pairs, missing


# ---------------------------------------------------------------------------
# The whole test
# ---------------------------------------------------------------------------

@dataclass
class BinarizationTestResult:
    """Outcome of :func:`run_binarization_test`."""

    per_image: pd.DataFrame
    summary: pd.DataFrame
    out_dir: Path
    files: list[Path] = field(default_factory=list)
    missing_references: list[Path] = field(default_factory=list)
    failures: list[tuple[str, str]] = field(default_factory=list)  # (image, method)
    cancelled: bool = False


def run_binarization_test(
    src: str | os.PathLike,
    ref_dir: str | os.PathLike,
    out_dir: str | os.PathLike,
    methods: list[str] | tuple[str, ...] | None = None,
    unet_model_path: str | None = None,
    rf_model_path: str | None = None,
    recursive: bool = True,
    min_cell_area: int = 10,
    progress: Callable[[int, int, str], bool | None] | None = None,
    method_kwargs: dict[str, dict] | None = None,
    **binarizer_kwargs,
) -> BinarizationTestResult:
    """Binarise *src* with every method and compare the masks with references.

    Parameters
    ----------
    src : str | os.PathLike
        One image or a directory of images.
    ref_dir : str | os.PathLike
        Reference binarized images, matched to the sources by
        :func:`find_reference`. Images without a reference are skipped.
    out_dir : str | os.PathLike
        Masks go to ``out_dir/<method>/<image path>``; the tables
        ``metrics_per_image`` / ``metrics_summary`` (``.tsv`` and ``.xlsx``)
        and the chart ``metrics_summary.png`` to *out_dir* itself. Both
        tables have a ``settings`` column with each method's parameters.
    methods, unet_model_path, rf_model_path, method_kwargs, **binarizer_kwargs :
        As in :func:`~openPicDens.binarization.binarize_image_all_methods`
        (*method_kwargs*: per-method settings).
    recursive : bool
        Include subdirectories of a source directory (default ``True``).
    min_cell_area : int
        Smallest lumen component counted as a cell (px, default ``10``).
    progress : callable | None
        ``progress(done, total, message)`` before each image; a truthy
        return value stops the test (the images done so far are evaluated).

    Raises
    ------
    ValueError
        If no source image has a reference, or no method can run.
    """
    out_dir = Path(out_dir)
    pairs, missing = pair_with_references(src, ref_dir, recursive, exclude=out_dir)
    for rel in missing:
        logger.warning("%s: нет эталона в %s - пропущен", rel, ref_dir)
    if not pairs:
        src_dir = Path(src) if Path(src).is_dir() else Path(src).parent
        if src_dir.resolve() == Path(ref_dir).resolve():
            raise ValueError(f"Папка эталонов совпадает с папкой снимков: {ref_dir}")
        raise ValueError(f"Ни для одного снимка не найден эталон в {ref_dir}")
    binarizers = _make_binarizers(methods, unet_model_path, rf_model_path, binarizer_kwargs, method_kwargs)
    settings = {method: binarizer.describe() for method, binarizer in binarizers.items()}
    for method, text in settings.items():
        logger.info("Тест бинаризации: %s - %s", method, text or "без параметров")

    rows: list[dict] = []
    failures: list[tuple[str, str]] = []
    cancelled = False
    for i, (image, rel, ref_path) in enumerate(pairs):
        if progress is not None and progress(i, len(pairs), f"Тест бинаризации {i + 1}/{len(pairs)}: {rel}"):
            cancelled = True
            break
        ref = load_reference(ref_path)
        if ref is None:
            logger.error("%s: не удалось прочитать эталон %s", rel, ref_path)
            failures.extend((str(rel), m) for m in binarizers)
            continue

        def compare(method: str, mask: np.ndarray, seconds: float) -> None:
            metrics = mask_metrics(mask, ref, min_cell_area)
            if metrics["resized"]:
                logger.warning(
                    "%s, %s: размер маски %s не совпадает с эталоном %s - маска приведена к эталону",
                    rel, method, mask.shape[:2], ref.shape[:2],
                )
            rows.append(dict(image=str(rel), method=method, seconds=seconds,
                             reference=str(ref_path), settings=settings[method], **metrics))

        try:
            _, failed = _binarize_with_all(image, out_dir, rel, binarizers, on_mask=compare)
        except Exception:
            logger.exception("%s: не удалось прочитать снимок", image)
            failed = list(binarizers)
        failures.extend((str(rel), m) for m in failed)

    per_image = _per_image_table(rows)
    summary = _summary_table(per_image, list(binarizers))
    summary["settings"] = summary["method"].map(settings)
    files = _write_outputs(per_image, summary, out_dir)
    logger.info(
        "Тест бинаризации: %d снимков × %d методов, без эталона %d, ошибок %d%s",
        per_image["image"].nunique() if len(per_image) else 0, len(binarizers), len(missing),
        len(failures), ", прервано" if cancelled else "",
    )
    return BinarizationTestResult(
        per_image=per_image, summary=summary, out_dir=out_dir, files=files,
        missing_references=missing, failures=failures, cancelled=cancelled,
    )


def _per_image_table(rows: list[dict]) -> pd.DataFrame:
    columns = ["image", "method", *METRIC_COLUMNS, "seconds", *_COUNT_COLUMNS, "resized", "reference", "settings"]
    return pd.DataFrame(rows, columns=columns)


def _summary_table(per_image: pd.DataFrame, methods: list[str]) -> pd.DataFrame:
    """Per-method means of the metrics, pooled F1/IoU, sorted by mean F1."""
    records = []
    for method in methods:
        group = per_image[per_image["method"] == method]
        if group.empty:
            continue
        record = {"method": method, "n_images": len(group)}
        record.update({c: group[c].mean() for c in METRIC_COLUMNS})
        record["f1_std"] = group["f1"].std(ddof=0)
        pooled = _ratios(*(float(group[c].sum()) for c in _COUNT_COLUMNS))
        record["f1_pooled"], record["iou_pooled"] = pooled["f1"], pooled["iou"]
        record["seconds_per_image"] = group["seconds"].mean()
        records.append(record)
    columns = ["method", "n_images", "f1", "f1_std", "f1_pooled", "iou", "iou_pooled",
               *[c for c in METRIC_COLUMNS if c not in ("f1", "iou")], "seconds_per_image"]
    summary = pd.DataFrame(records, columns=columns)
    return summary.sort_values("f1", ascending=False, na_position="last").reset_index(drop=True)


def _write_outputs(per_image: pd.DataFrame, summary: pd.DataFrame, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    files = []
    for name, table in (("metrics_per_image", per_image), ("metrics_summary", summary)):
        tsv = out_dir / f"{name}.tsv"
        table.to_csv(tsv, sep="\t", index=False)
        files.append(tsv)
        try:
            table.to_excel(out_dir / f"{name}.xlsx", index=False)
            files.append(out_dir / f"{name}.xlsx")
        except Exception:  # Excel is a convenience; the TSV already has everything
            logger.exception("%s: не удалось записать .xlsx", name)
    if not summary.empty:
        chart = out_dir / "metrics_summary.png"
        _plot_summary(summary, chart)
        files.append(chart)
    return files


def _plot_summary(summary: pd.DataFrame, path: Path) -> None:
    """Horizontal bars of mean F1 and IoU per method, best on top."""
    # matplotlib.figure.Figure, not pyplot: this may run in a GUI worker thread.
    from matplotlib.figure import Figure

    data = summary.iloc[::-1]
    y = np.arange(len(data))
    fig = Figure(figsize=(8, 0.4 * len(data) + 1.5))
    ax = fig.add_subplot()
    ax.barh(y + 0.2, data["f1"], height=0.4, label="F1 (Dice)", color="#2a6fb0")
    ax.barh(y - 0.2, data["iou"], height=0.4, label="IoU", color="#e08a2c")
    ax.set_yticks(y, data["method"])
    ax.set_xlim(0, 1)
    ax.set_xlabel("Среднее по снимкам")
    ax.set_title("Сравнение методов бинаризации с эталоном")
    ax.grid(axis="x", alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)


__all__ = [
    "METRIC_COLUMNS", "BinarizationTestResult", "build_stem_index", "find_reference",
    "load_reference", "mask_metrics", "pair_with_references", "run_binarization_test",
]
