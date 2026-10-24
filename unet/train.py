import inspect
import math
import os
import re
import time
from collections.abc import Callable, Sequence
from pathlib import Path

import cv2
import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F
from imutils import paths
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from unet.utils import _tile_starts, normalize_blur, preprocess_image

#: segmentation_models_pytorch architectures :func:`train` can build.
ARCHITECTURES = (
    "Unet", "UnetPlusPlus", "MAnet", "Linknet", "FPN", "PSPNet", "PAN", "DeepLabV3", "DeepLabV3Plus",
)


def model_metadata(model) -> dict:
    """What :func:`train` stored with *model* (``{}`` for older models):
    architecture, encoder, ``input_size``, ``tile_size``, ``scale``,
    ``in_channels``, ``median_lumen_area``, ..."""
    return dict(getattr(model, "picdens_meta", None) or {})


def train(
    imgs_path: str | Sequence[str],
    mask_path: str | Sequence[str],
    output_path: str,
    save_model_path: str,
    input_image_width: int | None = None,
    input_image_height: int | None = None,
    test_split: float | int = 0.15,
    batch_size: int = 8,
    device: str = "cpu",
    pin_memory: bool = False,
    init_lr: float | int = 3e-4,
    num_epochs: int = 100,
    on_epoch: Callable[..., bool | None] | None = None,
    architecture: str = "UnetPlusPlus",
    encoder_name: str = "efficientnet-b3",
    encoder_weights: str | None = "imagenet",
    metadata: dict | None = None,
    *,
    tile_size: int | None = None,
    scale: float = 1.0,
    in_channels: int = 1,
    blur: str | None = None,
    blur_ksize: int | None = None,
    test_images: Sequence[str] = (),
    groups: Sequence | None = None,
    augment: bool = True,
    weighted_loss: bool = True,
    small_lumen_area: int = 300,
    small_lumen_weight: float = 5.0,
    wall_ring: int = 3,
    patience: int | None = 10,
    scheduler: str | None = "plateau",
    seed: int = 42,
    loss: str = "bce_dice",
    lumen_weight: float = 1.0,
    focal_gamma: float = 2.0,
    tversky_beta: float = 0.7,
    select_metric: str = "iou_tol",
    boundary_tolerance: int = 1,
) -> torch.nn.Module:
    """Train a U-Net segmentation model on wood micrograph image–mask pairs.

    Two kinds of input:

    * ``tile_size`` given - *imgs_path* / *mask_path* are whole micrographs.
      They are split into training and validation sets **by image** first;
      every epoch then draws random ``tile_size`` crops from the training
      images (about as many as the images have whole tiles), and validation
      runs on a fixed grid of tiles over the validation images, the last
      tile flush with the image edge (as in ``unet.utils.predictImgMask``).
    * ``tile_size=None`` - the files are ready-made tiles (e.g. from
      ``unet.utils.cutAndPadImg``, already median-blurred - they are not
      blurred again), used whole. They are split by *groups*, by default
      the source image read from the file name: ``<name>_<number>.<ext>``
      -> ``<name>``, as ``cutAndPadImg`` names tiles.

    Parameters
    ----------
    imgs_path : str | Sequence[str]
        Directory with images, or a list of image files.
    mask_path : str | Sequence[str]
        Directory with masks (white - lumen), or a list of mask files in
        the same order as the images. Directories are paired by sorted
        file name, so image and mask names must sort alike.
    output_path : str
        Directory where the training plot (``lossPlot.jpeg``) is saved.
    save_model_path : str
        File for the best model (.pth). Rewritten whenever validation IoU
        improves, so an interrupted run keeps the best epoch so far.
    input_image_width, input_image_height : int | None
        Network input; tiles of another size are resized to it. Default
        (``None``): the tile itself, no resizing.
    test_split : float | int
        Validation share of the source images (``0.15``) or their number.
        With a single source image the last ``test_split`` of it (along its
        longer side) is used for validation instead.
    batch_size : int
        Tiles per gradient step.
    device : str
        ``"cuda"`` or ``"cpu"``.
    pin_memory : bool
        DataLoader pinned memory; ignored without CUDA (it deadlocks on
        CPU-only Windows machines).
    init_lr : float
        Initial learning rate of AdamW.
    num_epochs : int
        Maximum number of epochs.
    on_epoch : callable | None
        Called after every epoch as ``on_epoch(epoch, num_epochs,
        train_loss, val_loss)`` (epoch counted from 1), plus
        ``val_iou=...`` if it takes that keyword; a truthy return stops
        training after that epoch, keeping the best model so far. When
        given, the console progress bars are switched off.
    architecture : str
        A segmentation_models_pytorch model class, one of
        :data:`ARCHITECTURES` (default ``"UnetPlusPlus"``).
    encoder_name : str
        Backbone, e.g. ``"efficientnet-b3"`` (default), ``"resnet34"``.
    encoder_weights : str | None
        ``"imagenet"`` (default; downloaded on first use) or ``None``.
    metadata : dict | None
        Extra facts saved with the model; they override the computed ones
        (see below).
    tile_size : int | None
        Side of the training tiles cut from whole images (256 is a good
        choice: a tile then holds a whole cell with its neighbours).
    scale : float
        Images and masks are scaled by this before tiling (whole images
        only); binarization must use the same scale.
    in_channels : int
        1 - grayscale (default), 3 - RGB. See ``unet.utils.preprocess_image``.
    blur, blur_ksize : str | None, int | None
        Smoothing of whole images before tiling: ``"median"``, ``"gaussian"``,
        ``"bilateral"`` or ``"none"`` (``unet.utils.apply_blur``) with an odd
        kernel in px (default 5). ``None``: a 5 px median for grayscale (as
        before), none for RGB. Saved in the model; binarization applies the
        same. Ready-made tiles are not blurred again: here *blur* /
        *blur_ksize* say what they were already blurred with when cut
        (``cutAndPadImg``'s own *blur* / *blur_ksize*, by default the same
        5 px median) and are only recorded in the model.
    test_images : Sequence[str]
        Held-out images never used for training or validation: paths or
        names; a pair is dropped when its image's file stem (or, for tiles,
        its group) matches the stem of one of them.
    groups : Sequence | None
        One source-image key per (tile) file for the split; see above.
    augment : bool
        Random 90° rotations, flips and ±15% brightness/contrast of the
        training tiles.
    weighted_loss : bool
        Weight ×*small_lumen_weight* on lumens smaller than
        *small_lumen_area* px (after *scale*) and a *wall_ring* px ring of
        wall around them; they are the hardest part (latewood).
    patience : int | None
        Stop after this many epochs without a better validation IoU
        (``None`` or 0: run all *num_epochs*).
    scheduler : str | None
        ``"plateau"`` - halve the learning rate when validation IoU stalls
        for ``patience // 3`` epochs; ``"cosine"`` - cosine decay to
        ``init_lr / 100`` over *num_epochs*; ``None`` - constant.
    seed : int
        Seed of the split, the crops and the augmentations.
    loss : str
        Loss function, one of :data:`LOSSES`: ``"bce_dice"`` (default),
        ``"bce"``, ``"dice"``, ``"focal_dice"``, ``"bce_tversky"``,
        ``"tversky"`` - see :func:`make_loss`.
    lumen_weight : float
        Extra weight of every lumen pixel in the BCE / focal part (1 - as
        the wall; > 1 - a missed lumen pixel costs more than a false one).
    focal_gamma : float
        Focusing exponent of the focal loss (``"focal_dice"``): larger -
        less weight on pixels the network already gets right.
    tversky_beta : float
        Tversky weight of missed lumen pixels (false negatives); false
        positives get ``1 - tversky_beta``. 0.5 is Dice; 0.7 penalises
        misses more - for narrow latewood lumens.
    select_metric : str
        Validation metric (:data:`SELECT_METRICS`) that picks the best epoch
        and drives early stopping and the ``"plateau"`` scheduler: ``"iou"``,
        ``"iou_tol"`` (default - IoU ignoring a *boundary_tolerance* px band
        around reference lumen edges, where the annotation itself is
        uncertain), ``"lw_iou"`` (IoU in the latewood zone), ``"lw_iou_tol"``
        (the same without the band) or ``"small_recall"`` (share of small
        lumens found).
    boundary_tolerance : int
        Width in px of the band around reference lumen edges that
        ``iou_tol`` and ``lw_iou_tol`` ignore (default 1).

    Validation runs on whole validation images (tile predictions stitched
    back), so lumens cut by tile borders are judged whole; every metric is
    reported each epoch (see :func:`val_metrics`).

    Returns
    -------
    torch.nn.Module
        The model of the epoch with the best validation IoU. Its
        ``picdens_meta`` (:func:`model_metadata`) holds the architecture,
        ``tile_size``, ``input_size``, ``scale``, ``in_channels``, ``blur``, ``blur_ksize``,
        ``median_lumen_area`` (px, training masks after *scale*), the
        split, the best epoch and its ``best_val_iou``.
    """
    if architecture not in ARCHITECTURES:
        raise ValueError(f"unknown architecture {architecture!r}; known: {ARCHITECTURES}")
    if in_channels not in (1, 3):
        raise ValueError(f"in_channels must be 1 or 3, not {in_channels}")
    if scheduler not in (None, "plateau", "cosine"):
        raise ValueError(f"unknown scheduler {scheduler!r}")
    if select_metric not in SELECT_METRICS:
        raise ValueError(f"unknown select_metric {select_metric!r}; known: {tuple(SELECT_METRICS)}")
    loss_fn = make_loss(loss, lumen_weight=lumen_weight, focal_gamma=focal_gamma, tversky_beta=tversky_beta)
    blur, blur_ksize = normalize_blur(blur, blur_ksize, in_channels)
    # ready-made tiles (tile_size None) are not blurred again: blur only records how they were cut
    whole = tile_size is not None
    # pin_memory is only meaningful (and safe) when a CUDA device is available;
    # on CPU-only Windows machines it causes the DataLoader to deadlock.
    if pin_memory and not torch.cuda.is_available():
        print("[WARNING] pin_memory=True requested but CUDA is not available. "
              "Disabling pin_memory to prevent DataLoader deadlock on Windows.")
        pin_memory = False
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    quiet = on_epoch is not None

    imagePaths = sorted(paths.list_images(imgs_path)) if isinstance(imgs_path, str) else [str(p) for p in imgs_path]
    maskPaths = sorted(paths.list_images(mask_path)) if isinstance(mask_path, str) else [str(p) for p in mask_path]
    if len(imagePaths) != len(maskPaths):
        raise ValueError(f"{len(imagePaths)} images but {len(maskPaths)} masks")
    if whole:
        keys = [Path(p).stem for p in imagePaths]
    elif groups is not None:
        if len(groups) != len(imagePaths):
            raise ValueError(f"{len(groups)} groups for {len(imagePaths)} tiles")
        keys = [str(g) for g in groups]
    else:
        keys = [re.sub(r"_\d+$", "", Path(p).stem) for p in imagePaths]

    # --- held-out test images, then the split by source image ---
    held = {Path(str(t)).stem for t in test_images}
    keep = [i for i, p in enumerate(imagePaths) if keys[i] not in held and Path(p).stem not in held]
    n_held = len(imagePaths) - len(keep)
    if not keep:
        raise ValueError("no image–mask pairs left after removing the test images")
    sources = sorted({keys[i] for i in keep})
    rng.shuffle(sources)
    val_frac = test_split if test_split < 1 else 0.15  # for the fallbacks below
    n_val = int(test_split) if test_split >= 1 else round(test_split * len(sources))
    n_val = min(max(n_val, 1), len(sources) - 1)
    val_sources = set(sources[:n_val])
    train_idx = [i for i in keep if keys[i] not in val_sources]
    val_idx = [i for i in keep if keys[i] in val_sources]
    print(f"[INFO] {len(sources)} source images: {len(sources) - len(val_sources)} train, "
          f"{len(val_sources)} validation; {n_held} files held out for testing")

    def load(i: int) -> tuple[np.ndarray, np.ndarray]:
        return _load_pair(imagePaths[i], maskPaths[i], scale if whole else 1.0, in_channels,
                          blur=(blur, blur_ksize) if whole else None)

    train_items = [load(i) for i in train_idx]
    val_items = [load(i) for i in val_idx]
    if not val_items and whole:
        # a single source image: validate on a separate part of it
        print(f"[WARNING] only one source image - validating on its last {val_frac:.0%} along the longer side")
        (img, mask), = train_items
        if img.shape[0] < img.shape[1]:
            img, mask = img.swapaxes(0, 1), mask.swapaxes(0, 1)  # cut along the rows
        cut = max(1, round(img.shape[0] * (1 - val_frac)))
        train_items, val_items = [(img[:cut], mask[:cut])], [(img[cut:], mask[cut:])]
    elif not val_items:
        print("[WARNING] all tiles come from one source image - splitting the tiles at random, "
              "validation will be optimistic")
        if len(train_items) < 2:
            raise ValueError("at least two tiles are needed (training + validation)")
        order = rng.permutation(len(train_items))
        n = min(max(1, round(len(order) * val_frac)), len(order) - 1)
        val_items = [train_items[k] for k in order[:n]]
        train_items = [train_items[k] for k in order[n:]]

    # --- tile / input size ---
    if whole:
        tile_h = tile_w = int(tile_size)
    else:
        tile_h, tile_w = train_items[0][0].shape[:2]
    input_w = int(input_image_width or tile_w)
    input_h = int(input_image_height or input_image_width or tile_h)

    def weights(mask: np.ndarray) -> np.ndarray:
        if not weighted_loss:
            return np.ones(mask.shape, np.float32)
        return weight_map(mask, small_lumen_area, small_lumen_weight, wall_ring)

    train_items = [(img, mask, weights(mask)) for img, mask in train_items]
    val_items = [(img, mask, weights(mask)) for img, mask in val_items]
    median_area = _median_lumen_area([m for _, m, _ in train_items])

    trainDS = _CropDataset(train_items, (tile_h, tile_w), (input_h, input_w), random_crops=whole,
                           augment=augment, rng=rng)
    valDS = _GridDataset(val_items, (tile_h, tile_w), (input_h, input_w))
    print(f"[INFO] {len(trainDS)} training tiles per epoch, {len(valDS)} validation tiles, "
          f"tile {tile_w}x{tile_h} -> input {input_w}x{input_h}, median lumen {median_area:.0f} px")

    # num_workers=0: load data in the main process. On Windows, workers use
    # "spawn", which needs every DataLoader inside `if __name__ == "__main__"`
    # - train() is a library function (and runs in a GUI worker thread).
    trainLoader = DataLoader(trainDS, shuffle=True, batch_size=batch_size, pin_memory=pin_memory, num_workers=0)
    valLoader = DataLoader(valDS, shuffle=False, batch_size=batch_size, pin_memory=pin_memory, num_workers=0)

    print(f"[INFO] loading {architecture} / {encoder_name} "
          f"(encoder weights may be downloaded on first run)...")
    unet = getattr(smp, architecture)(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=1,  # binary segmentation: lumen / wall
    ).to(device)
    print(f"[INFO] model ready on {device}.")

    opt = torch.optim.AdamW(unet.parameters(), lr=init_lr, weight_decay=1e-4)
    if scheduler == "plateau":
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="max", factor=0.5, patience=max(1, (patience or 10) // 3))
    elif scheduler == "cosine":
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=num_epochs, eta_min=init_lr / 100)
    else:
        sched = None
    pass_iou = _accepts_keyword(on_epoch, "val_iou")
    pass_metrics = _accepts_keyword(on_epoch, "metrics")
    metric_kw = dict(small_area=small_lumen_area, tolerance=boundary_tolerance)

    meta = {
        "architecture": architecture,
        "encoder": encoder_name,
        "encoder_weights": encoder_weights,
        "tile_size": tile_w,
        "input_size": input_w,
        "in_channels": in_channels,
        "blur": blur,
        "blur_ksize": blur_ksize,
        "median_lumen_area": median_area,
        "loss": loss,
        "loss_params": dict(lumen_weight=lumen_weight, focal_gamma=focal_gamma, tversky_beta=tversky_beta,
                            small_lumen_weight=small_lumen_weight if weighted_loss else 1.0,
                            small_lumen_area=small_lumen_area, wall_ring=wall_ring),
        "select_metric": select_metric,
        "boundary_tolerance": boundary_tolerance,
        "augment": augment,
        "training": dict(device=device, batch_size=batch_size, init_lr=init_lr, max_epochs=num_epochs,
                         patience=patience, scheduler=scheduler, seed=seed, test_split=test_split,
                         weighted_loss=weighted_loss, input_size_hw=[input_h, input_w], whole_images=whole,
                         n_pairs=len(imagePaths), n_held_out_files=n_held,
                         train_tiles_per_epoch=len(trainDS), val_tiles=len(valDS)),
        "train_images": sorted({keys[i] for i in train_idx}),
        "val_images": sorted(val_sources),
        "test_images": sorted(held),
    }
    if whole:
        meta["scale"] = scale  # the scale of ready-made tiles is not known here
    meta.update(metadata or {})
    # all the settings, the split, the best epoch and every epoch's metrics, readable without Python
    params_path = os.path.join(output_path, Path(save_model_path).stem + "_params.txt")
    save = lambda: _save(unet, save_model_path, meta, H, best_epoch, best, params_path)  # noqa: E731

    H = {"train_loss": [], "val_loss": [], **{f"val_{k}": [] for k in SELECT_METRICS}}
    best_score, best_epoch, best_state, best = -1.0, 0, None, {}
    print("[INFO] training the network...")
    startTime = time.time()

    for e in tqdm(range(num_epochs), desc="Epochs", disable=quiet):
        unet.train()
        total, steps = 0.0, 0
        bar = tqdm(trainLoader, desc=f"  Epoch {e + 1}/{num_epochs} train", leave=False, unit="batch",
                   disable=quiet)
        for x, y, w in bar:
            x, y, w = x.to(device), y.to(device), w.to(device)
            batch_loss = loss_fn(unet(x), y, w)
            opt.zero_grad()
            batch_loss.backward()
            opt.step()
            total += batch_loss.item()
            steps += 1
            bar.set_postfix(loss=f"{batch_loss.item():.4f}")
        train_loss = total / max(steps, 1)

        val_loss, metrics = evaluate(unet, valDS, valLoader, device, loss_fn, **metric_kw)
        H["train_loss"].append(train_loss)
        H["val_loss"].append(val_loss)
        for k in SELECT_METRICS:
            H[f"val_{k}"].append(metrics[k])
        score = metrics[select_metric]
        if np.isnan(score):  # e.g. no small lumen in the validation images
            score = metrics["iou"]
        if scheduler == "plateau":
            sched.step(score)
        elif sched is not None:
            sched.step()

        improved = score > best_score
        if improved:
            best_score, best_epoch, best = score, e + 1, dict(metrics, val_loss=val_loss)
            best_state = {k: v.detach().clone() for k, v in unet.state_dict().items()}
            save()

        shown = ", ".join(f"{k} {metrics[k]:.4f}" for k in SELECT_METRICS)
        print(f"[INFO] EPOCH: {e + 1}/{num_epochs}  train loss {train_loss:.4f}, val loss {val_loss:.4f}, "
              f"val {shown}{' (best ' + select_metric + ')' if improved else ''}, lr {opt.param_groups[0]['lr']:.2e}")
        args = (e + 1, num_epochs, train_loss, val_loss)
        kwargs = {}
        if pass_iou:
            kwargs["val_iou"] = metrics["iou"]
        if pass_metrics:
            kwargs["metrics"] = dict(metrics, select=select_metric, best=best_score, best_epoch=best_epoch)
        if on_epoch is not None and on_epoch(*args, **kwargs):
            print(f"[INFO] training stopped after epoch {e + 1}")
            break
        if patience and e + 1 - best_epoch >= patience:
            print(f"[INFO] early stopping: no better validation {select_metric} for {patience} epochs")
            break

    print("[INFO] total time taken to train the model: {:.2f}s".format(time.time() - startTime))
    if best_state is not None:
        unet.load_state_dict(best_state)
    else:  # every epoch's validation metric was NaN (e.g. no lumen in the validation images)
        print("[WARNING] no epoch had a valid validation metric - keeping the last epoch's weights")
    meta["training"]["seconds"] = round(time.time() - startTime, 1)
    save()
    _plot(H, best_epoch, select_metric, loss, output_path)
    return unet


# ---------------------------------------------------------------------------
# loss and metrics
# ---------------------------------------------------------------------------

#: Loss functions :func:`train` accepts (``loss=``), see :func:`make_loss`.
LOSSES = {
    "bce_dice": "BCE + Dice",
    "bce": "BCE",
    "dice": "Dice",
    "focal_dice": "Focal + Dice",
    "bce_tversky": "BCE + Tversky",
    "tversky": "Tversky",
}

#: Validation metrics :func:`train` can pick the best epoch by (``select_metric=``), see :func:`val_metrics`.
SELECT_METRICS = {
    "iou": "IoU",
    "iou_tol": "IoU с допуском у границы",
    "lw_iou": "IoU поздней древесины",
    "lw_iou_tol": "IoU поздней древесины с допуском",
    "small_recall": "Доля найденных малых просветов",
}


def _pixel_weights(target: torch.Tensor, weight: torch.Tensor, lumen_weight: float) -> torch.Tensor:
    return weight * (1 + (lumen_weight - 1) * target) if lumen_weight != 1 else weight


def _weighted_mean(values: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return (values * weight).sum() / weight.sum().clamp_min(1e-6)


def _overlap_terms(logits, target, weight):
    """Soft TP / FP / FN over the valid (weight > 0) pixels of the batch."""
    valid = (weight > 0).float()
    prob = torch.sigmoid(logits) * valid
    target = target * valid
    tp = (prob * target).sum()
    return tp, (prob * (1 - target)).sum(), ((1 - prob) * target).sum()


def make_loss(name: str = "bce_dice", lumen_weight: float = 1.0, focal_gamma: float = 2.0,
              tversky_beta: float = 0.7) -> Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
    """``loss(logits, target, weight)`` - *weight* is the per-pixel map (small-lumen weights,
    0 in padding, which every term ignores):

    * ``"bce"`` - binary cross-entropy, weighted per pixel; *lumen_weight* multiplies the
      weight of lumen pixels (> 1: a missed lumen pixel costs more than a false one);
    * ``"dice"`` - soft Dice of the lumen class: 1 - 2TP / (2TP + FP + FN); insensitive to the
      lumen/wall ratio, rewards overlap of the lumens as a whole;
    * ``"focal_dice"`` - focal loss (BCE × (1 - p_true)^*focal_gamma*: easy pixels count less,
      the hard ones - edges, narrow lumens - more) + Dice;
    * ``"tversky"`` - 1 - TP / (TP + (1-β)·FP + β·FN) with β = *tversky_beta*; β > 0.5 punishes
      missed lumen pixels more than false ones (β = 0.5 is Dice);
    * ``"bce_dice"`` (default), ``"bce_tversky"`` - the sums.
    """
    if name not in LOSSES:
        raise ValueError(f"unknown loss {name!r}; known: {tuple(LOSSES)}")
    if not 0 < tversky_beta < 1:
        raise ValueError(f"tversky_beta must be in (0, 1), not {tversky_beta}")

    def bce(logits, target, weight):
        w = _pixel_weights(target, weight, lumen_weight)
        return _weighted_mean(F.binary_cross_entropy_with_logits(logits, target, reduction="none"), w)

    def focal(logits, target, weight):
        w = _pixel_weights(target, weight, lumen_weight)
        ce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        p_true = torch.exp(-ce)  # probability given to the true class
        return _weighted_mean(ce * (1 - p_true) ** focal_gamma, w)

    def dice(logits, target, weight):
        tp, fp, fn = _overlap_terms(logits, target, weight)
        return 1 - (2 * tp + 1) / (2 * tp + fp + fn + 1)

    def tversky(logits, target, weight):
        tp, fp, fn = _overlap_terms(logits, target, weight)
        # smoothing 0.5 so that beta = 0.5 gives exactly the Dice above
        return 1 - (tp + 0.5) / (tp + (1 - tversky_beta) * fp + tversky_beta * fn + 0.5)

    parts = {"bce": (bce,), "dice": (dice,), "focal_dice": (focal, dice), "bce_dice": (bce, dice),
             "bce_tversky": (bce, tversky), "tversky": (tversky,)}[name]

    def loss(logits, target, weight):
        return sum(part(logits, target, weight) for part in parts)

    return loss


def bce_dice_loss(logits: torch.Tensor, target: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Per-pixel weighted BCE plus soft Dice (``make_loss("bce_dice")``)."""
    return make_loss("bce_dice")(logits, target, weight)


@torch.no_grad()
def evaluate(model: torch.nn.Module, dataset: "_GridDataset", loader: DataLoader, device: str,
             loss_fn=bce_dice_loss, small_area: int = 300, tolerance: int = 1) -> tuple[float, dict]:
    """Mean loss over the validation tiles and :func:`val_metrics` of the predictions stitched
    back into whole validation images."""
    model.eval()
    probs = [np.zeros(shape, np.float32) for shape in dataset.shapes]
    total, steps = 0.0, 0
    for x, y, w, idx in loader:
        x, y, w = x.to(device), y.to(device), w.to(device)
        logits = model(x)
        total += loss_fn(logits, y, w).item()
        steps += 1
        prob = torch.sigmoid(logits)[:, 0].cpu().numpy()
        for p, i in zip(prob, idx.tolist()):
            dataset.place(probs, i, p)
    pairs = [(prob[:mask.shape[0], :mask.shape[1]] > 0.5, mask) for prob, (_, mask, _) in zip(probs, dataset.items)]
    return total / max(steps, 1), val_metrics(pairs, small_area, tolerance)


def val_metrics(pairs: list[tuple[np.ndarray, np.ndarray]], small_area: int = 300,
                tolerance: int = 1) -> dict[str, float]:
    """Metrics of (prediction, reference) boolean lumen masks, pooled over all images:

    * ``iou`` - pixel IoU of the lumens;
    * ``iou_tol`` - the same without a *tolerance* px band around reference lumen edges, where
      the annotation itself is uncertain (a 1 px shift of a whole reference costs ~0.1 IoU);
    * ``lw_iou`` - IoU in the latewood zone: pixels whose nearest reference lumen is smaller
      than *small_area* px, up to 15 px from it;
    * ``lw_iou_tol`` - the latewood IoU without the *tolerance* px band (narrow latewood lumens
      lose ~0.3 IoU to a 1 px shift of the reference, so ``lw_iou`` stalls near 0.6-0.7);
    * ``small_recall`` - share of reference lumens smaller than *small_area* px (not touching
      the image edge) that a predicted lumen overlaps with IoU >= 0.5 (NaN without any).
    """
    from scipy import ndimage as ndi

    kernel = np.ones((3, 3), np.uint8)
    sums = dict(i=0, u=0, ti=0, tu=0, li=0, lu=0, lti=0, ltu=0, found=0, small=0)
    for pred, ref in pairs:
        sums["i"] += int((pred & ref).sum())
        sums["u"] += int((pred | ref).sum())
        if tolerance > 0:
            r8 = ref.astype(np.uint8)
            band = (cv2.dilate(r8, kernel, iterations=tolerance) > 0) & ~(cv2.erode(r8, kernel, iterations=tolerance) > 0)
            keep = ~band
        else:
            keep = np.ones_like(ref)
        sums["ti"] += int((pred & ref & keep).sum())
        sums["tu"] += int(((pred | ref) & keep).sum())

        n, lab, stats, _ = cv2.connectedComponentsWithStats(ref.astype(np.uint8), connectivity=4)
        if n <= 1:
            continue
        area = stats[:, cv2.CC_STAT_AREA]
        small_lab = area < small_area
        small_lab[0] = False
        # latewood zone
        dist, (iy, ix) = ndi.distance_transform_edt(lab == 0, return_indices=True)
        zone = small_lab[lab[iy, ix]] & (dist <= 15)
        sums["li"] += int((pred & ref & zone).sum())
        sums["lu"] += int(((pred | ref) & zone).sum())
        sums["lti"] += int((pred & ref & zone & keep).sum())
        sums["ltu"] += int(((pred | ref) & zone & keep).sum())
        # small lumens found (IoU >= 0.5 with one predicted component)
        h, w = ref.shape
        x0, y0, bw, bh = stats[:, 0], stats[:, 1], stats[:, 2], stats[:, 3]
        inner = (x0 > 0) & (y0 > 0) & (x0 + bw < w) & (y0 + bh < h)
        ids = np.flatnonzero(small_lab & inner)
        if ids.size == 0:
            continue
        m, plab, pstats, _ = cv2.connectedComponentsWithStats(pred.astype(np.uint8), connectivity=4)
        both = (lab > 0) & (plab > 0)
        if both.any():
            pairs_ = np.stack([lab[both], plab[both]])
            uniq, inter = np.unique(pairs_, axis=1, return_counts=True)
            iou = inter / (area[uniq[0]] + pstats[uniq[1], cv2.CC_STAT_AREA] - inter)
            found = np.zeros(n, bool)
            found[uniq[0][iou >= 0.5]] = True
            sums["found"] += int(found[ids].sum())
        sums["small"] += int(ids.size)

    def ratio(a, b):
        return a / b if b else float("nan")

    return dict(iou=ratio(sums["i"], sums["u"]), iou_tol=ratio(sums["ti"], sums["tu"]),
                lw_iou=ratio(sums["li"], sums["lu"]), lw_iou_tol=ratio(sums["lti"], sums["ltu"]),
                small_recall=ratio(sums["found"], sums["small"]))


def weight_map(mask: np.ndarray, small_area: int = 300, weight: float = 5.0, ring: int = 3) -> np.ndarray:
    """1 everywhere, *weight* on lumens (``mask`` True) smaller than *small_area* px and on a
    *ring* px band of wall around them."""
    _, lab, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=4)
    small = (stats[:, cv2.CC_STAT_AREA] < small_area)[lab] & mask
    kernel = np.ones((2 * ring + 1, 2 * ring + 1), np.uint8)
    near = cv2.dilate(small.astype(np.uint8), kernel) > 0
    return np.where(near, weight, 1.0).astype(np.float32)


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------

def _load_pair(img_path: str, mask_path: str, scale: float, in_channels: int,
               blur: tuple[str, int] | None = ("median", 5)) -> tuple[np.ndarray, np.ndarray]:
    """Preprocessed image (uint8, as ``unet.utils.predictImgMask`` sees it) and boolean lumen
    mask of the same size. *blur*: ``(type, kernel)``; ``None``: ready-made tiles, already
    blurred by ``cutAndPadImg`` - only converted."""
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(img_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(mask_path)
    if mask.shape != img.shape[:2]:
        raise ValueError(f"mask {mask.shape} and image {img.shape[:2]} differ: {mask_path}")
    if blur is not None:
        img = preprocess_image(img, scale, in_channels, *blur)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY if in_channels == 1 else cv2.COLOR_BGR2RGB)
    if mask.shape != img.shape[:2]:
        mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_AREA)
    # masks may be 0/255, 0/113 (CVAT exports) or JPEG-noisy: half the brightest value
    return img, mask > max(int(mask.max()) / 2, 32)


def _median_lumen_area(masks: list[np.ndarray]) -> float:
    areas = [cv2.connectedComponentsWithStats(m.astype(np.uint8), connectivity=4)[2][1:, cv2.CC_STAT_AREA]
             for m in masks]
    areas = np.concatenate(areas) if areas else np.empty(0)
    return float(np.median(areas)) if areas.size else 0.0


def _pad_to(img, mask, weight, h, w):
    """Pad an image smaller than a tile (reflected image, weight 0 in the padding)."""
    ph, pw = max(0, h - img.shape[0]), max(0, w - img.shape[1])
    if not (ph or pw):
        return img, mask, weight
    img = cv2.copyMakeBorder(np.ascontiguousarray(img), 0, ph, 0, pw, cv2.BORDER_REFLECT)
    return img, np.pad(mask, ((0, ph), (0, pw))), np.pad(weight, ((0, ph), (0, pw)))


def _to_tensors(img, mask, weight, input_hw):
    h, w = input_hw
    if img.dtype == np.uint8:
        img = img.astype(np.float32) / 255.0
    if img.shape[:2] != (h, w):
        img = cv2.resize(np.ascontiguousarray(img), (w, h),
                         interpolation=cv2.INTER_AREA if img.shape[0] > h else cv2.INTER_LINEAR)
        mask = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
        weight = cv2.resize(np.ascontiguousarray(weight), (w, h), interpolation=cv2.INTER_NEAREST)
    x = img[np.newaxis] if img.ndim == 2 else img.transpose(2, 0, 1)
    return (torch.from_numpy(np.ascontiguousarray(x, np.float32)),
            torch.from_numpy(np.ascontiguousarray(mask[np.newaxis], np.float32)),
            torch.from_numpy(np.ascontiguousarray(weight[np.newaxis], np.float32)))


class _CropDataset(Dataset):
    """Training tiles: random crops from whole images (as many per epoch as an image has whole
    tiles), or the given tiles themselves; optionally augmented."""

    def __init__(self, items, tile_hw, input_hw, random_crops, augment, rng):
        self.tile_hw, self.input_hw = tile_hw, input_hw
        self.random_crops, self.augment, self.rng = random_crops, augment, rng
        th, tw = tile_hw
        self.items = [_pad_to(*it, th, tw) for it in items] if random_crops else items
        self.index = [k for k, (img, _, _) in enumerate(self.items)
                      for _ in range(math.ceil(img.shape[0] * img.shape[1] / (th * tw)) if random_crops else 1)]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        img, mask, weight = self.items[self.index[idx]]
        if self.random_crops:
            th, tw = self.tile_hw
            y = int(self.rng.integers(0, img.shape[0] - th + 1))
            x = int(self.rng.integers(0, img.shape[1] - tw + 1))
            img, mask, weight = img[y:y + th, x:x + tw], mask[y:y + th, x:x + tw], weight[y:y + th, x:x + tw]
        if self.augment:
            img, mask, weight = self._augment(img, mask, weight)
        return _to_tensors(img, mask, weight, self.input_hw)

    def _augment(self, img, mask, weight):
        rng = self.rng
        if img.shape[0] == img.shape[1]:
            k = int(rng.integers(4))
            img, mask, weight = np.rot90(img, k), np.rot90(mask, k), np.rot90(weight, k)
        if rng.random() < 0.5:
            img, mask, weight = img[:, ::-1], mask[:, ::-1], weight[:, ::-1]
        if rng.random() < 0.5:
            img, mask, weight = img[::-1], mask[::-1], weight[::-1]
        # brightness and contrast ±15%
        x = img.astype(np.float32) / 255.0
        mean = float(x.mean())
        x = (x - mean) * rng.uniform(0.85, 1.15) + mean * rng.uniform(0.85, 1.15)
        return np.clip(x, 0, 1), mask, weight


class _GridDataset(Dataset):
    """Validation tiles: a fixed grid over each image, the last tile flush with its edge.
    Items are ``(x, y, weight, tile index)``; :meth:`place` puts a tile's prediction back into
    a whole-image array (:attr:`shapes` - the padded image sizes, :attr:`items` - the images)."""

    def __init__(self, items, tile_hw, input_hw):
        self.input_hw, self.tile_hw, self.items = input_hw, tile_hw, items
        th, tw = tile_hw
        self.tiles, self.where, self.shapes = [], [], []
        for k, (img, mask, weight) in enumerate(items):
            img, mask, weight = _pad_to(img, mask, weight, th, tw)
            self.shapes.append(img.shape[:2])
            for y in _tile_starts(img.shape[0], th, th):
                for x in _tile_starts(img.shape[1], tw, tw):
                    self.tiles.append((img[y:y + th, x:x + tw], mask[y:y + th, x:x + tw],
                                       weight[y:y + th, x:x + tw]))
                    self.where.append((k, y, x))

    def __len__(self):
        return len(self.tiles)

    def __getitem__(self, idx):
        return (*_to_tensors(*self.tiles[idx], self.input_hw), idx)

    def place(self, arrays: list[np.ndarray], idx: int, pred: np.ndarray) -> None:
        k, y, x = self.where[idx]
        th, tw = self.tile_hw
        if pred.shape != (th, tw):
            pred = cv2.resize(pred, (tw, th), interpolation=cv2.INTER_LINEAR)
        arrays[k][y:y + th, x:x + tw] = pred


# ---------------------------------------------------------------------------
# saving
# ---------------------------------------------------------------------------

def _accepts_keyword(fn, name: str) -> bool:
    if fn is None:
        return False
    try:
        params = inspect.signature(fn).parameters.values()
    except (TypeError, ValueError):
        return False
    return any(p.name == name or p.kind is inspect.Parameter.VAR_KEYWORD for p in params)


def _save(model, path, meta, H, best_epoch, best: dict, params_path: str | None = None) -> None:
    # Facts needed to use the model the way it was trained (read back by
    # Binarizer via model_metadata); a plain attribute pickles with the model.
    model.picdens_meta = {
        **meta,
        "epochs": len(H["train_loss"]),
        "best_epoch": best_epoch,
        "best_val_iou": float(best.get("iou", float("nan"))),
        "best_val_metrics": {k: float(v) for k, v in best.items()},
        "history": {k: [float(v) for v in vals] for k, vals in H.items()},
        "trained": time.strftime("%Y-%m-%d %H:%M"),
    }
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save(model, path)
    if params_path:
        write_params(model.picdens_meta, params_path, model_path=path)


def write_params(meta: dict, params_path: str, model_path: str | None = None) -> None:
    """A readable text file with everything :func:`train` stored in the model (``picdens_meta``):
    settings, data split, the best epoch and a table of every epoch's loss and metrics."""
    meta = dict(meta)
    history = meta.pop("history", {})
    best = meta.pop("best_val_metrics", {})
    training = meta.pop("training", {})
    loss_params = meta.pop("loss_params", {})
    lists = {k: meta.pop(k, []) for k in ("train_images", "val_images", "test_images")}

    def fmt(v):
        if isinstance(v, float):
            return f"{v:.6g}"
        return str(v)

    lines = [f"Параметры модели U-Net - {meta.get('trained', '')}"]
    if model_path:
        lines.append(f"Файл модели: {os.path.abspath(model_path)}")
    lines.append("")
    lines.append("== Модель и данные")
    lines += [f"{k}: {fmt(v)}" for k, v in meta.items() if k not in ("best_epoch", "best_val_iou", "epochs", "trained")]
    lines += ["", "== Обучение"] + [f"{k}: {fmt(v)}" for k, v in training.items()]
    lines += ["", f"== Функция ошибки: {LOSSES.get(meta.get('loss'), meta.get('loss'))}"]
    lines += [f"{k}: {fmt(v)}" for k, v in loss_params.items()]
    lines += ["", "== Результат",
              f"эпох пройдено: {meta.get('epochs')}",
              f"лучшая эпоха: {meta.get('best_epoch')} (по метрике «{SELECT_METRICS.get(meta.get('select_metric'), '')}», "
              f"допуск у границы {meta.get('boundary_tolerance')} px)"]
    lines += [f"  {SELECT_METRICS.get(k, k)}: {v:.4f}" for k, v in best.items() if k in SELECT_METRICS]
    if "val_loss" in best:
        lines.append(f"  ошибка на проверке: {best['val_loss']:.4f}")
    for k, names in lists.items():
        title = {"train_images": "обучение", "val_images": "проверка", "test_images": "отложенный тест"}[k]
        lines += ["", f"== Снимки: {title} ({len(names)})"] + [f"  {n}" for n in names]
    if history.get("train_loss"):
        cols = ["train_loss", "val_loss"] + [f"val_{k}" for k in SELECT_METRICS]
        lines += ["", "== По эпохам", "\t".join(["эпоха"] + cols)]
        for e in range(len(history["train_loss"])):
            mark = " *" if e + 1 == meta.get("best_epoch") else ""
            lines.append("\t".join([f"{e + 1}{mark}"] + [f"{history[c][e]:.4f}" if c in history else "" for c in cols]))
        lines.append("(* - лучшая эпоха)")
    os.makedirs(os.path.dirname(os.path.abspath(params_path)), exist_ok=True)
    with open(params_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def _plot(H, best_epoch, select_metric, loss, output_path) -> None:
    # matplotlib.figure.Figure rather than pyplot, which is not safe outside
    # the main thread (train() runs in a GUI worker)
    from matplotlib.figure import Figure

    colors = {"iou": "#2a78d6", "iou_tol": "#eb6834", "lw_iou": "#1baf7a", "lw_iou_tol": "#eda100",
              "small_recall": "#e87ba4"}
    fig = Figure(figsize=(7.5, 6.5))
    ax_loss, ax_m = fig.subplots(2, 1, sharex=True)
    epochs = range(1, len(H["train_loss"]) + 1)
    ax_loss.plot(epochs, H["train_loss"], color="#52514e", lw=2, label="обучение")
    ax_loss.plot(epochs, H["val_loss"], color="#2a78d6", lw=2, label="проверка")
    ax_loss.set_ylabel(f"Ошибка ({LOSSES.get(loss, loss)})")
    ax_loss.legend(frameon=False)
    for k, name in SELECT_METRICS.items():
        vals = H[f"val_{k}"]
        if not np.all(np.isnan(vals)):
            ax_m.plot(epochs, vals, color=colors[k], lw=2.5 if k == select_metric else 1.5,
                      label=name + (" (выбор эпохи)" if k == select_metric else ""))
    for ax in (ax_loss, ax_m):
        ax.axvline(best_epoch, color="#8a8984", ls=":", lw=1)
        ax.grid(alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)
    ax_m.set_ylabel("Метрика на проверке")
    ax_m.set_xlabel("Эпоха")
    from matplotlib.ticker import MaxNLocator
    ax_m.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax_m.legend(frameon=False, fontsize=8, loc="lower right")
    best = H[f"val_{select_metric}"][best_epoch - 1] if best_epoch else float("nan")
    ax_loss.set_title(f"Лучшая эпоха {best_epoch}: {SELECT_METRICS[select_metric]} {best:.3f}", loc="left")
    fig.tight_layout()
    os.makedirs(output_path, exist_ok=True)
    fig.savefig(os.path.join(output_path, "lossPlot.jpeg"), dpi=120)
