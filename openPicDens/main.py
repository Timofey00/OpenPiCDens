"""
main.py — usage examples for OpenPiCDens.

Three independent scenarios; uncomment the one you need and run:

    python -m openPicDens.main

Scenarios
---------
1. scan_otsu                  — fast scan with classical Otsu thresholding
2. scan_unet                  — scan with U-Net binarization (requires a trained model)
3. scan_from_precomputed_masks — scan from already-binarized images
4. train_and_scan             — train U-Net from scratch, then scan
"""

from __future__ import annotations

import torch

from openPicDens import Binarizer, PICDens
from unet.train import train
from unet.utills import predictImgMask
from unet.preprocessing import tile_image_mask_pairs


# ---------------------------------------------------------------------------
# Scenario 1: scan with classical Otsu thresholding
# ---------------------------------------------------------------------------

def scan_otsu() -> None:
    """Standard scan without a neural network.

    Works well for most samples: fast, no GPU, no prior training needed.
    Tune ``ksize`` and ``gap_value`` to suit your material.
    """
    bi = Binarizer(
        method="Otsu",          # alternatives: "Mean", "Gaussian", "Triangle"
        blur="Median",          # alternatives: "Gaussian", "NBF", "Bilateral", None
        ksize=3,                # blur kernel size (must be odd)
        gamma_equalisation=False,
    )

    scan = PICDens(
        binarizer=bi,
        save_path="path/to/results/",
        root="path/to/images/",       # expected layout: root/<tree_id>/<ring_n>.jpg
        species_name="PS",            # Pinus sylvestris
        year_start=2024,
        norm_number=100,              # length of the normalised porosity profile
        pix_to_mcm_coef=0.42604,      # pixels per micrometre
        norm_method="median",         # "small_ring" | "median" | "mean" | "normNumber"
        sma_interval=90,              # SMA smoothing window
        gap_value=50,                 # max white-pixel run kept by gap filter (None = off)
    )
    scan.startScan()


# ---------------------------------------------------------------------------
# Scenario 2: scan with a pre-trained U-Net model
# ---------------------------------------------------------------------------

def scan_unet() -> None:
    """Scan using U-Net binarization.

    Use when classical methods produce poor segmentation quality
    (uneven illumination, complex background).

    Requires: a trained model at ``model_path``.
    """
    model_path = "path/to/model.pth"

    bi = Binarizer(
        method="UNET",
        model_path=model_path,
        scale=1.0,      # image scale factor before inference
    )

    scan = PICDens(
        binarizer=bi,
        save_path="path/to/results/",
        root="path/to/images/",
        species_name="PS",
        year_start=2024,
        norm_number=100,
        pix_to_mcm_coef=0.42604,
        sma_interval=90,
        gap_value=50,
    )
    scan.startScan()


# ---------------------------------------------------------------------------
# Scenario 3: scan from pre-binarized images
# ---------------------------------------------------------------------------

def scan_from_precomputed_masks() -> None:
    """Scan from already-binarized images.

    Useful when binarization was done beforehand with another tool
    (e.g. ImageJ). The ``binarizer`` argument is ignored in this mode —
    pass any instance as a placeholder.
    """
    bi = Binarizer(method="Otsu")   # not used when use_pred_bi_imgs=True

    scan = PICDens(
        binarizer=bi,
        save_path="path/to/results/",
        root="path/to/images/",             # not used when use_pred_bi_imgs=True
        species_name="PS",
        year_start=2024,
        norm_number=100,
        pix_to_mcm_coef=0.42604,
        use_pred_bi_imgs=True,
        bi_imgs_path="path/to/binary_images/",  # images are read from here
    )
    scan.startScan()


# ---------------------------------------------------------------------------
# Scenario 4: train U-Net from scratch, then scan
# ---------------------------------------------------------------------------

def train_and_scan() -> None:
    """Train a U-Net on your own data, then run a scan.

    Expected training data layout:
        train_imgs/   — raw micrographs (.jpg)
        train_masks/  — binary masks (.png, white pixels = pores)

    If images are too large for the model, tile them first with
    :func:`~unet.preprocessing.tile_image_mask_pairs`.
    """
    # --- optional: tile large images before training ----------------------
    # tile_image_mask_pairs(
    #     img_root="path/to/raw_imgs/",
    #     mask_root="path/to/raw_masks/",
    #     save_img_path="path/to/train_imgs/",
    #     save_mask_path="path/to/train_masks/",
    #     size=128,
    #     scale=1.0,
    # )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_path = "path/to/model.pth"

    # --- training ---------------------------------------------------------
    train(
        imgs_path="path/to/train_imgs/",
        mask_path="path/to/train_masks/",
        output_path="path/to/train_output/",
        save_model_path=model_path,
        input_image_width=128,
        input_image_height=128,
        test_split=0.15,
        batch_size=32,
        device=device,
        pin_memory=(device == "cuda"),
        init_lr=1e-3,
        num_epochs=40,
    )

    # --- predict mask for a single large image (tiled) -------------------
    # The image is split into 128×128 tiles, each tile is passed through
    # the model, and the predicted masks are stitched back together.
    model = torch.load(model_path).to(device)
    mask = predictImgMask(
        img_path="path/to/single_image.jpg",
        save_mask_path="path/to/predicted_mask.png",
        divide_size=128,
        model=model,
        scale=1.0,
    )

    # --- scan with the trained model --------------------------------------
    bi = Binarizer(method="UNET", model_path=model_path)

    scan = PICDens(
        binarizer=bi,
        save_path="path/to/results/",
        root="path/to/images/",
        species_name="PS",
        year_start=2024,
        norm_number=100,
        pix_to_mcm_coef=0.42604,
        sma_interval=90,
        gap_value=50,
    )
    scan.startScan()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Uncomment the scenario you need:
    scan_otsu()
    # scan_unet()
    # scan_from_precomputed_masks()
    # train_and_scan()