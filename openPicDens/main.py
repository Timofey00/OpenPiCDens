"""
main.py — usage examples for OpenPiCDens.

Five independent scenarios; uncomment the one you need and run:

    python -m openPicDens.main

Scenarios
---------
1. scan_otsu                  — fast scan with classical Otsu thresholding
2. scan_unet                  — scan with U-Net binarization (requires a trained model)
3. scan_from_precomputed_masks — scan from already-binarized images
4. train_and_scan             — train U-Net from scratch, then scan
5. scan_rf                    — train a random-forest binariser from masks, then scan
"""

from __future__ import annotations

import logging

import torch

from openPicDens import Binarizer, PICDens
from openPicDens.rf_binarization import train_rf_model
from unet.train import train
from unet.utils import predictImgMask


# ---------------------------------------------------------------------------
# Scenario 1: scan with classical Otsu thresholding
# ---------------------------------------------------------------------------

def scan_otsu() -> None:
    """Standard scan without a neural network.

    Works well for most samples: fast, no GPU, no prior training needed.
    Tune ``ksize`` and ``gap_value`` to suit your material.
    """
    bi = Binarizer(
        method="Otsu",          # any of binarization.BI_METHODS, e.g. "Sauvola", "Li", "Triangle"
        blur="Median",          # alternatives: "Gaussian", "NBF", "Bilateral", "" (no blur)
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
        pix_to_mcm_coef=0.42604,      # micrometres per pixel
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

    Expected training data layout (whole micrographs, paired by sorted name):
        train_imgs/   — raw micrographs (.jpg)
        train_masks/  — binary masks (white pixels = lumens)

    The images are split into training/validation by image and cut into
    random tiles every epoch; the tile size, scale and blur are saved in
    the model and reused by ``Binarizer(method="UNET")``.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_path = "path/to/model.pth"

    # --- training ---------------------------------------------------------
    train(
        imgs_path="path/to/train_imgs/",
        mask_path="path/to/train_masks/",
        output_path="path/to/train_output/",   # lossPlot.jpeg, <model>_params.txt
        save_model_path=model_path,
        tile_size=256,                          # tiles cut from the whole images
        scale=1.0,                              # binarize with the same scale
        test_split=0.15,
        batch_size=8,
        device=device,
        pin_memory=(device == "cuda"),
        num_epochs=100,                         # stops earlier without improvement
    )

    # --- predict mask for a single large image (tiled) -------------------
    # The image is cut into 512x512 tiles (the last one flush with the edge),
    # each passed through the model with the blur it was trained with, and the
    # tile cores are stitched back together.
    # weights_only=False: the .pth holds the whole pickled model (torch >= 2.6).
    model = torch.load(model_path, map_location=device, weights_only=False)
    mask = predictImgMask(
        img_path="path/to/single_image.jpg",
        save_mask_path="path/to/predicted_mask.png",
        divide_size=512,
        model=model,
        scale=1.0,
        overlap=32,
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
# Scenario 5: random-forest pixel classifier (no GPU, no manual labelling)
# ---------------------------------------------------------------------------

def scan_rf() -> None:
    """Train a random-forest binariser from existing masks, then scan.

    Much faster to train than a U-Net (about a minute on CPU) and
    transfers better to other species. Existing U-Net predictions can be
    used as training masks. For a new species, draw a few strokes of
    lumen (white) and wall (black) on a grey (128) canvas and train with
    ``sparse=True``.
    """
    model_path = "path/to/pine_rf.joblib"
    train_rf_model(
        img_paths=["path/to/2003.jpg"],
        mask_paths=["path/to/2003_bi.jpg"],
        save_path=model_path,
        work_scale=0.5,     # images are halved before classification
    )

    bi = Binarizer(
        method="RF",
        model_path=model_path,
        scale=1.0,          # e.g. 2.0 if cells are half as large in pixels
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
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Progress of the scan and of U-Net training goes to the logging module.
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    # Uncomment the scenario you need:
    scan_otsu()
    # scan_unet()
    # scan_from_precomputed_masks()
    # train_and_scan()
    # scan_rf()