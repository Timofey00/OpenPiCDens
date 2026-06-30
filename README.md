# OpenPiCDens

Open-source Python library for **pixel-contrast (PiC) densitometric analysis** of wood cellular structure images. OpenPiCDens converts microphotographs of cross-sections (cores, stem discs, thin sections) into quantitative porosity profiles and ring-width chronologies, exportable in formats compatible with standard dendrochronological software (e.g. CooRecorder, dplR, TSAP).

## Method

Pixel-contrast densitometry estimates relative wood density from ordinary light-microscope photographs, without the need for X-ray equipment. The core idea is that the proportion of cell-wall material versus lumen (empty space) within a small image window is a proxy for wood density — denser latewood has a higher fraction of cell-wall pixels, while porous earlywood or vessel-rich tissue has more lumen pixels.

The pipeline implemented here follows the two-step approach described in Khudykh et al. (2024):

1. **Smoothing.** The grayscale image is denoised (Gaussian, median, or bilateral blur) to suppress noise before thresholding, since binarization is highly sensitive to pixel-level artifacts.
2. **Binarization.** Each pixel is classified as either *cell wall* or *lumen* using a global or adaptive threshold (Otsu, Mean/Gaussian adaptive, Triangle), or — for more structurally complex angiosperm wood — a trained U-Net segmentation model.
3. **Scanning.** A sliding window moves across the binarized image from one ring boundary to the other. Within each window, the local fraction of "wall" pixels gives one point of the density (porosity) profile; multiple overlapping windows are averaged to reduce the effect of uneven illumination and local artifacts.
4. **Derived metrics.** From the resulting profile the library computes ring width, earlywood/latewood width and porosity, sectoral porosity, extreme and summary statistics (minimum, maximum, mean, and quantile porosity, e.g. 5th/95th percentile), normalized (Z-score) profiles, and long-term multi-tree porosity chronologies.

## Features

- Multiple binarization strategies: Otsu, adaptive (mean/Gaussian), Triangle (scikit-image), and U-Net semantic segmentation (PyTorch / `segmentation_models_pytorch`).
- Configurable pre-processing: Gaussian, median, bilateral, and box blur; optional histogram equalization.
- Sliding-window porosity scanning with configurable window size and step, producing smoothed density profiles per ring.
- Automatic earlywood/latewood separation and sectoral porosity analysis.
- Extreme and summary porosity statistics per ring: minimum, maximum, mean, and quantile values (e.g. 5th/95th percentile porosity).
- Long-term, multi-sample chronology building with SMA smoothing and Z-score normalization.
- Export to plain-text (`.txt`) tables and Tucson (`.rwl`) chronology format for direct use in dendrochronological software.
- U-Net training utilities: tiling large images into patches, dataset preparation, training loop, and tiled inference with stitching back to full resolution.
- Works from raw microphotographs, from pre-binarized images (e.g. produced in ImageJ), or with a custom-trained segmentation model.

## Installation

```bash
git clone https://github.com/Timofey00/OpenPiCDens.git
cd OpenPiCDens
pip install -r requirements.txt
```

Requires Python 3.10+ (uses `X | Y` type hints). GPU is optional but recommended for U-Net training; inference and classical thresholding run fine on CPU.

## Quick start

```python
from openPicDens.binarization import Binarizer
from openPicDens.analysis import PICDens

# 1. Choose a binarization method
binarizer = Binarizer(method="Otsu", blur="Gaussian")

# 2. Configure and run the scan
scan = PICDens(
    binarizer=binarizer,
    save_path="path/to/results/",
    root="path/to/images/",
    species_name="PS",
    year_start=2024,
    norm_number=100,
    pix_to_mcm_coef=0.42604,  # microns per pixel, from your microscope calibration
)
scan.startScan()
```

This produces porosity profiles, ring-width measurements, and earlywood/latewood statistics for every image found under `root`, saved to `save_path` as text tables, with `.rwl` chronologies built automatically across all samples.

See `openPicDens/main.py` for complete, runnable examples covering all four supported workflows:

- scanning from raw microphotographs with classical thresholding,
- scanning with a pre-trained U-Net model,
- scanning from already-binarized images (e.g. from ImageJ),
- training a U-Net from scratch on your own annotated data and then scanning with it.

## Repository structure

```
openPicDens/
    binarization.py   — image pre-processing and thresholding (Binarizer)
    analysis.py        — sliding-window scanning and porosity metrics (PICDens)
    io.py               — text-file I/O and RWL chronology export
    config.py           — default paths, filenames, and constants
    main.py             — runnable usage examples for all workflows
unet/
    dataset.py          — PyTorch Dataset for image/mask pairs
    preprocessing.py    — tiling large images and masks for training
    train.py            — U-Net training loop
    predict.py          — single-tile inference
    utils.py            — image splitting/stitching for full-size prediction
```

## Publications

This software was developed and used in the following studies:

- Khudykh, T. A., Belokopytova, L. V., Yang, B., Kholdaenko, Y. A., Babushkina, E. A., & Vaganov, E. A. (2024). New Methods in Digital Wood Anatomy: The Use of Pixel-Contrast Densitometry with Example of Angiosperm Shrubs in Southern Siberia. *Biology*, 13(4), 223. <https://doi.org/10.3390/biology13040223>
- Khudykh, T. A., Belokopytova, L. V., Kholdaenko, Y. A., Karmanovskaya, N. V., Portnyagin, D. G., Babushkina, E. A., & Vaganov, E. A. (2026). Tree-ring width and wood porosity of *Pinus sylvestris* L. and *Populus tremula* L. show different climate responses in southern Siberia. *Dendrochronologia*, 126502. <https://doi.org/10.1016/j.dendro.2026.126502>
- Khudykh, T. A., Belokopytova, L. V., Babushkina, E. A., & Vaganov, E. A. (2025, August). Wood Porosity: A Universal Parameter for Dendroclimatic Analysis. In *Doklady Biological Sciences* (Vol. 523, No. 1, pp. 204–208). Moscow: Pleiades Publishing. <https://doi.org/10.1134/S0012496625600150>

If you use OpenPiCDens in your research, please cite the relevant publication(s) above.

## License

See repository for license details.