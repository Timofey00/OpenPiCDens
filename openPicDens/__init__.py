"""
OpenPiCDens — pixel-contrast densitometry for wood micrographs.

Typical usage::

    from openPicDens import Binarizer, PICDens

    bi = Binarizer(method="Otsu", blur="Median")
    scan = PICDens(binarizer=bi, save_path="results/", ...)
    scan.startScan()
"""

from .analysis import PICDens
from .binarization import Binarizer, binarize_dir_all_methods, binarize_image_all_methods
from .io import read_df, rw2rwl, save_df, save_list

__all__ = [
    # Core API — used in every scan
    "PICDens",
    "Binarizer",
    # Comparing binarization methods
    "binarize_image_all_methods",
    "binarize_dir_all_methods",
    # I/O helpers — useful when working with results programmatically
    "read_df",
    "save_df",
    "save_list",
    "rw2rwl",
]