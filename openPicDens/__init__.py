"""
OpenPiCDens — pixel-contrast densitometry for wood micrographs.

Typical usage::

    from openPicDens import Binarizer, PICDens

    bi = Binarizer(method="Otsu", blur="Median")
    scan = PICDens(binarizer=bi, save_path="results/", ...)
    scan.startScan()
"""

from .analysis import PICDens
from .binarization import Binarizer
from .io import read_df, rw2rwl, save_df, save_list

__all__ = [
    # Core API — used in every scan
    "PICDens",
    "Binarizer",
    # I/O helpers — useful when working with results programmatically
    "read_df",
    "save_df",
    "save_list",
    "rw2rwl",
]