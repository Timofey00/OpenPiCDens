"""
Numeric and filesystem utilities for OpenPiCDens.

This module covers:
- list/DataFrame padding and rounding
- smoothing (SMA, Savitzky-Golay)
- porosity-profile normalisation via interpolation
- directory-tree helpers
"""

from __future__ import annotations

import os
from math import floor
from statistics import mean, median

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def pad_dict_list(dict_list: dict, padel=np.nan) -> dict:
    """Pad all lists in *dict_list* to the length of the longest one.

    Parameters
    ----------
    dict_list : dict
        Dictionary whose values are lists.
    padel :
        Fill value (default: ``np.nan``).

    Returns
    -------
    dict
        The same dictionary with all lists padded to equal length.
    """
    if not dict_list:
        return dict_list
    max_len = max(len(v) for v in dict_list.values())
    for key, lst in dict_list.items():
        shortage = max_len - len(lst)
        if shortage:
            dict_list[key] = lst + [padel] * shortage
    return dict_list


def mathRound(n: int | float) -> int:
    """Round *n* using standard mathematical rounding (0.5 rounds up).

    Python's built-in ``round()`` uses banker's rounding; this function
    always rounds half-up.
    """
    return int(n) + 1 if n - int(n) > 0.5 else int(n)


# ---------------------------------------------------------------------------
# Smoothing
# ---------------------------------------------------------------------------

def sma(series: list, sma_interval: int = 20) -> list:
    """Compute a Simple Moving Average of *series*.

    Parameters
    ----------
    series : list
    sma_interval : int
        Window size.

    Returns
    -------
    list
        Smoothed series of length ``len(series) - sma_interval``.
    """
    return [
        sum(series[i - sma_interval:i]) / sma_interval
        for i in range(sma_interval, len(series))
    ]


def smaDF(df: pd.DataFrame, sma_interval: int, smooth_type: str = "sma") -> pd.DataFrame:
    """Smooth every column in *df*.

    Parameters
    ----------
    df : pd.DataFrame
    sma_interval : int
        Window size.
    smooth_type : str
        ``"sma"`` for Simple Moving Average, ``"sovgol"`` for
        Savitzky–Golay filter (polynomial order 3).

    Returns
    -------
    pd.DataFrame
    """
    dispatch = {
        "sma": lambda col: sma(col, sma_interval),
        "sovgol": lambda col: savgol_filter(col, sma_interval, 3).tolist(),
    }
    if smooth_type not in dispatch:
        raise ValueError(f"Unknown smooth_type: {smooth_type!r}")

    return pd.DataFrame({
        col: dispatch[smooth_type](df[col].tolist())
        for col in df.columns
    })


# ---------------------------------------------------------------------------
# Porosity profile normalisation
# ---------------------------------------------------------------------------

def getNormalisationPorosityProfile(
    porosity_profile: list,
    req_len: int,
    interpolation_type: str = "cubic",
) -> list:
    """Resample *porosity_profile* to *req_len* points via interpolation.

    Parameters
    ----------
    porosity_profile : list
        Raw porosity profile.
    req_len : int
        Desired length of the output profile.
    interpolation_type : str
        Passed directly to :func:`scipy.interpolate.interp1d`
        (default: ``"cubic"``).

    Returns
    -------
    list
        Resampled profile of length *req_len*.
    """
    x_old = np.linspace(0, 1, len(porosity_profile))
    x_new = np.linspace(0, 1, req_len)
    f = interp1d(x_old, porosity_profile, kind=interpolation_type)
    return list(f(x_new))


# ---------------------------------------------------------------------------
# Path / directory utilities
# ---------------------------------------------------------------------------

def initPath(path: str) -> None:
    """Create *path* (and any missing parents) if it does not exist."""
    os.makedirs(path, mode=0o754, exist_ok=True)


def initDirTree(root: str, dirs: dict) -> None:
    """Recursively create a directory tree described by *dirs*.

    The tree format is a dictionary where each key is a directory name
    and its value is a list of sub-directories (which can themselves be
    dictionaries for deeper nesting)::

        {
            "dir_a": ["sub_1", "sub_2", {"sub_3": ["leaf_1"]}],
            "dir_b": [],
        }

    Parameters
    ----------
    root : str
        Base path under which the tree is created.
    dirs : dict
    """
    initPath(root)
    for name, children in dirs.items():
        sub_path = os.path.join(root, name)
        initPath(sub_path)
        for child in children:
            if isinstance(child, dict):
                initDirTree(sub_path, child)
            else:
                initPath(os.path.join(sub_path, child))


def initResultsPathsFromImages(root: str, trees_path: str) -> None:
    """Create the standard output directory tree for a scan run.

    Parameters
    ----------
    root : str
        Root of the results directory.
    trees_path : str
        Directory containing per-tree micrograph sub-directories.
    """
    tree_names = sorted(os.listdir(trees_path))
    dir_tree = {
        "areaPorosity": tree_names,
        "naturalValuesPorosity": [],
        "normValuesPorosity": [],
        "rawPorosity": [],
        "rwl": [
            "EW", "EWPOR", "LWPOR", "maxPorosity", "maxPorosityQ",
            "meanPorosity", "meanPorosityQ", "minPorosity",
            "minPorosityQ", "sectors", "rawPorosity",
        ],
        "sectorsPorosity": [],
    }
    initDirTree(root, dir_tree)


def getTreeDirs(trees_path: str) -> dict[str, list[str]]:
    """Build a mapping of tree sub-directory names → sorted image filenames.

    Parameters
    ----------
    trees_path : str
        Directory containing per-tree sub-directories (integer-named).

    Returns
    -------
    dict[str, list[str]]
        ``{tree_id: [img_filename, ...]}`` sorted numerically.
    """
    sub_dir_names = sorted(os.listdir(trees_path), key=int)
    return {
        d: sorted(
            os.listdir(os.path.join(trees_path, d)),
            key=lambda f: int(f.split(".")[0]),
        )
        for d in sub_dir_names
    }