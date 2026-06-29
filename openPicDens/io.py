"""
File I/O helpers for OpenPiCDens.

Covers reading/writing delimited text files and exporting
chronologies to Tucson dendrochronology (RWL) format.
"""

from __future__ import annotations

from math import floor

import pandas as pd


# ---------------------------------------------------------------------------
# Delimited text files
# ---------------------------------------------------------------------------

def read_df(file_path: str, sep: str = "\t") -> pd.DataFrame:
    """Read a delimited text file into a DataFrame.

    Parameters
    ----------
    file_path : str
    sep : str
        Column separator (default: tab).
    """
    return pd.read_csv(file_path, sep=sep)


def save_df(data: pd.DataFrame, file_path: str, sep: str = "\t") -> None:
    """Save *data* to a delimited text file.

    Parameters
    ----------
    data : pd.DataFrame
    file_path : str
    sep : str
        Column separator (default: tab).
    """
    data.to_csv(file_path, sep=sep)


def save_list(
    data: list,
    file_path: str,
    col_name: str = "x",
    sep: str = "\t",
) -> None:
    """Wrap *data* in a single-column DataFrame and save it.

    Parameters
    ----------
    data : list
    file_path : str
    col_name : str
        Column header (default: ``"x"``).
    sep : str
        Column separator (default: tab).
    """
    save_df(pd.DataFrame({col_name: data}), file_path, sep)


# ---------------------------------------------------------------------------
# RWL export
# ---------------------------------------------------------------------------

def rw2rwl(
    data: pd.DataFrame,
    save_path: str,
    end_year: int = 2022,
    coef: int = 1,
) -> str:
    """Convert a DataFrame to Tucson dendrochronology (RWL) format and save it.

    Each column is treated as one tree/core series.  Values are written
    in decade blocks from the most-recent year backwards.

    Parameters
    ----------
    data : pd.DataFrame
        Ring-width data; each column is a series ID, rows are yearly
        values ordered newest → oldest.
    save_path : str
        Destination file path.
    end_year : int
        Most recent calendar year in the data (default: ``2022``).
    coef : int
        Multiplier applied to all values before writing (e.g. ``1000``
        converts fractional porosity to per-mille integers).

    Returns
    -------
    str
        The full RWL text written to disk.
    """
    data = (data * coef).replace(-1000, -1)
    rwl_lines: list[str] = []

    for col in sorted(data.columns):
        values = [
            str(int(v))
            for v in data[col].tolist()
            if str(v) != "nan"
        ]
        if not values:
            continue

        start_year = end_year - len(values) + 1
        end_dec = end_year
        start_dec = (
            int(floor(end_dec / 10) * 10)
            if len(values) > 6
            else end_dec - len(values) + 1
        )
        remaining = list(values)
        col_str = col.replace(" ", "")
        block_lines: list[str] = []

        while remaining:
            year_chunk = []
            for _ in range(start_dec, end_dec + 1):
                year_chunk.insert(0, remaining.pop(0))

            values_str = "".join(f"{v:>6}" for v in year_chunk)
            line = f"{col_str:<8}{start_dec}{values_str}"
            if end_dec == end_year:
                line += "  -9999"
            block_lines.insert(0, line)

            end_dec = start_dec - 1
            start_dec = max(start_year, start_dec - 10)

        rwl_lines.extend(block_lines)

    rwl_text = "\n".join(rwl_lines) + "\n"
    with open(save_path, "w") as fh:
        fh.write(rwl_text)
    return rwl_text