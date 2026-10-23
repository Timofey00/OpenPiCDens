"""
File I/O helpers for OpenPiCDens.

Covers reading/writing delimited text files and exporting
chronologies to Tucson dendrochronology (RWL) format.
"""

from __future__ import annotations

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


def save_df(data: pd.DataFrame, file_path: str, sep: str = "\t", index: bool = False) -> None:
    """Save *data* to a delimited text file.

    Parameters
    ----------
    data : pd.DataFrame
    file_path : str
    sep : str
        Column separator (default: tab).
    index : bool
        Whether to write the DataFrame index as a column (default:
        ``False``). Writing the index produces an unnamed leading
        column that pandas re-reads as ``"Unnamed: 0"`` — if that file
        is later fed into :func:`rw2rwl`, the index gets treated as an
        extra tree/core series. Keep this ``False`` unless the index
        carries meaningful data you intend to preserve.
    """
    data.to_csv(file_path, sep=sep, index=index)


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
    oldest year first: the first row from the series' first year to the end of
    its decade, then one row per decade.

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
        # Skip pandas' auto-generated index column ("Unnamed: 0", etc.)
        if str(col).startswith("Unnamed"):
            continue

        raw_values = [
            int(v)
            for v in data[col].tolist()
            if str(v) != "nan"
        ]
        if not raw_values:
            continue

        col_str = str(col).replace(" ", "")
        start_year = end_year - len(raw_values) + 1

        # Build year → value mapping (rows are ordered newest → oldest)
        year_val: dict[int, int] = {
            end_year - i: raw_values[i]
            for i in range(len(raw_values))
        }

        block_lines: list[str] = []
        current_year = start_year

        while current_year <= end_year:
            row_start = current_year
            # First row ends at the decade boundary; subsequent rows are full decades
            if current_year == start_year and current_year % 10 != 0:
                row_end = min((current_year // 10) * 10 + 9, end_year)
            else:
                row_end = min(current_year + 9, end_year)

            year_chunk = [str(year_val[y]) for y in range(row_start, row_end + 1)]
            values_str = "".join(f"{v:>6}" for v in year_chunk)
            line = f"{col_str:<8}{row_start}{values_str}"
            if row_end == end_year and len(year_chunk) < 10:
                line += f"{'-9999':>6}"
            block_lines.append(line)
            if row_end == end_year and len(year_chunk) == 10:
                block_lines.append(f"{col_str:<8}{end_year + 1}{'-9999':>6}")
            current_year = row_end + 1

        rwl_lines.extend(block_lines)

    rwl_text = "\n".join(rwl_lines) + "\n"
    with open(save_path, "w") as fh:
        fh.write(rwl_text)
    return rwl_text