"""
support functions
"""

import numpy as np
import pandas as pd

def pad_dict_list(dict_list: dict, padel=np.nan) -> dict:
    """
    fills short lists with padels
    
    Parameters
    ----------
    dict_list : dict
    padel

    Returns
    -------
    dict_list: dict
    """
    lmax = 0
    for lname in dict_list.keys():
        lmax = max(lmax, len(dict_list[lname]))
    for lname in dict_list.keys():
        ll = len(dict_list[lname])
        if  ll < lmax:
            dict_list[lname] += [padel] * (lmax - ll)
    return dict_list

def sma(series: list, interv: int=20) -> list:
    """
    Simple Moving Average

    Parameters
    ----------
    series : list
    interv: int

    Returns
    -------
    smaSeries: list
    """
    smaSeries = []
    for d in range(interv, len(series)):
        avg = sum(series[d-interv:d]) / interv
        smaSeries.append(avg)
    return smaSeries

def mathRound(n: int | float) -> int:
    """
    mathematical rounding

    Parameters
    ----------
    n : int | float

    Returns
    -------
    n: int
    """
    if n - int(n) > 0.5:
        return int(n)+1
    else:
        return int(n)

def smaDF(df: pd.DataFrame) -> pd.DataFrame:
    """
    smoothing of all curves in the dataframe

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    smaDF: pd.DataFrame
    """
    newDict = {}
    for c in df.columns:
        newDict.update({c:sma(df[c].tolist())})
    smaDF = pd.DataFrame(newDict)
    return smaDF