"""
support functions
"""

import numpy as np
import pandas as pd
import os
from math import floor
from scipy.interpolate import interp1d

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

def sma(series: list, smaInterval: int=20) -> list:
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
    for d in range(smaInterval, len(series)):
        avg = sum(series[d-smaInterval:d]) / smaInterval
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

def smaDF(df: pd.DataFrame, smaInterval) -> pd.DataFrame:
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
        newDict.update({c:sma(df[c].tolist(), smaInterval)})
    smaDF = pd.DataFrame(newDict)
    return smaDF

def rw2rwl(data : pd.DataFrame, savePath: str, fileName: str, end_year: int=2022, coef=1) -> str:
    """
    saves the dataFrame in rwl format

    Parameters
    ----------
    data : pd.DataFrame
        saving data
    savePath: str
        path to save data
    fileName : str
        file name
    end_year : int
        year end
    coef : int
        multiplier
    
    Returns
    -------
    rwl_text: str
        rwl data
    """
    ext = '.rwl'
    fileName = fileName + ext
    savePath = os.path.join(savePath, fileName).replace('\\', '/')
    # print(savePath)
    data = data * coef
    data = data.replace(-1000, -1)
    # Convert rw file in rwl-format
    rwl_text = ''
    for col in data.columns[:].sort_values():
        list_strings = list(filter(lambda x: str(x)!='nan', data[col].tolist()))
        list_strings = [str(int(n)) for n in list_strings]
        
        start_year = end_year - len(list_strings) + 1
        end_dec_year = end_year
        start_dec_year = int(floor(end_dec_year / 10) * 10)
        
        while list_strings:
            year_str = ''
            for y in range(start_dec_year, end_dec_year+1):
                new_year = list_strings.pop(0)
                new_year = ' ' * (6 - len(new_year)) + new_year
                year_str = new_year + year_str

            new_str = f'{col.replace(" ", "")}' + ' ' * (8-len(col.replace(" ", ""))) + f'{start_dec_year}{year_str}'
            if end_dec_year == end_year:
                new_str += ' -9999'
            new_str += '\n'
            rwl_text = new_str + rwl_text
            # print(rwl_text)

            end_dec_year = start_dec_year - 1
            start_dec_year -= 10
            if start_dec_year < start_year:
                start_dec_year = start_year
    with open(savePath, 'w') as f:
        f.write(rwl_text)
        f.close()
    return rwl_text


def getNormalisationPorosityProfile(porosityProfile: list, reqLen: int, interpolationtype='cubic') -> list:
    """normalizes the porosity profile along the length in accordance with the conversion coefficient

    Parameters
    ----------
    porosityProfile : List
        porosity profile
    reqLen : int
        required porosity profile length
    
    Returns
    -------
    normProfile : List
    """

    original_length = len(porosityProfile)
    x_old = np.linspace(0, 1, original_length)
    x_new = np.linspace(0, 1, reqLen)

    f = interp1d(x_old, porosityProfile, kind=interpolationtype)  # Можно попробовать 'cubic' для сглаживания
    normProfile = list(f(x_new))

    return normProfile



