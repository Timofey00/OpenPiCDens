"""module that contains utilities for automating PiC densitometry

"""

import numpy as np
import pandas as pd
import os
from statistics import mean

from .measure import *

def get_rw(
    por_df: pd.DataFrame, 
    name_file: str='rw', 
    ext: str='txt', 
    save_dir: str='rw'):
    """return tree ring width chronology
    
    Parameters
    ----------
    por_df : pd.DataFrame
    name_file : str
    ext : str
        extension
    save_dir : str
        directory to save

    Returns
    -------
    trw_df: pd.DataFrame
    """
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir, 0o754)
    trw = {'trw': []}
    for col in por_df.columns:
        trw['trw'].append(por_df[col].count())
    trw_df = pd.DataFrame(data=trw)
    trw_df.to_csv(f'{save_dir}/{name_file}.{ext}', sep='\t', index=False)
    return trw_df

def get_mxmn_por(
    por_df: pd.DataFrame, 
    mod='mxmn', 
    names_files={'mxp': 'mxp', 'mnp':'mnp'},
    ext: str='txt', 
    save_dir: str='mxmn'
    ):
    """return max, min and mean porosity profile chronology
    
    Parameters
    ----------
    por_df : pd.DataFrame
    mod : str
    names_files : dict
    ext : str
        extension
    save_dir : str
        directory to save

    Returns
    -------
    mxp_df : pd.DataFrame
    mnp_df : pd.DataFrame
    """

    if not os.path.isdir(save_dir):
        os.mkdir(save_dir, 0o754)

    max_por = {'mxp': []}
    min_por = {'mnp': []}
    for col in por_df.columns:
        max_por['mxp'].append(por_df[col].max())
        min_por['mnp'].append(por_df[col].min())
    if mod == 'mxmn':
        mxp_df = pd.DataFrame(data=max_por)
        mnp_df = pd.DataFrame(data=min_por)
        mxp_df.to_csv(f'{save_dir}/{names_files['mxp']}.{ext}', sep='\t', index=False)
        mnp_df.to_csv(f'{save_dir}/{names_files['mnp']}.{ext}', sep='\t', index=False)
        return mxp_df, mnp_df

def scan_all_kind_dir(dir_path: str):
    """determines the porosity profile for all trees in the library, 
    as well as their other characteristics (TRW, MaxPor, MinPor, MeanPor)
    
    Parameters
    ----------
    dir_path : str

    Returns
    -------
    all_trees : List
    """
    tree_dirs = [dir_path + d for d in sorted(os.listdir(dir_path), key=lambda f: int(f))]
    all_trees = []
    for td in tree_dirs:
        print(os.path.split(td)[-1])
        print(td)
        prename_file = os.path.split(td)[-1]
        tree_df = scan_dir(td+'/', name_file=f"{prename_file} por all year")
        get_mxmn_por(tree_df, names_files={'mxp': f'{prename_file} mxp', 'mnp':f'{prename_file} mnp'})
        get_rw(tree_df, name_file=f'{prename_file} rw')
        all_trees.append(tree_df)
    return all_trees

def norm_por_df(
    por_df: pd.DataFrame,
    name_file: str='norm_por',
    ext: str='txt', 
    save_dir: str='norm_porosity',
    ):
    """normalizes all density profiles (by the length of the smallest ring)
    
    Parameters
    ----------
    por_df : pd.DataFrame

    Returns
    -------
    all_trees : List
    """
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir, 0o754)

    clear_por = {}
    min_rw = 10000
    for col in por_df.columns:
        clear_por.update({col: list(filter(lambda x: str(x)!='nan', por_df[col].tolist()))})
        if len(clear_por[col]) == 0:
            clear_por.pop(col)
            continue
        if min_rw > len(clear_por[col]):
            min_rw = len(clear_por[col])
    n_p = {}
    for cp in clear_por:
        n_p.update({cp: norm_por(clear_por[cp], min_rw/len(clear_por[cp]))})
        if len(n_p[cp]) < min_rw:
            min_rw = len(n_p[cp])
    for n in n_p:
        if len(n_p[n]) > min_rw:
            n_p[n] = n_p[n][:min_rw]

    n_p_df = pd.DataFrame(data=n_p)
    n_p_df.to_csv(f'{save_dir}/{name_file}.{ext}', sep='\t', index=False)
    return n_p_df

def get_long_norm_por(dir_path: str):
    """create a normalized long-term porosity profile
    
    Parameters
    ----------
    dir_path : str

    Returns
    -------

    """
    all_trees = scan_all_kind_dir(dir_path)
    files_names = sorted(os.listdir(dir_path), key=lambda f: int(f))
    all_years = {y: {} for y in range(1982, 2023)}

    long_norm_por = {f: [] for f in files_names}
    mxn = {'mxn': []}

    for y in range(1982, 2023):
        i = 0
        for df in all_trees:
            # print(df.columns.tolist())
            if y in df.columns.tolist():
                l = list(filter(lambda x: x!= np.nan, df[y].tolist()))
                all_years[y].update({files_names[i]: l})
            i+= 1
    all_df = []
    for y in all_years:
        print(y)
        d = pad_dict_list(all_years[y])
        df = pd.DataFrame(d)
        n_p = norm_por_df(df)
        all_df.append(n_p)

        # n_p.to_csv(f'{y} n_p.txt', sep='\t', index=False)
    pd.concat(all_df).to_csv(f'long norm porosity.txt', sep='\t', index=False)

def scan_dir(
    dir_path: str, 
    name_file: str='por all year', 
    year_start: int=2022,
    ext: str='txt', 
    save_dir: str='multiscan',

    ):
    """scans a directory with individual trees
    
    Parameters
    ----------
    dir_path : str
    name_file: str, optional
    year_start: int, optional       
    """
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir, 0o754)
    files_list = sorted(os.listdir(dir_path), key=lambda f: int(f.split('.')[0]))
    por_all_year = {}
    y = 0
    for f in files_list:
        img_path = dir_path + f
        por_df = multi_window_scan(img_path)
        por_df.to_csv(f'{save_dir}/{f} multiscan.txt', sep='\t', index=False)
        avg_por = get_average_por(por_df)
        por_all_year.update({year_start-y: avg_por['average por'].values.tolist()})
        y += 1
    por_all_year = pad_dict_list(por_all_year)
    por_all_year_df = pd.DataFrame(data=por_all_year)
    por_all_year_df.to_csv(f'{save_dir}/{name_file}.{ext}', sep='\t', index=False)
    return por_all_year_df