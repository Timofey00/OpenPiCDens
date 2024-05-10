import cv2
import numpy as np
import pandas as pd
import os
from statistics import mean

from settings import pix_to_mcm


def find_raw_por(binary_img):
    """Return raw porosity, where x_axis is numbers of pixels

    Parameters
    ----------
    binary_img : np.ndarray            
    """
    por = []
    binary_img = binary_img.transpose()
    x_img_size = len(binary_img)
    y_img_size = len(binary_img[0])
    for x in range(x_img_size):
        b_pix = np.sum(binary_img[x] == 255)
        por.append(b_pix / y_img_size)
    return por

def norm_por(por: list, convert_coef=pix_to_mcm):
    """normalizes the porosity profile along the length in accordance with the conversion coefficient

    Parameters
    ----------
    por : List
    convert_coef : int        
    """
    norm_por = []
    size_por = len(por)
    req_len = len(por) * convert_coef
    step_sum = 0

    if convert_coef < 1:
        if convert_coef == pix_to_mcm:
            req_len += 20
            convert_coef = req_len / size_por
        step = 1 / convert_coef
        while step_sum < size_por-step:
            n = int(round(step, 0))
            cur_base_index = int(round(step_sum, 0))
            norm_por.append(sum((por[cur_base_index + i] for i in range(n))) / n)
            step_sum += step
            if step_sum > size_por:
                step_sum = size

    elif convert_coef > 1:
        step = size_por / (req_len - size_por)
        i = 0
        rb = size_por-step - 1 
        while rb < size_por:
            lb = int(round(step_sum, 0))
            rb = int(round(step_sum + step, 0))
            if rb == lb :
                pre_part = []
                rb += 1
            elif rb > size_por:
                rb = size_por
            else:
                pre_part = por[lb: rb]

            cur_sublist = [mean(por[lb: rb])]
            norm_por += (pre_part + cur_sublist)

            step_sum += step
    else:
        return por

    return norm_por

def multi_window_scan(img_path: str, window_size: int = 1000, step: int= 200):
    """scans the image in several areas (size areas - window_size, step between areas - step)
    return dataframe with porosity profiles from this areas

    Parameters
    ----------
    img_path : str
    window_size : int
    step : int            
    """
    b_i = get_binary_img(img_path)
    data = {}
    for i in range(int((len(b_i)//step))-int((window_size/step))):
        por = scan_img_part(b_i[i*step:i*step+window_size])
        data.update({i:sma(por)})
    df = pd.DataFrame(data=data)
    return df

def scan_img_part(b_i_area: np.ndarray):
    """finds the porosity profile from a given area of a binary image

    Parameters
    ----------
    b_i_area : np.ndarray              
    """
    raw_por = find_raw_por(b_i_area)
    por = norm_por(raw_por)
    return por

def get_average_por(por_df: pd.DataFrame):
    """finds the average porosity profile from several
    
    Parameters
    ----------
    por_df : pd.DataFrame           
    """
    avg_por = {'average': []}
    l = len(por_df)
    avg_df = pd.DataFrame(data=por_df.mean(axis=1), columns=['average por'])
    return avg_df

def pad_dict_list(dict_list: dict, padel=np.nan):
    """fills short lists with padels
    
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

def get_binary_img(
    img_path: str, 
    blur_type: str='Gaussian', 
    th_method: str='Otsu', 
    gamma_eq: bool=False, 
    ksize: int=7, 
    th_up: int=0):
    """converts the image into binary format 
    (using the Otsu method)
    
    Parameters
    ----------
    img_path : str
    blur_type: str
        Choises: Gaussian (default), Median, Normalized Box Filter (NBF), Bilateral
    th_method: str
        Choises: Otsu, Mean, Gaussian
    gamma_eq: bool
    ksize: int
        Kernel size
    th_up: int
        value by which the threshold is raised, obtained using the Otsu method

    Returns
    -------
    bi_img: np.ndarray
    """
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    if gamma_eq:
        img = cv2.equalizeHist(img)

    # Choice bluring method
    if blur_type == 'Gaussian':
        img = cv2.GaussianBlur(img, (ksize, ksize), 0)
    elif blur_type == 'Median':
        img= cv2.medianBlur(img, ksize)
    elif blur_type == 'NBF':
        img = cv2.blur(img, (ksize, ksize)) 
    elif blur_type == 'Bilateral':
        img = cv2.bilateralFilter(img, 11, 41, 21)

    # Choice thresholding method
    if th_method == 'Otsu':
        th, bi_img = cv2.threshold(img, 230, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        if th_up:
        _, bi_img = cv2.threshold(img, th+th_up, 255, cv2.THRESH_BINARY)
    elif th_method == 'Mean':
        bi_img = cv2.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY,11,2)
    elif th_method == 'Gaussian':
        bi_img = cv2.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY,11,2)
    return bi_img

def sma(series, interv=20):
    new_data = []
    for d in range(interv, len(series)):
        sma_d = sum(series[d-interv:d]) / interv
        new_data.append(sma_d)
    return new_data