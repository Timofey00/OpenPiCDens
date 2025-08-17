"""
Main App-module
"""

import os
from statistics import mean, median, stdev
import pathlib

import numpy as np
import pandas as pd
import cv2
import torch
import math
from scipy.interpolate import interp1d

from .utils import *
from .config import *
from unet.utills import predictImgMask
from itertools import groupby


class BI:
    """
    Parameters
    ----------
    imgPath : str
        path to image
    blurType: str
        Choises: Gaussian (default), Median, Normalized Box Filter (NBF), Bilateral (default=Median)
    biMethod: str
        Choises: Otsu, Mean, Gaussian, UNET (default=Otsu)
    gammaEqualisation: bool
        if True, the image is gamma-corrected (default=False)
    ksize: int
        Kernel size. Takes only odd values (default=3)
    constantTh : int
        Only if none of the binarization methods is selected. 
        Binarization is performed using a fixed threshold value constantTh (default=235)

    Methods
    -------
    binarizationRoot(root: str, saveDir: str) -> None
    getBinaryImg(imgPath: str) -> np.ndarray

    """
    def __init__(
        self,
        blurType: str = 'Median',  
        gammaEqualisation: bool = False, 
        ksize: int = 3,
        constantTh: int = 235,
        modelPath: None = None,
        biMethod: str = 'Otsu'
        ):

        self.blurType = blurType
        self.gammaEqualisation = gammaEqualisation
        self.ksize = ksize
        self.constantTh = constantTh
        self.biMethod = biMethod
        
        if self.biMethod == 'UNET':
            DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
            self.model = torch.load(modelPath).to(DEVICE)

    def binarizationRoot(self, root: str, saveDir: str) -> None:
        sortedNames = sorted(os.listdir(root), key=lambda f: int(f))
        dirs = [os.path.join(root, f'{d}/') for d in sortedNames]

        for d in dirs:
            imgsList = sorted(os.listdir(d), key=lambda f: int(f.split('.')[0]))    
            imgsPathList = [os.path.join(d, i) for i in imgsList]

            for i in imgsPathList:
                biImg = self.getBinaryImg(i)
                names = i.split('/')
                savePath = os.path.join(saveDir, names[-2])
                if not os.path.isdir(savePath):
                    os.mkdir(savePath, 0o754)
                savePath = os.path.join(savePath, names[-1])
                cv2.imwrite(savePath, biImg)

    def getBinaryImg(self, imgPath: str) -> np.ndarray:
        """
        Parameters
        ----------
        imgPath : str
            path to image

        converts the image into binary format 
        (default, using the Otsu method)
        
        Returns
        -------
        biImg: np.ndarray
        """
        img = cv2.imread(imgPath)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        if self.gammaEqualisation:
            img = cv2.equalizeHist(img)

        # Choice bluring method
        if self.blurType == 'Gaussian':
            img = cv2.GaussianBlur(img, (self.ksize, self.ksize), 0)
        elif self.blurType == 'Median':
            img= cv2.medianBlur(img, self.ksize)
        elif self.blurType == 'NBF':
            img = cv2.blur(img, (self.ksize, self.ksize)) 
        elif self.blurType == 'Bilateral':
            img = cv2.bilateralFilter(img, 11, 41, 21)
        else:
            pass

        # Choice thresholding method
        if self.biMethod == 'Otsu':
            th, biImg = cv2.threshold(img, 230, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        elif self.biMethod == 'Mean':
            biImg = cv2.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY,11,2)
        elif self.biMethod == 'Gaussian':
            biImg = cv2.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY,11,2)
        elif self.biMethod == 'Triangle':
            thresh = filters.threshold_triangle(img)
            biImg = img > thresh
            biImg = biImg.astype(np.uint8) * 255
        elif self.biMethod == 'UNET':
            biImg = predictImgMask(imgPath=imgPath, saveMaskPath=None, divideSize=128, model=self.model)
            biImg = cv2.cvtColor(biImg, cv2.COLOR_RGB2GRAY)
            th, biImg = cv2.threshold(biImg, 200, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        else:
            th, biImg = cv2.threshold(img, constantTh, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        return biImg


class PICDens():
    """
    A main class of app

    Attributes
    ----------
    savePath : str
        path to save data
    root : str
        path to image directory
    pixToMcmCoef : float
        coefficient for converting porosity length to micrometers (the default is 1)
    speciesName : str
        species name
    yearStart : int
        start year
    normNumber : int
        length of normalized porosity profile
    normMethod : str
        normalization method (the default is 'median')
    smaInterval : int
        smoothing window size (the default is 90)
    gapValue : int | None
        the size of the filter gap window. If set to None, the filter is not applied
        (default is None)


    biMethod : str
        method of binarization
    blurType : str
        Blur type. Choises: Gaussian (default), Median, Normalized Box Filter (NBF), Bilateral. (Default: Median)
    gammaEqualisation : bool
    ksize : int
        Kernel size. Takes only odd values (default=3)
    constantTh
    usePredBIImgs: bool
        set this variable to true if you have pre-binarized images (default is False)
    biImgsPath : str
        if the variable usePredBIImgs is True, the program uses this path as a source of binarized images. 
        In other case, this path is used as a directory for saving binarized images
        (default is False)
    modelPath : str | bool

    Methods
    -------
    startScan() -> None
    scanSubDir(subDir: str, imgsList: list) -> pd.DataFrame
    getLongPorosityProfile(porosityDict: dict, normMethod: str) -> tuple
    getNormPorosityDF(porosityDF: pd.DataFrame, normMethod: str="median") -> pd.DataFrame
    getPorProfilesNaturalValues(porosityProfiles: pd.DataFrame) -> pd.DataFrame
    getNormPorosityProfiles(porosityProfiles: pd.DataFrame) -> pd.DataFrame
    getRW(porosityProfiles: pd.DataFrame) -> List
    getPorosityCharacteristics(porosityProfiles: pd.DataFrame) -> tuple
    getPorosityCharacteristicsProcentile(porosityProfiles: pd.DataFrame) -> tuple
    getEarlyLateWidth(porosityProfiles: pd.DataFrame) -> tuple
    getSectorPorosity(porosityProfiles: pd.DataFrame, sectorsNumber: int=10) -> tuple
    scanImg(imgPath: str, windowSize: int = 1000, step: int= 200) -> pd.DataFrame
    getPorosityProfileToPix(biImg: np.ndarray) -> list
    gapFilter(x: np.array) -> list
    saveConfigFile() -> None
    getSD -> tuple
    
    """
    def __init__(
        self, 
        savePath: str,
        root: str,
        speciesName: str,
        yearStart: int,
        normNumber: int,
        pixToMcmCoef: float|int = 1,
        normMethod: str="median",
        smaInterval: int=90,
        gapValue=None,
        
        biMethod: str='Otsu',
        blurType: str = 'Median',  
        gammaEqualisation: bool = False, 
        ksize: int = 3,
        constantTh: int = 235,
        modelPath: None = None,
        usePredBIImgs=False,
        biImgsPath: str=False,
        ):

        # main settings

        self.savePath = savePath
        self.root = root
        self.speciesName = speciesName
        self.yearStart = yearStart
        self.normNumber = normNumber
        self.pixToMcmCoef = pixToMcmCoef
        self.normMethod = normMethod
        self.smaInterval = smaInterval
        self.gapValue = gapValue

        # BI settings

        self.biMethod = biMethod
        self.blurType = blurType
        self.gammaEqualisation = gammaEqualisation
        self.ksize = ksize
        self.constantTh = constantTh
        self.modelPath = modelPath
        self.usePredBIImgs = usePredBIImgs
        self.biImgsPath = biImgsPath

        if self.usePredBIImgs:
            self.root = self.biImgsPath

        self.saveConfigFile()

    def startScan(self) -> None:
        """Performs an analysis of the root directory containing subdirectories with images"""

        # Init save paths
        treesDirs = getTreeDirs(treesPath=self.root)
        natPath = os.path.join(self.savePath, SAVE_PATHS["natural_path"])
        normPath = os.path.join(self.savePath, SAVE_PATHS["norm_path"])
        rawPath = os.path.join(self.savePath, SAVE_PATHS["raw_path"])
        rwPath = os.path.join(self.savePath, SAVE_NAMES["rw"])
        maxPPath = os.path.join(self.savePath, SAVE_NAMES["max"])
        minPPath = os.path.join(self.savePath, SAVE_NAMES["min"])
        meanPPath = os.path.join(self.savePath, SAVE_NAMES["mean"])
        maxPQPath = os.path.join(self.savePath, SAVE_NAMES["maxQ"])
        minPQPath = os.path.join(self.savePath, SAVE_NAMES["minQ"])
        meanPQPath = os.path.join(self.savePath, SAVE_NAMES["meanQ"])
        ewPath = os.path.join(self.savePath, SAVE_NAMES["ew"])
        lwPath = os.path.join(self.savePath, SAVE_NAMES["lw"])
        ewprPath = os.path.join(self.savePath, SAVE_NAMES["ewpr"])
        lwprPath = os.path.join(self.savePath, SAVE_NAMES["lwpr"])
        ewpPath = os.path.join(self.savePath, SAVE_NAMES["ewp"])
        lwpPath = os.path.join(self.savePath, SAVE_NAMES["lwp"])
        longPath = os.path.join(self.savePath, SAVE_NAMES["long"])
        avgPath = os.path.join(self.savePath, SAVE_NAMES["avg"])
        secDir = os.path.join(self.savePath, SAVE_PATHS["sec_path"])
        secPaths = [os.path.join(secDir, SAVE_NAMES[sec]) for sec in range(10)]

        rw_rwl_Path = os.path.join(self.savePath, 'rwl', SAVE_RWL_NAMES["rw"])
        max_rwl_Path = os.path.join(self.savePath, 'rwl', SAVE_RWL_NAMES["max"])
        min_rwl_Path = os.path.join(self.savePath, 'rwl', SAVE_RWL_NAMES["min"])
        mean_rwl_Path = os.path.join(self.savePath, 'rwl', SAVE_RWL_NAMES["mean"])
        maxq_rwl_Path = os.path.join(self.savePath, 'rwl', SAVE_RWL_NAMES["maxQ"])
        minq_rwl_Path = os.path.join(self.savePath, 'rwl', SAVE_RWL_NAMES["minQ"])
        meanq_rwl_Path = os.path.join(self.savePath, 'rwl', SAVE_RWL_NAMES["meanQ"])
        ew_rwl_Path = os.path.join(self.savePath, 'rwl', SAVE_RWL_NAMES["ew"])
        lw_rwl_Path = os.path.join(self.savePath, 'rwl', SAVE_RWL_NAMES["lw"])
        ewpr_rwl_Path = os.path.join(self.savePath, 'rwl', SAVE_RWL_NAMES["ewpr"])
        lwpr_rwl_Path = os.path.join(self.savePath, 'rwl', SAVE_RWL_NAMES["lwpr"])
        ewp_rwl_Path = os.path.join(self.savePath, 'rwl', SAVE_RWL_NAMES["ewp"])
        lwp_rwl_Path = os.path.join(self.savePath, 'rwl', SAVE_RWL_NAMES["lwp"])

        initResultsPathsFromImages(root=self.savePath, treesPath=self.root)

        # Init dicts for results
        rawPorosityDict = {}
        rwDict = {}
        maxPorosityDict = {}
        minPorosityDict = {}
        meanPorosityDict = {}
        maxPorosityQDict = {}
        minPorosityQDict = {}
        meanPorosityQDict = {}
        ewDict = {}
        lwDict = {}
        ewprDict = {}
        lwprDict = {}
        ewPorosityDict = {}
        lwPorosityDict = {}
        naturalPDict ={}
        normPDict =  {}
        sectorsPorosityDict = {sec: {} for sec in range(10)}

        for t in treesDirs:
            # Init save paths for individuals tress
            treePath = os.path.join(self.root, t)
            natTreePath = os.path.join(natPath, f"{t}.txt")
            normTreePath = os.path.join(normPath, f"{t}.txt")
            rawTreePath = os.path.join(rawPath, f"{t}.txt")

            # Scan subdir
            porosityByYearsDF = self.scanSubDir(subDir=treePath, imgsNames=treesDirs[t])
            porosityByYearsSMADF = smaDF(porosityByYearsDF, smaInterval=self.smaInterval)
            rawPorosityDict.update({t: porosityByYearsSMADF})

            # Get main features
            rw = self.getRW(porosityProfiles=porosityByYearsDF)
            maxP, minP, meanP = self.getPorosityCharacteristics(porosityProfiles=porosityByYearsSMADF)
            maxQP, minQP, meanQP = self.getPorosityCharacteristicsProcentile(porosityProfiles=porosityByYearsSMADF)
            ew, lw, ewpr, lwpr, meanPew, meanPlw = self.getEarlyLateWidth(porosityProfiles=porosityByYearsSMADF)
            sectP = self.getSectorPorosity(porosityProfiles=porosityByYearsSMADF, sectorsNumber=10)
            naturalP = self.getPorProfilesNaturalValues(porosityProfiles=porosityByYearsSMADF)
            normP = self.getNormPorosityProfiles(porosityProfiles=porosityByYearsSMADF)

            # Update res-dicts
            rwDict.update({t: rw})
            maxPorosityDict.update({t: maxP})
            minPorosityDict.update({t: minP})
            meanPorosityDict.update({t: meanP})
            maxPorosityQDict.update({t: maxQP})
            minPorosityQDict.update({t: minQP})
            meanPorosityQDict.update({t: meanQP})
            ewDict.update({t: ew})
            lwDict.update({t: lw})
            ewprDict.update({t: ewpr})
            lwprDict.update({t: lwpr})
            ewPorosityDict.update({t: meanPew})
            lwPorosityDict.update({t: meanPlw})
            naturalPDict.update({t: naturalP})
            normPDict.update({t: normP})

            for sec in range(len(sectorsPorosityDict)):
                sectorsPorosityDict[sec].update({t: sectP[sec]})

            # Saving individual results
            saveDFasTXT(data=naturalP, filePath=natTreePath, sep='\t')
            saveDFasTXT(data=normP, filePath=normTreePath, sep='\t')

        longPorosityProfilesDF, AVG = self.getLongPorosityProfile(porosityDict=rawPorosityDict, normMethod="median")

        # Pad 
        rwDict = pad_dict_list(dict_list=rwDict)
        maxPorosityDict = pad_dict_list(dict_list=maxPorosityDict)
        minPorosityDict = pad_dict_list(dict_list=minPorosityDict)
        meanPorosityDict = pad_dict_list(dict_list=meanPorosityDict)
        maxPorosityQDict = pad_dict_list(dict_list=maxPorosityQDict)
        minPorosityQDict = pad_dict_list(dict_list=minPorosityQDict)
        meanPorosityQDict = pad_dict_list(dict_list=meanPorosityQDict)
        ewDict = pad_dict_list(dict_list=ewDict)
        lwDict = pad_dict_list(dict_list=lwDict)
        ewprDict = pad_dict_list(dict_list=ewprDict)
        lwprDict = pad_dict_list(dict_list=lwprDict)
        ewPorosityDict = pad_dict_list(dict_list=ewPorosityDict)
        lwPorosityDict = pad_dict_list(dict_list=lwPorosityDict)
        for sec in range(len(sectorsPorosityDict)):
            sectorsPorosityDict[sec] = pad_dict_list(dict_list=sectorsPorosityDict[sec])

        # Get dataframes for results
        rwDF = pd.DataFrame(data=rwDict)
        maxPorosityDF = pd.DataFrame(data=maxPorosityDict)
        minPorosityDF = pd.DataFrame(data=minPorosityDict)
        meanPorosityDF = pd.DataFrame(data=meanPorosityDict)
        maxPorosityQDF = pd.DataFrame(data=maxPorosityQDict)
        minPorosityQDF = pd.DataFrame(data=minPorosityQDict)
        meanPorosityQDF = pd.DataFrame(data=meanPorosityQDict)
        ewDF = pd.DataFrame(data=ewDict)
        lwDF = pd.DataFrame(data=lwDict)
        ewprDF = pd.DataFrame(data=ewprDict)
        lwprDF = pd.DataFrame(data=lwprDict)
        ewPorosityDF = pd.DataFrame(data=ewPorosityDict)
        lwPorosityDF = pd.DataFrame(data=lwPorosityDict)

        # Save results in rwl-extension
        rw2rwl(data=rwDF, savePath=rw_rwl_Path, end_year=self.yearStart, coef=1)
        rw2rwl(data=maxPorosityDF, savePath=max_rwl_Path, end_year=self.yearStart, coef=1000)
        rw2rwl(data=minPorosityDF, savePath=min_rwl_Path, end_year=self.yearStart, coef=1000)
        rw2rwl(data=meanPorosityDF, savePath=mean_rwl_Path, end_year=self.yearStart, coef=1000)
        rw2rwl(data=maxPorosityQDF, savePath=maxq_rwl_Path, end_year=self.yearStart, coef=1000)
        rw2rwl(data=minPorosityQDF, savePath=minq_rwl_Path, end_year=self.yearStart, coef=1000)
        rw2rwl(data=meanPorosityQDF, savePath=meanq_rwl_Path, end_year=self.yearStart, coef=1000)
        rw2rwl(data=ewDF, savePath=ew_rwl_Path, end_year=self.yearStart, coef=1000)
        rw2rwl(data=lwDF, savePath=lw_rwl_Path, end_year=self.yearStart, coef=1000)
        rw2rwl(data=ewprDF, savePath=ewpr_rwl_Path, end_year=self.yearStart, coef=1000)
        rw2rwl(data=lwprDF, savePath=lwpr_rwl_Path, end_year=self.yearStart, coef=1000)
        rw2rwl(data=ewPorosityDF, savePath=ewp_rwl_Path, end_year=self.yearStart, coef=1000)
        rw2rwl(data=lwPorosityDF, savePath=lwp_rwl_Path, end_year=self.yearStart, coef=1000)



        # Save results
        saveDFasTXT(data=rwDF, filePath=rwPath, sep='\t')
        saveDFasTXT(data=maxPorosityDF, filePath=maxPPath, sep='\t')
        saveDFasTXT(data=minPorosityDF, filePath=minPPath, sep='\t')
        saveDFasTXT(data=meanPorosityDF, filePath=meanPPath, sep='\t')
        saveDFasTXT(data=maxPorosityQDF, filePath=maxPQPath, sep='\t')
        saveDFasTXT(data=minPorosityQDF, filePath=minPQPath, sep='\t')
        saveDFasTXT(data=meanPorosityQDF, filePath=meanPQPath, sep='\t')
        saveDFasTXT(data=ewDF, filePath=ewPath, sep='\t')
        saveDFasTXT(data=lwDF, filePath=lwPath, sep='\t')
        saveDFasTXT(data=ewprDF, filePath=ewprPath, sep='\t')
        saveDFasTXT(data=lwPorosityDF, filePath=lwprPath, sep='\t')
        saveDFasTXT(data=longPorosityProfilesDF, filePath=longPath, sep='\t')
        saveDFasTXT(data=AVG, filePath=avgPath, sep='\t')

        for sec in range(len(sectorsPorosityDict)):
            secDF = pd.DataFrame(data=sectorsPorosityDict[sec])
            saveDFasTXT(data=secDF, filePath=secPaths[sec], sep='\t')
            

    def scanSubDir(self, subDir: str, imgsNames: list) -> pd.DataFrame:
        """Scans images from a directory
        
        Parameters
        ----------
        subDir : str
            subdirectory
        imgsNames : list
            names of images

        Returns
        -------
        porosityByYearsDF : pd.DataFrame
            dataframe with yearly porosity profiles
        """

        porosityByYearsDict = {}
        porosity_SD_Dict = {}
        porosity_SMA_SD_Dict = {}
        
        treeN = subDir.split('/')[-1]
        sdPorosityPath = os.path.join(self.savePath, SAVE_PATHS["sd_porosity_path"], treeN)
        SD_path = os.path.join(sdPorosityPath, f'{subDir.split('/')[-1]}_SD.txt')
        SMA_SD_path = os.path.join(sdPorosityPath, f'{subDir.split('/')[-1]}_SMA_SD.txt')
        rawPorosityDir = os.path.join(self.savePath, SAVE_PATHS["raw_path"], treeN)

        for imName in imgsNames:
            imN = int(imName.split('.')[0])
            imPath = os.path.join(subDir, imName)
            porosityDF = self.scanImg(imPath)
            porosityDF['finalPorosityProfile'] = porosityDF.mean(axis=1)
            rawProfile = pd.DataFrame(data={imN: porosityDF['finalPorosityProfile']})

            rawPorosityPath = os.path.join(rawPorosityDir, f'{imN}.txt')

            porosity_SD_List, detrend_SMA_SD_List, porosityDF = self.getSD(
                porosityDF=porosityDF, 
                nTree=imN, 
                savePath=sdPorosityPath,
                r=300,
                step=5,
                leftBorderPerc=0.1,
                rightBorderPerc=0.3)
            
            porosity_SD_Dict.update({imN : porosity_SD_List})
            porosity_SMA_SD_Dict.update({imN : detrend_SMA_SD_List})
            porosityByYearsDict.update({self.yearStart-imN+1: porosityDF['finalPorosityProfile'].tolist()})

            saveDFasTXT(data=rawProfile, filePath=rawPorosityPath, sep='\t')

        porosity_SD_Dict = pad_dict_list(porosity_SD_Dict)
        porosity_SMA_SD_Dict = pad_dict_list(porosity_SMA_SD_Dict)
        porosityByYearsDict = pad_dict_list(porosityByYearsDict)

        porosity_SD = pd.DataFrame(data=porosity_SD_Dict)
        porosity_SMA_SD = pd.DataFrame(data=porosity_SMA_SD_Dict)
        porosityByYearsDF = pd.DataFrame(data=porosityByYearsDict)

        saveDFasTXT(data=porosity_SD, filePath=SD_path, sep='\t')
        saveDFasTXT(data=porosity_SMA_SD, filePath=SMA_SD_path, sep='\t')

        return porosityByYearsDF


    def getSD(
        self, 
        porosityDF: pd.DataFrame, 
        nTree: int, 
        savePath: str, 
        r: int=300, 
        step: int=5, 
        leftBorderPerc:float|int=0.1, 
        rightBorderPerc:float|int=0.3):
        """returns standard deviations for chronologies and standard deviations for detrended chronologies
        
        Parameters
        ----------
        porosityDF : pd.DataFrame
            dataframe with porosity profiles
        nTree : int
            number of tree
        savePath : str
            save path
        r : int
            maximum smoothing interval (default=300)
        step : step
            step that is added to the smoothing interval at each iteration (default=5)
        leftBorderPerc : float | int
            the fraction of the ring porosity profile from which the analysis will begin (default=0.1)
        rightBorderPerc : float | int
            the fraction of the ring porosity profile at which the analysis will end (default=0.3)

        Returns
        -------
        porosity_SD_List : List
            standard deviations of porosity profiles
        detrend_SMA_SD_List : List
            standard deviations of detrended porosity profiles
        porosityDF : pd.DataFrame
            dataframe with porosity profiles, supplemented with smooth curves
        """

        porosity_SD_List = []
        detrend_SMA_SD_List = []
        detrendDf = pd.DataFrame({0: []})
        detrendSMA = pd.DataFrame({0: []})

        smoothPorositySavePath = os.path.join(savePath, f'sd_{nTree}.txt')
        detrendPorositySavePath = os.path.join(savePath, f'detrend_{nTree}.txt')

        for i in range(5, r+1, step):
            if len(porosityDF['finalPorosityProfile'].dropna()) < i:
                break

            porosityDF[f'smooth_{i}'] = porosityDF['finalPorosityProfile'].rolling(i).mean()
            porosityDF[f'smooth_{i}'] = porosityDF[f'smooth_{i}'].shift(periods=-i//2)

            leftBorder = int(len(porosityDF[f'smooth_{i}'].dropna())*leftBorderPerc)
            rightBorder = int(len(porosityDF[f'smooth_{i}'].dropna())*rightBorderPerc)

            porosity_SD_List.append(porosityDF[f'smooth_{i}'][leftBorder: rightBorder].std())
            detrendDf[f'detrend_{i}'] = porosityDF['finalPorosityProfile'] - porosityDF[f'smooth_{i}']

        if 'detrend_300' in detrendDf.columns:
            detrend_SMA_SD_List.append(detrendDf['detrend_150'].std())
            for i in range(0, 300, step):
                detrendSMA[i] = detrendDf['detrend_150'].rolling(i).mean()
                detrendSMA[i] = detrendSMA[i].shift(periods=-i//2)
                detrend_SMA_SD_List.append(detrendSMA[i].std())

        saveDFasTXT(data=porosityDF, filePath=smoothPorositySavePath, sep='\t')
        saveDFasTXT(data=detrendDf, filePath=detrendPorositySavePath, sep='\t')

        return porosity_SD_List, detrend_SMA_SD_List, porosityDF


    def getLongPorosityProfile(self, porosityDict: dict, normMethod: str) -> tuple:
        """create a normalized long-term porosity profile
        
        Parameters
        ----------
        porosityDict : dict
            dictionary containing dataframes with individual porosity profiles
        normMethod : Str
            normalization method (small_ring, median, mean)

        Returns
        -------
        longPorosityProfilesDF : pd.DataFrame
            long-term porosity profile
        AVG : pd.DataFrame
            yearly average porosity profiles
        """

        # determine the smallest year in the chronology
        yearEnd = self.yearStart
        
        for n in porosityDict:
            cols = porosityDict[n].columns
            if min(cols) < yearEnd:
                yearEnd = min(cols)
        longPorosityDict = {y: {} for y in range(yearEnd, self.yearStart+1)}

        for y in range(yearEnd, self.yearStart+1):
            
            for n in porosityDict:
                cols = porosityDict[n].columns
                
                if y in cols:
                    clearPorosity = list(filter(lambda x: str(x) != 'nan', porosityDict[n][y].tolist()))
                    longPorosityDict[y].update({n: clearPorosity})

        longPorosityProfilesList = []
        meansProfilesDict = {}
        
        for y in longPorosityDict:
            padPorosity = pad_dict_list(longPorosityDict[y])
            subDF = pd.DataFrame(data=padPorosity)
            if subDF.empty:
                continue
            normPorosityDF = self.getNormPorosityDF(porosityDF=subDF, normMethod=normMethod)
            normPorosityDF['MEAN'] = normPorosityDF.mean(axis=1)
            meansProfilesDict.update({y: normPorosityDF['MEAN'].tolist()})
            longPorosityProfilesList.append(normPorosityDF)
        meansProfilesDict = pad_dict_list(meansProfilesDict)
        AVG = pd.DataFrame(data=meansProfilesDict)
        longPorosityProfilesDF = pd.concat(longPorosityProfilesList)
        return longPorosityProfilesDF, AVG

    def getNormPorosityDF(self, porosityDF: pd.DataFrame, normMethod: str="median") -> pd.DataFrame:
        """normalizes all density profiles (by the length of the smallest ring)
        
        Parameters
        ----------
        porosityDF : pd.DataFrame
            non-normalized averaged annualized porosity profiles of one tree
        normMethod : str
            Choices: small_ring, median_ring, mean_ring

        Returns
        -------
        normPorosityDF : pd.DataFrame
            normalized averaged annualized porosity profiles of one tree

        """

        clearPorosityProfilesDict = {}
        lens = []

        for col in porosityDF.columns:
            clearPorosity = list(filter(lambda x: str(x)!='nan', porosityDF[col].tolist()))
            lenPorosity = len(clearPorosity)
            if lenPorosity == 0:
                continue

            clearPorosityProfilesDict.update({col: clearPorosity})
            lens.append(lenPorosity)

        if not lens:
            return porosityDF
        if normMethod == "small_ring":
            reqRW = min(lens)
        elif normMethod == "median":
            reqRW = median(lens)
        elif normMethod == "mean":
            reqRW = mean(lens)
        else:
            reqRW = lens[0]

        reqRW = mathRound(reqRW)

        normPorosityDict = {}
        for cp in clearPorosityProfilesDict:
            if len(clearPorosityProfilesDict[cp]) <= 5:
                normPorosity = [0 for i in range(reqRW)]
            else:
                normPorosity = getNormalisationPorosityProfile(clearPorosityProfilesDict[cp], reqRW)
            normPorosityDict.update({cp: normPorosity})

        normPorosityDF = pd.DataFrame(data=normPorosityDict)
        return normPorosityDF
    
    def getPorProfilesNaturalValues(self, porosityProfiles: pd.DataFrame) -> pd.DataFrame:
        """
        Normalizes “raw” porosity profiles using a conversion factor. 
        The resulting porosity profiles have lengths in micrometers

        Parameters
        ----------
        porosityProfiles : pd.DataFrame
            averaged annualized porosity profiles of one tree

        Returns
        -------
        porosityProfilesNaturalValues : pd.DataFrame
            normalized averaged annualized porosity profiles of one tree
        """
        PorProfilesNaturalValuesDict = {}

        for col in porosityProfiles.columns:
            clearPorosity = list(filter(lambda x: str(x) != 'nan', porosityProfiles[col].tolist()))
            if len(clearPorosity) < 5:
                normPorosity = [0 for i in range(self.normNumber)]
            else:
                reqRW = mathRound(len(clearPorosity) * self.pixToMcmCoef)
                normPorosity = getNormalisationPorosityProfile(clearPorosity, reqRW)
            PorProfilesNaturalValuesDict.update({col: normPorosity})

        PorProfilesNaturalValuesDict = pad_dict_list(PorProfilesNaturalValuesDict)
        porosityProfilesNaturalValues = pd.DataFrame(data=PorProfilesNaturalValuesDict)
        return porosityProfilesNaturalValues

    def getNormPorosityProfiles(self, porosityProfiles: pd.DataFrame) -> pd.DataFrame:
        """
        normalizes “raw” porosity profiles to a single dimension defined by the self.normNumber value

        Parameters
        ----------
        porosityProfiles : pd.DataFrame
            averaged annualized porosity profiles of one tree

        Returns
        -------
        normPorosityProfilesDF : pd.DataFrame
            normalized averaged annualized porosity profiles of one tree
        """
        normPorosityProfilesDict = {}

        for col in porosityProfiles.columns:
            clearPorosity = list(filter(lambda x: str(x) != 'nan', porosityProfiles[col].tolist()))
            if len(clearPorosity) < 5:
                normPorosity = [0 for i in range(self.normNumber)]
            else:
                normPorosity = getNormalisationPorosityProfile(clearPorosity, self.normNumber)
            normPorosityProfilesDict.update({col: normPorosity})

        normPorosityProfilesDF = pd.DataFrame(data=normPorosityProfilesDict)
        return normPorosityProfilesDF

    def getRW(self, porosityProfiles: pd.DataFrame) -> list:
        """return tree ring width chronology
        
        Parameters
        ----------
        porosityProfiles : pd.DataFrame
            averaged annualized porosity profiles of one tree

        Returns
        -------
        trw: List
            annual ring width
        """
        trw = []
        columns = porosityProfiles.columns
        years = sorted(list(map(lambda c: int(c), columns)))
        maxYear = max(years)
        minYear = min(years)
        
        for y in range(minYear, self.yearStart+1):
            if y not in columns:
                trw.append(-1)
            else:
                trw.append(int(porosityProfiles[y].count() * self.pixToMcmCoef)) 

        trw = list(map(lambda i: int(i), trw))[::-1]
        return trw

    def getPorosityCharacteristics(self, porosityProfiles: pd.DataFrame) -> tuple:
        """return max, min and mean porosity profile chronology
        
        Parameters
        ----------
        porosityProfiles : pd.DataFrame
            averaged annualized porosity profiles of one tree

        Returns
        -------
        maxPorosity : List
            yearly maximum porosity
        minPorosity : List
            yearly minimum porosity
        meanPorosity : List
            yearly mean porosity
        """

        columns = porosityProfiles.columns
        years = sorted(list(map(lambda c: int(c), columns)))
        maxYear = max(years)
        minYear = min(years)
        
        maxPorosity = []
        minPorosity = []
        meanPorosity = []

        for y in range(minYear, self.yearStart+1):
            if y not in columns:
                maxPorosity.append(-1)
                minPorosity.append(-1)
                meanPorosity.append(-1)
            else:
                if len(porosityProfiles[y].dropna()) == 0:
                    maxPorosity.append(-1)
                    minPorosity.append(-1)
                    meanPorosity.append(-1)
                    continue

                maxPorosity.append(porosityProfiles[y].max())
                minPorosity.append(porosityProfiles[y].min())
                meanPorosity.append(porosityProfiles[y].mean())

        maxPorosity = maxPorosity[::-1]
        minPorosity = minPorosity[::-1]
        meanPorosity = meanPorosity[::-1]

        return maxPorosity, minPorosity, meanPorosity


    def getPorosityCharacteristicsProcentile(self, porosityProfiles: pd.DataFrame) -> tuple:
        """return max, min and mean porosity profile chronology
        
        Parameters
        ----------
        porosityProfiles : pd.DataFrame
            averaged annualized porosity profiles of one tree

        Returns
        -------
        maxPorosity : List
            yearly maximum porosity
        minPorosity : List
            yearly minimum porosity
        meanPorosity : List
            yearly mean porosity
        """

        columns = porosityProfiles.columns
        years = sorted(list(map(lambda c: int(c), columns)))
        maxYear = max(years)
        minYear = min(years)
        
        maxPorosityQ = []
        minPorosityQ = []
        meanPorosityQ = []

        for y in range(minYear, self.yearStart+1):
            if y not in columns:
                maxPorosityQ.append(-1)
                minPorosityQ.append(-1)
                meanPorosityQ.append(-1)
            else:
                if len(porosityProfiles[y].dropna()) == 0:
                    maxPorosityQ.append(-1)
                    minPorosityQ.append(-1)
                    meanPorosityQ.append(-1)
                    continue

                maxPorosityQ.append(porosityProfiles[y].quantile(0.95))
                minPorosityQ.append(porosityProfiles[y].quantile(0.05))
                meanPorosityQ.append(porosityProfiles[y].quantile(0.5))
        maxPorosityQ = maxPorosityQ[::-1]
        minPorosityQ = minPorosityQ[::-1]
        meanPorosityQ = meanPorosityQ[::-1]

        return maxPorosityQ, minPorosityQ, meanPorosityQ

    def getEarlyLateWidth(self, porosityProfiles: pd.DataFrame) -> tuple:
        """return early width, late width, early fraction, late fraction, 
        late wood porosity and early wood porosity chronology
        
        Parameters
        ----------
        porosityProfiles : pd.DataFrame
            averaged annualized porosity profiles of one tree

        Returns
        -------
        earlyWidthList : List
            yearly early width
        lateWidthList : List
            yearly late width
        earlyPercList : List
            yearly earlywood fraction
        latePercList : List
            yearly latewood fraction
        meanPorEarlyWoodList : List
            yearly earlywood porosity
        meanPorLateWoodList : List
            yearly latewood porosity
        """
        columns = porosityProfiles.columns
        years = sorted(list(map(lambda c: int(c), columns)))
        maxYear = max(years)
        minYear = min(years)

        earlyWidthList = []
        lateWidthList = []
        earlyPercList = []
        latePercList = []

        meanPorEarlyWoodList = []
        meanPorLateWoodList = []
        morcList = []

        earlyWidthDict = {}
        lateWidthDict = {}
        earlyPercDict = {}
        latePercDict = {}
        meanPorEarlyWoodDict = {}
        meanPorLateWoodDict = {}

        for y in range(minYear, self.yearStart+1):
            if y not in columns:
                earlyWidthList.append(-1)
                lateWidthList.append(-1)
                earlyPercList.append(-1)
                latePercList.append(-1)
                meanPorEarlyWoodList.append(-1)
                meanPorLateWoodList.append(-1)
                morcList.append(-1)
            else:
                if len(porosityProfiles[y].dropna()) == 0:
                    earlyWidthList.append(-1)
                    lateWidthList.append(-1)
                    earlyPercList.append(-1)
                    latePercList.append(-1)
                    meanPorEarlyWoodList.append(-1)
                    meanPorLateWoodList.append(-1)
                    morcList.append(-1)
                    continue

                cleanPor = porosityProfiles[y].dropna()

                meanValue = cleanPor.min()+(cleanPor.max()-cleanPor.min())/2
                porTh = np.percentile(cleanPor, 75)
                th = 0
                bestSTD = 100
                # rightBorder = int(len(cleanPor)*0.9)
                for i in range(len(cleanPor)-1, 0, -1):
                    if cleanPor.iloc[i] >= porTh:
                        th = i
                        break

                earlyWidth = th
                lateWidth = len(cleanPor) - earlyWidth
                earlyPerc = earlyWidth / len(cleanPor)
                latePerc = 1 - earlyPerc

                meanPorEarlyWood = cleanPor.iloc[:th].mean()
                meanPorLateWood = cleanPor.iloc[th:].mean()


                earlyWidthList.append(earlyWidth)
                lateWidthList.append(lateWidth)
                earlyPercList.append(earlyPerc)
                latePercList.append(latePerc)
                meanPorEarlyWoodList.append(meanPorEarlyWood)
                meanPorLateWoodList.append(meanPorLateWood)
        
        earlyWidthList = earlyWidthList[::-1]
        lateWidthList = lateWidthList[::-1]
        earlyPercList = earlyPercList[::-1]
        latePercList = latePercList[::-1]
        meanPorEarlyWoodList = meanPorEarlyWoodList[::-1]
        meanPorLateWoodList = meanPorLateWoodList[::-1]

        return earlyWidthList, lateWidthList, earlyPercList, latePercList, meanPorEarlyWoodList, meanPorLateWoodList

    def getSectorPorosity(self, porosityProfiles: pd.DataFrame, sectorsNumber: int=10) -> tuple:
        """
        returns the average porosity for each of the sectorsNumber of sectors
        
        Parameters
        ----------
        porosityProfiles : pd.DataFrame
            averaged annualized porosity profiles of one tree
        sectorsNumber : int
            number of secctors (default=10)

        Returns
        -------
        sectorsList : List
            list of average porosity by sector
        """
        columns = porosityProfiles.columns
        years = sorted(list(map(lambda c: int(c), columns)))
        maxYear = max(years)
        minYear = min(years)

        sectorsList = [[] for i in range(sectorsNumber)]
        meanPorLateWoodDict = {}

        for y in range(minYear, self.yearStart+1):
            if y not in columns:
                for s in range(sectorsNumber):
                    sectorsList[s].append(-1)
            else:
                if len(porosityProfiles[y].dropna()) == 0:
                    sectorsList[s].append(-1)
                    continue
                    
                cleanPor = porosityProfiles[y].dropna()
                step = len(cleanPor) // sectorsNumber
                for s in range(sectorsNumber):
                    sectorPorosity = cleanPor.iloc[step*s : len(cleanPor) - step * (sectorsNumber-1-s)].mean()
                    sectorsList[s].append(sectorPorosity)
        for s in range(sectorsNumber):
            sectorsList[s] = sectorsList[s][::-1]

        return sectorsList

    def scanImg(self, imgPath: str, windowSize: int = 1000, step: int= 200, windowsNumbers:int = 5) -> pd.DataFrame:
        """
        scans the image with a movable window that moves in “step” increments

        Parameters
        ----------
        imgPath : str
            path to image
        windowSize : int
            sliding window size
        step : int
            window moving step
        
        Returns
        -------
        porosityProfilesDF : pd.DataFrame
            porosity profiles obtained by scanning several windows      
        """

        print(f'[INFO] Scan image {imgPath}')
        if self.usePredBIImgs:
            binaryImage = cv2.imread(imgPath)
            binaryImage = cv2.cvtColor(binaryImage, cv2.COLOR_RGB2GRAY)
            th, binaryImage = cv2.threshold(binaryImage, 10, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        else:
            bi = BI(        
                blurType = self.blurType,  
                gammaEqualisation = self.gammaEqualisation, 
                ksize = self.ksize,
                constantTh = self.constantTh,
                modelPath = self.modelPath,
                biMethod = self.biMethod
                )
            binaryImage = bi.getBinaryImg(imgPath)
        porosityProfilesDict = {}

        if windowSize == -1:
            porosityProfileToPix = self.getPorosityProfileToPix(binaryImage)
            porosityProfilesDict.update({0: porosityProfileToPix})
        else:
            for i in range(windowsNumbers):
                biImgFragment = binaryImage[i*step:i*step+windowSize]
                porosityProfilePix = self.getPorosityProfileToPix(biImgFragment)
                porosityProfilesDict.update({i:porosityProfilePix})

        porosityProfilesDF = pd.DataFrame(data=porosityProfilesDict)
        porosityProfilesDF['finalPorosityProfile'] = porosityProfilesDF.mean(axis=1)

        return porosityProfilesDF

    def getPorosityProfileToPix(self, biImg: np.ndarray) -> list:
        """
        this method obtains porosity profiles from a given image

        Parameters
        ----------
        biImg : np.ndarray
            binary image

        Returns
        -------
        pixPorosityProfile : List
            “Raw” porosity profile. The length of the profile is equal to the length of the photo in pixels        
        """
        lumWallProfile = []
        pixPorosityProfile = []
        biImg = biImg.transpose()
        xImgSize, yImgSize = biImg.shape
        for x in range(xImgSize):
            if self.gapValue:
                xFiltered = self.gapFilter(biImg[x])
                whitePixN = np.sum(xFiltered == 255)
                pixPorosityProfile.append(whitePixN / len(xFiltered))
            else:
                whitePixN = np.sum(biImg[x] == 255)
                pixPorosityProfile.append(whitePixN / yImgSize)

        return pixPorosityProfile

    def gapFilter(self, x: np.array) -> list:
        """
        this method filters out “gaps” in the scan lines

        Parameters
        ----------
        x : np.ndarray
            scan line

        Returns
        -------
        filteredScanLine : List
            gapless scanning line
            
        """
        xgroup = [list(g) for k, g in groupby(x)]
        onDel = []
        for idx in range(len(xgroup)):
            if xgroup[idx][0] == 255 and len(xgroup[idx]) > self.gapValue:
                onDel.append(xgroup[idx])
        for od in onDel:
            xgroup.remove(od)
        filteredScanLine = []
        for idx in range(len(xgroup)):
            filteredScanLine += xgroup[idx]
        filteredScanLine = np.array(filteredScanLine)
        return filteredScanLine

    def saveConfigFile(self) -> None:
        """Performs an analysis of the root directory containing subdirectories with images"""

        config = pd.Series(data={
            'savePath': self.savePath,
            'root': self.root,
            'speciesName': self.speciesName,
            'start year': self.yearStart,
            'value of gap': self.gapValue,
            'SMA window': self.smaInterval
        }, index = ['savePath', 'root', 'speciesName', 'start year', 'value of gap', 'SMA window'])
        
        saveDFasTXT(data=config, filePath=os.path.join(self.savePath, 'config.txt'), sep='\t')

