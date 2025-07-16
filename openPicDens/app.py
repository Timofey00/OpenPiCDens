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
from unet.utills import predictImgMask
from itertools import groupby


class BI:
    """
    Parameters
    ----------
    imgPath : str
        path to image
    blurType: str
        Choises: Gaussian (default), Median, Normalized Box Filter (NBF), Bilateral
    biMethod: str
        Choises: Otsu, Mean, Gaussian, UNET
    gammaEqualisation: bool
        if True, the image is gamma-corrected
    ksize: int
        Kernel size

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
    biMethod : str
    options : list

    usePredBIImgs: bool
        set this variable to true if you have pre-binarized images (default is False)
    biImgsPath : str
        if the variable usePredBIImgs is True, the program uses this path as a source of binarized images. 
        In other case, this path is used as a directory for saving binarized images
        (default is False)
    modelPath : str | bool

    gapValue : int | None
        the size of the filter gap window. If set to None, the filter is not applied
        (default is None)

    Methods
    -------
    initPath(path: str) -> str
    getTreeDirs() -> dict
    initResDict() -> tuple
    saveDataToFile(savePath: str, data: pd.DataFrame, fileName: str, ext: str='txt') -> None
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
        options: list=['__all__'],
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

        self.savePath = savePath
        self.root = root
        self.speciesName = speciesName
        self.yearStart = yearStart
        self.normNumber = normNumber
        self.pixToMcmCoef = pixToMcmCoef
        self.normMethod = normMethod
        self.smaInterval = smaInterval
        self.options = options
        self.gapValue = gapValue

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

        # Init save-path
        self.areaPorSavePath = os.path.join(self.savePath, 'areaPorosity')
        self.initPath(self.areaPorSavePath)

        self.saveConfigFile()


    def saveConfigFile(self):
        config = pd.Series(data={
            'savePath': self.savePath,
            'root': self.root,
            'speciesName': self.speciesName,
            'start year': self.yearStart,
            'value of gap': self.gapValue,
            'SMA window': self.smaInterval
        }, index = ['savePath', 'root', 'speciesName', 'start year', 'value of gap', 'SMA window'])
        saveDFasTXT(data=config, filePath=os.path.join(self.savePath, 'config.txt'), sep='\t')

    def initPath(self, path: str) -> str:
        """
        this method creates the specified directory if it does not exist
        
        Parameters
        ----------
        path : str
            init path
        """
        path = path.replace('\\', '/')
        if not os.path.isdir(path):
            os.mkdir(path, 0o754)
        return path


    def getTreeDirs(self) -> dict:
        """
        this method builds a tree of paths to the images
        
        Returns
        ----------
        treeDirs : dict
             image path tree
        """
        sortedSubDirNames = sorted(os.listdir(self.root), key=lambda f: int(f))
        treeDirs = {}

        for d in sortedSubDirNames:
            subDir = os.path.join(self.root, d)
            sortedImgNames = sorted(os.listdir(subDir), key=lambda f: int(f.split('.')[0]))
            treeDirs.update({d : sortedImgNames})

        return treeDirs

    def initResDict(self) -> tuple:
        """
        
        
        Returns
        ----------
        dictsRes : tuple
        """
        rwDict = {}
        maxPorosityDict = {}
        minPorosityDict = {}
        meanPorosityDict = {}
        earlyWidthDict = {}
        lateWidthDict = {}
        earlyPercDict = {}
        latePercDict = {}
        maxPorosityQDict = {}
        minPorosityQDict = {}
        meanPorosityQDict = {}
        meanPorEarlyWoodDict = {}
        meanPorLateWoodDict = {}

        treesPorosityDict = {}
        normPorosityDict = {}
        sectorsDict = {s: {} for s in range(10)}
        longPorosityProfile = 0
        AVG = 0

        oneFileRes = {
            'rw' : [
                self.getRW, [rwDict], ['rw']
            ],
            'mainPorosityFeatures' : [
                self.getPorosityCharacteristics, [maxPorosityDict, minPorosityDict, meanPorosityDict],
                ['maxPorosity', 'minPorosity', 'meanPorosity']
            ],
            'earlyLateWoodFeatures' : [
                self.getEarlyLateWidth, [earlyWidthDict, lateWidthDict, earlyPercDict, latePercDict, meanPorEarlyWoodDict, meanPorLateWoodDict],
                ['earlyWidth', 'lateWidth', 'earlyPerc', 'latePerc', 'meanPorEarlyWood', 'meanPorLateWood']
            ],
            'mainPorosityFeaturesQuantile' : [
                self.getPorosityCharacteristicsProcentile, [maxPorosityQDict, minPorosityQDict, meanPorosityQDict],
                ['maxPorosityQ', 'minPorosityQ', 'meanPorosityQ']
            ],
        }

        severalFileRes = {
            'treesPorosity' : [
                self.getPorProfilesNaturalValues, [treesPorosityDict], ['naturalValuesPorosity']
            ],
            'normPorosity' : [
                self.getNormPorosityProfiles, [normPorosityDict], ['normValuesPorosity']
            ],
        }
        sectorsFileRes = {
            'sectorsPorosity' : [
                self.getSectorPorosity, [sectorsDict], ['sectorsPorosity']
            ]
        }

        longFileRes = {
            'annualPorosity' : [
                self.getLongPorosityProfile, [longPorosityProfile, AVG], ['longNormPorosity', 'AVG']
            ]
        }

        if self.options == ['__all__']:
            return oneFileRes, severalFileRes, sectorsFileRes, longFileRes

    def saveDataToFile(self, savePath: str, data: pd.DataFrame, fileName: str, ext: str='txt', reverse=True) -> None:
        """
        A method for automated scanning of a directory containing images of micrographs of woody plants.
        Parameters
        -------
        savePath : str
            saving path
        data: pd.DataFrame
            dataframe to save
        fileName : str
            name of file to save
        ext : str
            file extension

        """
        savePath = os.path.join(savePath, f'{fileName}.{ext}')
        savePath = savePath.replace('\\', '/')
        if reverse:
            data = data[::-1]
        
        saveDFasTXT(data=data, filePath=savePath, sep='\t')


    def startScan(self) -> None:
        """
        A method for automated scanning of a directory containing images of micrographs of woody plants

        """
        
        treeDirs = self.getTreeDirs()
        oneFileResDict, severalFileResDict, sectorsFileResDict, longRawResDict = self.initResDict()
        rawPorosityDict = {}

        for d in treeDirs:
            subDir = os.path.join(self.root, d)
            porosityByYearsDF = self.scanSubDir(subDir=subDir, imgsList=treeDirs[d])
            porosityByYearsSMADF = smaDF(porosityByYearsDF, smaInterval=self.smaInterval)
            rawPorosityDict.update({d: porosityByYearsDF})


            for opt in oneFileResDict:
                func = oneFileResDict[opt][0]
                results = func(porosityByYearsDF) if opt == 'rw' else func(porosityByYearsSMADF)

                for i in range(len(oneFileResDict[opt][1])):
                    if type(results) is tuple:
                        oneFileResDict[opt][1][i].update({d : results[i]})
                    else:
                        oneFileResDict[opt][1][i].update({d : results})

            for opt in severalFileResDict:
                func = severalFileResDict[opt][0]
                results = func(porosityByYearsSMADF)

                for i in range(len(severalFileResDict[opt][1])):
                    if type(results) is tuple:
                        severalFileResDict[opt][1][i].update({d : results[i]})
                    else:
                        severalFileResDict[opt][1][i].update({d : results})

            for opt in sectorsFileResDict:
                func = sectorsFileResDict[opt][0]
                results = func(porosityByYearsSMADF)

                for i in range(len(sectorsFileResDict[opt][1])):
                    for sector in sectorsFileResDict[opt][1][i]:
                        sectorData = results[sector]
                        sectorsFileResDict[opt][1][i][sector].update({d : sectorData})


        rwlSavePath = os.path.join(self.savePath, 'rwl')
        for opt in oneFileResDict:
            if not os.path.isdir(rwlSavePath):
                os.mkdir(rwlSavePath, 0o754)

            for i in range(len(oneFileResDict[opt][1])):
                oneFileResDict[opt][1][i] = pad_dict_list(oneFileResDict[opt][1][i])
                oneFileResDict[opt][1][i] = pd.DataFrame(data=oneFileResDict[opt][1][i])
                self.saveDataToFile(savePath=self.savePath, data=oneFileResDict[opt][1][i], fileName=oneFileResDict[opt][2][i])

                if oneFileResDict[opt][2][i] in ('rw', 'earlyWidth', 'lateWidth'):
                    coef = 1
                else:
                    coef = 1000
                rw2rwl(data=oneFileResDict[opt][1][i], fileName=oneFileResDict[opt][2][i], savePath=rwlSavePath, end_year=self.yearStart, coef=coef)
        
        for opt in severalFileResDict:
            for i in range(len(severalFileResDict[opt][1])):
                for treeN in severalFileResDict[opt][1][i]:
                    data = severalFileResDict[opt][1][i][treeN]
                    subDir = severalFileResDict[opt][2][i]
                    sp = os.path.join(self.savePath, subDir)
                    self.initPath(sp)
                    self.saveDataToFile(savePath=sp, data=data, fileName=f'{treeN}')

        for opt in sectorsFileResDict:
            for i in range(len(sectorsFileResDict[opt][1])):
                subDir = sectorsFileResDict[opt][2][i]
                for sector in sectorsFileResDict[opt][1][i]:
                    data = pad_dict_list(sectorsFileResDict[opt][1][i][sector])
                    data = pd.DataFrame(data)
                    sp = os.path.join(self.savePath, subDir)
                    coef = 1000
                    self.initPath(sp)
                    self.saveDataToFile(savePath=sp, data=data, fileName=f'sector_{sector}')
                    rw2rwl(data=data, fileName=f'sector_{sector}', savePath=rwlSavePath, end_year=self.yearStart, coef=coef)

        for rp in rawPorosityDict:
            sp = os.path.join(self.savePath, 'rawPorosity')
            self.initPath(sp)
            self.saveDataToFile(savePath=sp, data=rawPorosityDict[rp], fileName=rp)

        for opt in longRawResDict:
            func = longRawResDict[opt][0]
            results = func(severalFileResDict['normPorosity'][1][0], normMethod='median')

            for r in range(len(results)):
                fileName = longRawResDict[opt][2][r]
                self.saveDataToFile(savePath=self.savePath, data=results[r], fileName=fileName)


    def scanSubDir(self, subDir: str, imgsList: list) -> pd.DataFrame:
        """
        this method scans a directory of wood microstructure images
        
        Parameters
        ----------
        subDir : str
            image subdirectory
        imgsList : list
            list of images in the directory

        Returns
        -------
        porosityByYearsDF : pd.DataFrame
            averaged annualized porosity profiles of one tree 
        """
        print(f'[INFO] Start scan direction {subDir}')
        
        imgsPathList = [os.path.join(subDir, i) for i in imgsList]
        porosityByYearsDict = {}
        areaSDPath = os.path.join(self.areaPorSavePath, subDir.split('/')[-1])
        self.initPath(areaSDPath)
        porositySDDict = {}
        porositySMASDDict = {}

        for ip, im in zip(imgsPathList, imgsList):
            porositySDList = []
            imageNumber = int(im.split('.')[0])
            porosityDF = self.scanImg(ip)
            porosityDF['finalPorosityProfile'] = porosityDF.mean(axis=1)

            porositySDList, detrendSMASDList, porosityDF = self.sdSmooth(porosityDF, im.split('.')[0], areaSDPath)
            porositySDDict.update({im : porositySDList})
            porositySMASDDict.update({im : detrendSMASDList})
            porosityByYearsDict.update({self.yearStart-imageNumber+1: porosityDF['finalPorosityProfile'].tolist()})


        porositySDDict = pad_dict_list(porositySDDict)
        porositySMASDDict = pad_dict_list(porositySMASDDict)
        porositySD = pd.DataFrame(data=porositySDDict)
        porositySMASD = pd.DataFrame(data=porositySMASDDict)
        self.saveDataToFile(data=porositySD, fileName=f'{subDir.split('/')[-1]}_SD', savePath=areaSDPath)
        self.saveDataToFile(data=porositySMASD, fileName=f'{subDir.split('/')[-1]}_SMASD', savePath=areaSDPath)

        porosityByYearsDict = pad_dict_list(porosityByYearsDict)
        porosityByYearsDF = pd.DataFrame(data=porosityByYearsDict)
        return porosityByYearsDF

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
            convertCoef = reqRW/len(clearPorosityProfilesDict[cp])
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
            # Если windowSize определен как -1, то сканируем изображение полностью, не разбивая его на окна
            porosityProfileToPix = self.getPorosityProfileToPix(binaryImage)
            porosityProfilesDict.update({0: porosityProfileToPix})
        else:
            for i in range(windowsNumbers):
                biImgFragment = binaryImage[i*step:i*step+windowSize]
                porosityProfilePix = self.getPorosityProfileToPix(biImgFragment)
                porosityProfilesDict.update({i:porosityProfilePix})

        porosityProfilesDF = pd.DataFrame(data=porosityProfilesDict)

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

    def sdSmooth(self, porosityDF, nTree, savePath, r=300, step=5, leftBorderPerc=0.1, rightBorderPerc=0.3):
        """
        
        Parameters
        ----------


        Returns
        -------

        """
        porositySDList = []
        detrendSMASDList = []
        detrendDf = pd.DataFrame({0: []})
        detrendSMA = pd.DataFrame({0: []})


        for i in range(5, r+1, step):
            if len(porosityDF['finalPorosityProfile'].dropna()) < i:
                break

            porosityDF[f'smooth_{i}'] = porosityDF['finalPorosityProfile'].rolling(i).mean()
            porosityDF[f'smooth_{i}'] = porosityDF[f'smooth_{i}'].shift(periods=-i//2)

            leftBorder = int(len(porosityDF[f'smooth_{i}'].dropna())*leftBorderPerc)
            rightBorder = int(len(porosityDF[f'smooth_{i}'].dropna())*rightBorderPerc)

            porositySDList.append(porosityDF[f'smooth_{i}'][leftBorder: rightBorder].std())

            detrendDf[f'detrend_{i}'] = porosityDF['finalPorosityProfile'] - porosityDF[f'smooth_{i}']

        if 'detrend_300' in detrendDf.columns:
            detrendSMASDList.append(detrendDf['detrend_150'].std())
            for i in range(0, 300, step):
                detrendSMA[i] = detrendDf['detrend_150'].rolling(i).mean()
                detrendSMA[i] = detrendSMA[i].shift(periods=-i//2)
                detrendSMASDList.append(detrendSMA[i].std())

        self.saveDataToFile(data=porosityDF, fileName=f'sd_{nTree}', savePath=savePath)
        self.saveDataToFile(data=detrendDf, fileName=f'detrend_{nTree}', savePath=savePath)
        return porositySDList, detrendSMASDList, porosityDF