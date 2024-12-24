"""
Main App-module
"""

import os
from statistics import mean, median
import pathlib

import numpy as np
import pandas as pd
import cv2

from .utils import *

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
        coefficient for converting porosity length to micrometers
    speciesName : str
        species name
    yearStart : int
        start year
    normNumber : int
        length of normalized porosity profile
    normMethod : str
        normalization method

    Methods
    -------
    startScan
    getLongPorosityProfile(porosityDict: dict, normMethod: str)
    getNormPorosityDF(porosityDF: pd.DataFrame, normMethod: str="median")
    scanRoot
    getPorProfilesNaturalValues(porosityProfiles: pd.DataFrame)
    getNormPorosityProfiles(porosityProfiles: pd.DataFrame)
    getRW(self, porosityProfiles: pd.DataFrame)
    getPorosityCharacteristics(porosityProfiles: pd.DataFrame)
    getNormalisationPorosityProfile(porosityProfile: list, convertCoef: float | int)
    scanSubDir(path : str)
    scanImg(imgPath: str, windowSize: int = 1000, step: int= 200)
    getPorosityProfileToPix(biImg: np.ndarray)
    getBinaryImg(imgPath: str, blurType: str='Median', biMethod: str='Otsu', gammaEqualisation: bool=False, ksize: int=7)
    

    """
    def __init__(
        self, 
        savePath,
        root,
        speciesName,
        yearStart,
        normNumber,
        pixToMcmCoef = 1,
        normMethod="median",
        smaInterval=20,
        biMethod='Otsu',
        predBI=None
        ):

        self.savePath = savePath
        self.root = root
        self.pixToMcmCoef = pixToMcmCoef
        self.speciesName = speciesName
        self.yearStart = yearStart
        self.normNumber = normNumber
        self.normMethod="median"
        self.smaInterval = smaInterval
        self.biMethod = biMethod


        # Init save-path
        if not os.path.isdir(self.savePath):
            os.mkdir(self.savePath, 0o754)


    def startScan(self) -> tuple:
        """
        A method for automated scanning of a directory containing images of micrographs of woody plants.

        Returns
        -------
        longPorosityProfileInNaturalValuesDF : pd.DataFrame
            long-term porosity profile in micrometers
        longPorosityProfileNorm : pd.DataFrame
            long-term normalized porosity profile

        """
        
        treesPorosityDict, normPorosityDict, rwDF, maxPorosityDF, minPorosityDF, meanPorosityDF, rawPorosityDict, earlyWidthDF, lateWidthDF, earlyPercDF, latePercDF, meanPorEarlyWoodDF, meanPorLateWoodDF = self.scanRoot()
        longPorosityProfileInNaturalValuesDF, _ = self.getLongPorosityProfile(treesPorosityDict, normMethod=self.normMethod)
        longPorosityProfileNorm, AVG = self.getLongPorosityProfile(normPorosityDict, normMethod=None)   
        
        saveAVGPorosity = os.path.join(self.savePath, 'AVG.txt')
        savePathLongPorosity = os.path.join(self.savePath, 'longPorosity.txt')
        savePathLongNormPorosity = os.path.join(self.savePath, 'longNormPorosity.txt')
        savePathRW = os.path.join(self.savePath, 'rw.txt')
        savePathMaxPorosity = os.path.join(self.savePath, 'maxPorosity.txt')
        savePathMinPorosity = os.path.join(self.savePath, 'minPorosity.txt')
        savePathMeanPorosity = os.path.join(self.savePath, 'meanPorosity.txt')
        savePathPorosity = os.path.join(self.savePath, 'porosity')
        savePathNormPorosity = os.path.join(self.savePath, 'normPorosity')
        savePathRawPorosity = os.path.join(self.savePath, 'rawPorosityData')

        savePathEarlyWidth = os.path.join(self.savePath, 'earlyWidth.txt')
        savePathLateWidth = os.path.join(self.savePath, 'lateWidth.txt')
        savePathEarlyPerc = os.path.join(self.savePath, 'earlyPerc.txt')
        savePathLatePerc = os.path.join(self.savePath, 'latePerc.txt')
        savePathMeanPorEarlyWood = os.path.join(self.savePath, 'meanPorEarlyWood.txt')
        savePathMeanPorLateWood = os.path.join(self.savePath, 'meanPorLateWood.txt')


        AVG.to_csv(saveAVGPorosity, sep='\t')
        longPorosityProfileInNaturalValuesDF.to_csv(savePathLongPorosity, sep='\t')
        longPorosityProfileNorm.to_csv(savePathLongNormPorosity, sep='\t')
        rwDF.to_csv(savePathRW, sep='\t')
        maxPorosityDF.to_csv(savePathMaxPorosity, sep='\t')
        minPorosityDF.to_csv(savePathMinPorosity, sep='\t')
        meanPorosityDF.to_csv(savePathMeanPorosity, sep='\t')

        earlyWidthDF.to_csv(savePathEarlyWidth, sep='\t')
        lateWidthDF.to_csv(savePathLateWidth, sep='\t')
        earlyPercDF.to_csv(savePathEarlyPerc, sep='\t')
        latePercDF.to_csv(savePathLatePerc, sep='\t')
        meanPorEarlyWoodDF.to_csv(savePathMeanPorEarlyWood, sep='\t')
        meanPorLateWoodDF.to_csv(savePathMeanPorLateWood, sep='\t')

        for path, por in zip((savePathPorosity, savePathNormPorosity, savePathRawPorosity), (treesPorosityDict, normPorosityDict, rawPorosityDict)):
            txtPath = os.path.join(f'{path}/', 'txt')
            csvPath = os.path.join(f'{path}/', 'csv')
            for p in (path, txtPath, csvPath):
                if not os.path.isdir(p):
                    os.mkdir(p, 0o754)

            for n in por:
                por[n].to_csv(os.path.join(f'{txtPath}/', f'{self.speciesName}{n}.txt'), sep='\t')
                por[n].to_csv(os.path.join(f'{csvPath}/', f'{self.speciesName}{n}.csv'), sep='\t')

        return longPorosityProfileInNaturalValuesDF, longPorosityProfileNorm


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
        meansProfilesDict = {}                                            ############################
        
        for y in longPorosityDict:
            padPorosity = pad_dict_list(longPorosityDict[y])
            subDF = pd.DataFrame(data=padPorosity)
            
            if subDF.empty:
                continue
            normPorosityDF = self.getNormPorosityDF(porosityDF=subDF, normMethod=normMethod)
            normPorosityDF['MEAN'] = normPorosityDF.mean(axis=1)          ############################
            meansProfilesDict.update({y: normPorosityDF['MEAN'].tolist()})############################
            longPorosityProfilesList.append(normPorosityDF)
        meansProfilesDict = pad_dict_list(meansProfilesDict)
        AVG = pd.DataFrame(data=meansProfilesDict)############################
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
            normPorosity = self.getNormalisationPorosityProfile(clearPorosityProfilesDict[cp], convertCoef=convertCoef)

            if len(normPorosity) != reqRW:
                convertCoef = reqRW/len(normPorosity)
                normPorosity = self.getNormalisationPorosityProfile(normPorosity, convertCoef=convertCoef)
            
            normPorosityDict.update({cp: normPorosity})

        normPorosityDF = pd.DataFrame(data=normPorosityDict)
        return normPorosityDF


    def scanRoot(self) -> tuple:
        """determines the porosity profile for all trees in the library, 
        as well as their other characteristics (TRW, MaxPor, MinPor, MeanPor)
        
        Returns
        -------
        treesPorosityDict : dict
        normPorosityDict : dict
        rwDF : pd.DataFrame
            annual ring width data (in micrometers)
        maxPorosityDF : pd.DataFrame
            yearly maximum porosity data
        minPorosityDF : pd.DataFrame
            yearly minimum porosity data
        meanPorosityDF : pd.DataFrame
            yearly mean porosity data
        """
        
        sortedNames = sorted(os.listdir(self.root), key=lambda f: int(f))
        dirs = [os.path.join(self.root, f'{d}/') for d in sortedNames]

        rawPorosityDict = {}
        treesPorosityDict = {}
        normPorosityDict = {}
        rwDict = {}
        maximumPorosityDict = {}
        minimumPorosityDict = {}
        meanPorosityDict = {}

        earlyWidthDict = {}
        lateWidthDict = {}
        earlyPercDict = {}
        latePercDict = {}
        meanPorEarlyWoodDict = {}
        meanPorLateWoodDict = {}
        for d, n in zip(dirs, sortedNames):
            treeDF = self.scanSubDir(path=d)
            porosityProfilesNaturalValues = self.getPorProfilesNaturalValues(treeDF) # Профили пористости в натуральную величину
            normPorosityProfiles = self.getNormPorosityProfiles(treeDF)
            maxPorosity, minPorosity, meanPorosity = self.getPorosityCharacteristics(treeDF)
            rw = self.getRW(porosityProfilesNaturalValues)
            earlyWidthList, lateWidthList, earlyPercList, latePercList, meanPorEarlyWoodList, meanPorLateWoodList = self.getEarlyLateWidth(porosityProfilesNaturalValues)
            treeDF = smaDF(treeDF)

            rawPorosityDict.update({n: treeDF})
            treesPorosityDict.update({n: porosityProfilesNaturalValues})
            normPorosityDict.update({n: normPorosityProfiles})
            rwDict.update({n: rw})
            maximumPorosityDict.update({n: maxPorosity})
            minimumPorosityDict.update({n: minPorosity})
            meanPorosityDict.update({n: meanPorosity})

            earlyWidthDict.update({n: earlyWidthList})
            lateWidthDict.update({n: lateWidthList})
            earlyPercDict.update({n: earlyPercList})
            latePercDict.update({n: latePercList})
            meanPorEarlyWoodDict.update({n: meanPorEarlyWoodList})
            meanPorLateWoodDict.update({n: meanPorLateWoodList})

        
        maximumPorosityDict = pad_dict_list(maximumPorosityDict)
        minimumPorosityDict = pad_dict_list(minimumPorosityDict)
        meanPorosityDict = pad_dict_list(meanPorosityDict)
        rwDict = pad_dict_list(rwDict)

        earlyWidthDict = pad_dict_list(earlyWidthDict)
        lateWidthDict = pad_dict_list(lateWidthDict)
        earlyPercDict = pad_dict_list(earlyPercDict)
        latePercDict = pad_dict_list(latePercDict)
        meanPorEarlyWoodDict = pad_dict_list(meanPorEarlyWoodDict)
        meanPorLateWoodDict = pad_dict_list(meanPorLateWoodDict)

        rwDF = pd.DataFrame(data=rwDict)
        maxPorosityDF = pd.DataFrame(data=maximumPorosityDict)
        minPorosityDF = pd.DataFrame(data=minimumPorosityDict)
        meanPorosityDF = pd.DataFrame(data=meanPorosityDict)

        earlyWidthDF = pd.DataFrame(data=earlyWidthDict)
        lateWidthDF = pd.DataFrame(data=lateWidthDict)
        earlyPercDF = pd.DataFrame(data=earlyPercDict)
        latePercDF = pd.DataFrame(data=latePercDict)
        meanPorEarlyWoodDF = pd.DataFrame(data=meanPorEarlyWoodDict)
        meanPorLateWoodDF = pd.DataFrame(data=meanPorLateWoodDict)

        return treesPorosityDict, normPorosityDict, rwDF, maxPorosityDF, minPorosityDF, meanPorosityDF, rawPorosityDict, earlyWidthDF, lateWidthDF, earlyPercDF, latePercDF, meanPorEarlyWoodDF, meanPorLateWoodDF
    
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
            # convertCoef = self.normNumber / len(clearPorosity)
            normPorosity = self.getNormalisationPorosityProfile(clearPorosity, convertCoef=self.pixToMcmCoef)
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
            convertCoef = self.normNumber / len(clearPorosity)
            normPorosity = self.getNormalisationPorosityProfile(clearPorosity, convertCoef=convertCoef)

            if len(normPorosity) != self.normNumber:
                convertCoef= self.normNumber / len(normPorosity)
                normPorosity = self.getNormalisationPorosityProfile(normPorosity, convertCoef=convertCoef)
            normPorosityProfilesDict.update({col: normPorosity})

        normPorosityProfilesDF = pd.DataFrame(data=normPorosityProfilesDict)
        return normPorosityProfilesDF

    def getRW(self, porosityProfiles: pd.DataFrame):
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
                trw.append(0)
            else:
                trw.append(int(porosityProfiles[y].count()))

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
                maxPorosity.append(0)
                minPorosity.append(0)
                meanPorosity.append(0)
            else:
                maxPorosity.append(porosityProfiles[y].max())
                minPorosity.append(porosityProfiles[y].min())
                meanPorosity.append(porosityProfiles[y].mean())
        maxPorosity = maxPorosity[::-1]
        minPorosity = minPorosity[::-1]
        meanPorosity = meanPorosity[::-1]

        return maxPorosity, minPorosity, meanPorosity

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

        for y in range(minYear, self.yearStart+1):
            if y not in columns:
                earlyWidthList.append(0)
                lateWidthList.append(0)
                earlyPercList.append(0)
                latePercList.append(0)
                meanPorEarlyWoodList.append(0)
                meanPorLateWoodList.append(0)
            else:
                bestMeanVar = 99999999999
                cleanPor = porosityProfiles[y].dropna()

                meanValue = cleanPor.min()+(cleanPor.max()-cleanPor.min())/2
                th = 0
                for i in range(len(cleanPor)-1, 0, -1):
                    if cleanPor.iloc[i] >= meanValue:
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


    def getNormalisationPorosityProfile(self, porosityProfile: list, convertCoef: float | int) -> list:
        """normalizes the porosity profile along the length in accordance with the conversion coefficient

        Parameters
        ----------
        porosityProfile : List
            porosity profile
        convertCoef : float | int
            conversion factor for image normalization
        
        Returns
        -------
        normPorosity : List
        """
        normPorosity = []
        lenPorosity = len(porosityProfile)
        reqLen = mathRound(len(porosityProfile) * convertCoef) + 20
        convertCoef = reqLen / lenPorosity
        stepped = 0

        if convertCoef < 1:
            step = 1 / convertCoef

            while stepped < lenPorosity-step:
                n = int(round(step, 0))
                cur_base_index = int(round(stepped, 0))
                normPorosity.append(sum((porosityProfile[cur_base_index + i] for i in range(n))) / n)
                stepped += step

                if stepped > lenPorosity:
                    stepped = lenPorosity

        elif convertCoef > 1:
            normPorosity = []

            while len(normPorosity) != reqLen:
                if normPorosity:
                    porosityProfile = normPorosity
                    normPorosity = []
                    lenPorosity = len(porosityProfile)
                unToAdd = mathRound(reqLen - lenPorosity)

                if unToAdd > lenPorosity:
                    unToAdd = lenPorosity

                step = lenPorosity / unToAdd

                i = 0
                stepped = i + step

                for u in range(unToAdd):
                    normPorosity += porosityProfile[mathRound(i):mathRound(stepped)]
                    normPorosity += [porosityProfile[mathRound(stepped)-1]]

                    if len(normPorosity) + step > reqLen and len(normPorosity) < reqLen:
                        normPorosity += porosityProfile[int(len(normPorosity)-reqLen):]
                        break

                    stepped += step
                    i += step

        else:
            normPorosity = porosityProfile
        normPorosity = sma(normPorosity)

        return normPorosity

    def scanSubDir(self, path: str) -> pd.DataFrame:
        """
        this method scans a directory of wood microstructure images
        
        Parameters
        ----------
        path : str

        Returns
        -------
        porosityByYearsDF : pd.DataFrame
            averaged annualized porosity profiles of one tree 
        """
        print('scan subdir')
        imgsList = sorted(os.listdir(path), key=lambda f: int(f.split('.')[0]))
        
        imgsPathList = [os.path.join(path, i) for i in imgsList]
        porosityByYearsDict = {}

        for ip, im in zip(imgsPathList, imgsList):
            imageNumber = int(im.split('.')[0])
            porosityDF = self.scanImg(ip)
            porosityDF['finalPorosityProfile'] = porosityDF.mean(axis=1)
            porosityByYearsDict.update({self.yearStart-imageNumber+1: porosityDF['finalPorosityProfile'].tolist()})

        porosityByYearsDict = pad_dict_list(porosityByYearsDict)
        porosityByYearsDF = pd.DataFrame(data=porosityByYearsDict)
        return porosityByYearsDF

    def scanImg(self, imgPath: str, windowSize: int = 1000, step: int= 200) -> pd.DataFrame:
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

        binaryImage = self.getBinaryImg(imgPath, biMethod=self.biMethod)
        porosityProfilesDict = {}

        if windowSize == -1:
            # Если windowSize определен как -1, то сканируем изображение полностью, не разбивая его на окна
            porosityProfileToPix = self.getPorosityProfileToPix(binaryImage)
            porosityProfilesDict.update({0: porosityProfileToPix})
        else:
            windowsNumbers = int((len(binaryImage)//step))-int((windowSize/step))

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
            blackPixN = np.sum(biImg[x] == 255)
            pixPorosityProfile.append(blackPixN / yImgSize)

        return pixPorosityProfile


    def getBinaryImg(
        self,
        imgPath: str, 
        blurType: str='Median', 
        biMethod: str='Otsu', 
        gammaEqualisation: bool=False, 
        ksize: int=3,
        constantTh=235
        ) -> np.ndarray:
        """
        converts the image into binary format 
        (default, using the Otsu method)
        
        Parameters
        ----------
        imgPath : str
            path to image
        blurType: str
            Choises: Gaussian (default), Median, Normalized Box Filter (NBF), Bilateral
        biMethod: str
            Choises: Otsu, Mean, Gaussian
        gammaEqualisation: bool
            if True, the image is gamma-corrected
        ksize: int
            Kernel size

        Returns
        -------
        biImg: np.ndarray
        """
        img = cv2.imread(imgPath)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        if gammaEqualisation:
            img = cv2.equalizeHist(img)

        # Choice bluring method
        if blurType == 'Gaussian':
            img = cv2.GaussianBlur(img, (ksize, ksize), 0)
        elif blurType == 'Median':
            img= cv2.medianBlur(img, ksize)
        elif blurType == 'NBF':
            img = cv2.blur(img, (ksize, ksize)) 
        elif blurType == 'Bilateral':
            img = cv2.bilateralFilter(img, 11, 41, 21)
        else:
            pass

        # Choice thresholding method
        if biMethod == 'Otsu':
            th, biImg = cv2.threshold(img, 230, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        elif biMethod == 'Mean':
            biImg = cv2.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY,11,2)
        elif biMethod == 'Gaussian':
            biImg = cv2.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY,11,2)
        elif biMethod == 'Triangle':
            thresh = filters.threshold_triangle(img)
            biImg = img > thresh
            biImg = biImg.astype(np.uint8) * 255
        else:
            th, biImg = cv2.threshold(img, constantTh, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        return biImg