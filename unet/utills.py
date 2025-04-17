import cv2
import numpy as np
import random
import string
import os
import matplotlib.pyplot as plt
from skimage import exposure
from unet.predict import make_predictions
from typing import Tuple
import torch
import segmentation_models_pytorch as smp

def cutAndPadImg(newNamePrefix: str, imgPath: str, savePath: str, size: int, isMask: bool=False) -> Tuple[list, int, int, int, int]:
	"""
	this method divides the image into pieces with the specified size 
	(if necessary, it supplements them with black color to the required size). 
	After that, if savePath is specified, it saves cropped images.

	Parameters
	----------
	newNamePrefix: str
		prefix for cropped image names
	imgPath: str
		image path
	savePath: str
		save path
	size: int
		desired size of cropped images
	isMask: bool=False
		if True, reads the image in binary format

	Returns
	----------
	cuts : list
		list with cropped images of a given size
	yLen : int
		image width
	xLen : int
		image height
	padRight : int
		augmented image width size
	padBottom : int
		augmented image height size
	"""	
	img = cv2.imread(imgPath)
	img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
	if isMask:
		th, img = cv2.threshold(img, 230, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

	img = cv2.medianBlur(img, 7)

	padRight = (len(img[0]) // size + 1) * size - len(img[0])
	padBottom = ((len(img) // size + 1) * size - len(img))

	padImg = np.pad(img, ((0, padBottom), (0, padRight)), mode='constant', constant_values=0)

	dy = 0
	dx = 0

	i = 0
	cuts = []

	for dy in range(0, len(padImg), size):
		for dx in range(0, len(padImg[0]), size):
			cutImg = padImg[dy: dy+size, dx: dx+size]
			newName = f'{newNamePrefix}_{i}'
			if savePath:
				cv2.imwrite(os.path.join(savePath, f'{newName}.jpg'), cutImg)
			cuts.append(cutImg)
			i += 1
	yLen = len(img)
	xLen = len(img[0])
	return cuts, len(img), len(img[0]), padRight, padBottom


def cutAndPadImgsAndMasks(imgRoot: str, maskRoot: str, savePathImg: str, savePathMask: str, size: int) -> None:
	"""
	this function automatically crops images and masks to smaller images with specified dimensions

	Parameters
	----------
	imgRoot: str
		images path
	maskRoot: str
		masks path
	savePathImg: str
		path for saving cropped images
	savePathMask: str
		path for saving cropped masks
	size: int
		cropped image size

	"""	
	maskNames = os.listdir(maskRoot)

	for n in maskNames:
		splitName = n.split('.')[0]
		maskP = os.path.join(maskRoot, f'{splitName}.png')
		imgP = os.path.join(imgRoot, f'{splitName}.jpg')
		newNamePrefix = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(10))
		cutAndPadImg(newNamePrefix=newNamePrefix, imgPath=imgP, savePath=savePathImg, size=size, isMask=False)
		cutAndPadImg(newNamePrefix=newNamePrefix, imgPath=maskP, savePath=savePathMask, size=size, isMask=True)

def moveImgs(root: str, toPath: str, maxSizeY: int=512, maxSizeX: int=512) -> None:
	"""
	recursively traverses root, moves all images from root to a single folder, 
	and splits them into parts with sizes not exceeding maxSizeY and maxSizeX

	Parameters
	----------
	root: str
		processed path
	toPath: str
		path for moving images
	maxSizeY: int
		maximum image height
	maxSizeX: int
		maximum image width

	"""
	sortedDirs = os.listdir(root)
	dirs = [os.path.join(root, f'{d}/') for d in sortedDirs]

	for d in dirs:
		sortedNames = os.listdir(d)
		for n in sortedNames:
			dirImg = os.path.join(d, n)
			img = cv2.imread(dirImg)
			ySize = img.shape[0]
			xSize = img.shape[1]
			if ySize > maxSizeY:
				print(img.shape)
				for y in range(ySize//maxSizeY+1):
					for x in range(xSize//maxSizeX+1):
						newName = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(5)) + '.jpg'
						if y == ySize//maxSizeY:
							newImg = img[y*maxSizeY:ySize, x*maxSizeX:(x+1)*maxSizeX]
						if x == xSize//maxSizeX:
							newImg = img[y*maxSizeY:(y+1)*maxSizeY, x*maxSizeX:xSize]
						else:
							newImg = img[y*maxSizeY:(y+1)*maxSizeY, x*maxSizeX:(x+1)*maxSizeX]
						cv2.imwrite(os.path.join(toPath, newName), newImg)

def joinDivideImgs(imgsList: list, ySize: int, xSize: int, padRight: int, padBottom: int) -> np.array:
	"""
	this method merges the split image

	Parameters
	----------
	imgsList: str
		list with cropped images of a given size
	ySize: str
		image width
	xSize: int
		image height
	padRight : int
		augmented image width size
	padBottom : int
		augmented image height size

	Returns
	----------
	img : np.array
		merged image

	"""	
	joinRow = []
	rows = []
	size = len(imgsList[0])
	a = 0 if xSize+padRight == size else 1
	for i, im in enumerate(imgsList):
		if not len(joinRow):
			joinRow = im
			continue
		joinRow = np.concatenate((joinRow, im), axis=a)
		if len(joinRow[0]) == xSize+padRight:
			joinRow = joinRow[:, :xSize]
			rows.append(joinRow)
			joinRow = []
			
	img = np.concatenate(rows, axis=0)
	img = img[:ySize]
	return img


def predictImgMask(imgPath: str, saveMaskPath: str, divideSize: int, model: smp.Unet) -> np.array:
	"""
	this method predicts the image mask

	Parameters
	----------
	imgPath: str
		image path
	saveMaskPath: str
		save path
	divideSize: int
		split image size
	model : int
		predictive model
	Returns
	----------
	mask : np.array
		predicted mask

	"""	
	cuts, ySize, xSize, padRight, padBottom = cutAndPadImg(newNamePrefix=None, imgPath=imgPath, savePath=None, size=divideSize, isMask=False)

	predMasksList = []

	print("[INFO] start prediction...")
	for i in cuts:
		DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
		mask = make_predictions(model=model, imagePath=None, INPUT_IMAGE_HEIGHT=divideSize, INPUT_IMAGE_WIDTH=divideSize, DEVICE=DEVICE, THRESHOLD=0.5, image=i)
		predMasksList.append(mask)

	mask = joinDivideImgs(predMasksList, ySize, xSize, padRight, padBottom)
	# if saveMaskPath:
	# 	cv2.imwrite(saveMaskPath, mask)
	return mask