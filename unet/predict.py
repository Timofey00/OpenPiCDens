# USAGE
# python predict.py
import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2
import os
import random
import segmentation_models_pytorch as smp

def make_predictions(
	model: smp.Unet, 
	INPUT_IMAGE_HEIGHT: int, 
	INPUT_IMAGE_WIDTH: int, 
	DEVICE: str, 
	THRESHOLD: float,
	imagePath: str,
	image: np.array =None, 
	) -> np.array:
	"""
	function for model training

	Parameters
	----------
	model: smp.Unet
		predictive model
	imagePath: str
		if it's not None, uses the image in the path for prediction
	INPUT_IMAGE_HEIGHT: int
		image height
	INPUT_IMAGE_WIDTH: int
		image width
	DEVICE: str
		device
	THRESHOLD: float
		minimum probability required for a positive prediction
	image: np.array | None =None
		if it's not None, uses this image to predict

	Returns
	----------
	predMask: np.array
		predicted mask

	""" 

	# set model to evaluation mode
	model.eval()
	# turn off gradient tracking
	with torch.no_grad():
		# load the image from disk, swap its color channels, cast it
		# to float data type, and scale its pixel values
		if imagePath:
			image = cv2.imread(imagePath)
		else:
			image = np.expand_dims(image, 2)
			image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
		image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
		image = image.astype("float32") / 255.0
		
		# resize the image and make a copy of it for visualization
		image = cv2.resize(image, (INPUT_IMAGE_HEIGHT, INPUT_IMAGE_HEIGHT))
		
		orig = image.copy()
		# find the filename and generate the path to ground truth
		# mask

		# make the channel axis to be the leading one, add a batch
		# dimension, create a PyTorch tensor, and flash it to the
		# current device
		image = np.expand_dims(image, 0)
		image = np.expand_dims(image, 0)
		image = torch.from_numpy(image).to(DEVICE)

		# make the prediction, pass the results through the sigmoid
		# function, and convert the result to a NumPy array
		predMask = model(image).squeeze()
		predMask = torch.sigmoid(predMask)
		predMask = predMask.cpu().numpy()
		predMask = predMask.transpose(1, 2, 0)

		# filter out the weak predictions and convert them to integers
		predMask = (predMask > THRESHOLD) * 255
		predMask = predMask.astype(np.uint8)
		
		return predMask