from .app import PICDens
from unet.train import train
from unet.predict import make_predictions

import torch

def main():
    # if you want use UNET model, you can train UNET on your images and masks
    # trainImgsPath = 'path/to/imgs/'
    # maskPath = 'path/to/masks/'
    outputPath = 'path/to/output/'
    saveModelPath = 'path/to/model/'
    INPUT_IMAGE_WIDTH = 128
    INPUT_IMAGE_HEIGHT = 128
    saveModelPath = saveModelPath
    outputPath = outputPath
    TEST_SPLIT = 0.15
    BATCH_SIZE = 32

    # determine the device to be used for training and evaluation
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    # determine if we will be pinning memory during data loading
    PIN_MEMORY = True if DEVICE == "cuda" else False
    INIT_LR = 0.001
    NUM_EPOCHS = 40
    THRESHOLD = 0.5

    train(
        imgsPath=trainImgsPath, 
        maskPath=maskPath, 
        outputPath=outputPath,
        saveModelPath=saveModelPath,
        INPUT_IMAGE_WIDTH=INPUT_IMAGE_WIDTH,
        INPUT_IMAGE_HEIGHT=INPUT_IMAGE_HEIGHT,
        TEST_SPLIT=TEST_SPLIT,
        BATCH_SIZE=BATCH_SIZE,
        DEVICE=DEVICE,
        PIN_MEMORY=PIN_MEMORY,
        INIT_LR=INIT_LR,
        NUM_EPOCHS=NUM_EPOCHS
        )

    model = torch.load(saveModelPath).to(DEVICE)
    imgPath = 'path/to/img/'

    # Predict one image
    predMask = make_predictions(
        model=model, 
        imagePath=imgPath, 
        INPUT_IMAGE_HEIGHT=INPUT_IMAGE_HEIGHT, 
        INPUT_IMAGE_WIDTH=INPUT_IMAGE_WIDTH, 
        DEVICE=DEVICE, 
        THRESHOLD=THRESHOLD, 
        image=None,
        plotMod=False
        )

    # Or binarization root
    bi = BI(biMethod='UNET', modelPath="path/to/model.pth")


    # Start scan
    savePath = "path/to/save_dir/"
    root = "path/to/img_root_dir/"
    pixToMcmCoef = 0.42604
    speciesName = 'PS' # Pinus Sylvestris
    normNumber = 100
    yearStart = 2025
    biMethod='UNET'
    modelPath="path/to/model.pth"
    usePredBIImgs=True
    biImgsPath="path/to/biImgs/"
    smaInterval=90
    gapValue=50

    pD = PICDens(
        savePath=savePath, 
        root=root, 
        pixToMcmCoef=pixToMcmCoef, 
        speciesName=speciesName, 
        normNumber=normNumber, 
        yearStart=yearStart, 
        biMethod=biMethod, 
        modelPath=modelPath,
        usePredBIImgs=usePredBIImgs,
        biImgsPath=biImgsPath,
        smaInterval=smaInterval,
        gapValue=gapValue
        )
    pD.startScan()