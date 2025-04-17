# USAGE
# python train.py
# import the necessary packages
from unet.dataset import SegmentationDataset
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from torchvision import transforms, models
from torchvision.models.detection import maskrcnn_resnet50_fpn
from imutils import paths
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import time
import os
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.encoders import get_preprocessing_fn


def train(
    imgsPath: str, 
    maskPath: str, 
    outputPath: str, 
    saveModelPath: str, 
    INPUT_IMAGE_WIDTH: int, 
    INPUT_IMAGE_HEIGHT: int, 
    TEST_SPLIT: float | int, 
    BATCH_SIZE: int, 
    DEVICE: str, 
    PIN_MEMORY: bool, 
    INIT_LR: float | int, 
    NUM_EPOCHS: int
    ) -> smp.Unet:
    """
    function for model training

    Parameters
    ----------
    imgsPath: str
        images path
    maskPath: str
        masks path
    outputPath: str
        path for information file
    saveModelPath: str
        path for save model
    INPUT_IMAGE_WIDTH: int
        image width
    INPUT_IMAGE_HEIGHT: int
        image height
    TEST_SPLIT: float | int
        test image fraction
    BATCH_SIZE: int
        batch size
    DEVICE: str
        device
    PIN_MEMORY: bool
        If True, the data loader will copy tensors into CUDA pinned memory before returning them
    INIT_LR: float | int
        learning rate
    NUM_EPOCHS: int
        number of epochs

    Returns
    ----------
    bestModel: smp.Unet
        least-loss model

    """ 
    # load the image and mask filepaths in a sorted manner
    imagePaths = sorted(list(paths.list_images(imgsPath)))
    maskPaths = sorted(list(paths.list_images(maskPath)))

    # partition the data into training and testing splits using 85% of
    # the data for training and the remaining 15% for testing
    split = train_test_split(imagePaths, maskPaths,
    	test_size=TEST_SPLIT, random_state=42)

    # unpack the data split
    (trainImages, testImages) = split[:2]
    (trainMasks, testMasks) = split[2:]

    # define transformations
    transforms_ = transforms.Compose([transforms.ToPILImage(),
        transforms.Resize((INPUT_IMAGE_HEIGHT,
            INPUT_IMAGE_WIDTH)),
        transforms.ToTensor()])

    # create the train and test datasets
    trainDS = SegmentationDataset(imagePaths=trainImages, maskPaths=trainMasks,
        transforms=transforms_)

    testDS = SegmentationDataset(imagePaths=testImages, maskPaths=testMasks,
        transforms=transforms_)

    print(f"[INFO] found {len(trainDS)} examples in the training set...")
    print(f"[INFO] found {len(testDS)} examples in the test set...")

    # create the training and test data loaders
    trainLoader = DataLoader(trainDS, shuffle=True,
        batch_size=BATCH_SIZE, pin_memory=PIN_MEMORY,)

    testLoader = DataLoader(testDS, shuffle=False,
        batch_size=BATCH_SIZE, pin_memory=PIN_MEMORY,)

    # initialize our UNet model
    # unet = maskrcnn_resnet50_fpn(weights='DEFAULT').to(config.DEVICE)
    isMarkCNN = True
    unet = smp.Unet(
        encoder_name="resnet34",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
        in_channels=1,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        classes=3,                    # model output channels (number of classes in your dataset)
    )

    # initialize loss function and optimizer
    lossFunc = BCEWithLogitsLoss()
    opt = Adam(unet.parameters(), lr=INIT_LR)

    # calculate steps per epoch for training and test set
    trainSteps = len(trainDS) // BATCH_SIZE
    testSteps = len(testDS) // BATCH_SIZE

    # initialize a dictionary to store training history
    H = {"train_loss": [], "test_loss": []}

    # loop over epochs
    print("[INFO] training the network...")

    startTime = time.time()

    bestTestLoss = 100
    bestModel = None

    for e in tqdm(range(NUM_EPOCHS)):
        # set the model in training mode
        unet.train()

        # initialize the total training and validation loss
        totalTrainLoss = 0
        totalTestLoss = 0

        # loop over the training set
        for (i, (x, y)) in enumerate(trainLoader):
            # send the input to the device
            (x, y) = (x.to(DEVICE), y.to(DEVICE))
            # perform a forward pass and calculate the training loss
            pred = unet(x)
            loss = lossFunc(pred, y)

            # first, zero out any previously accumulated gradients, then
            # perform backpropagation, and then update model parameters
            opt.zero_grad()
            loss.backward()
            opt.step()

            # add the loss to the total training loss so far
            totalTrainLoss += loss

        # switch off autograd
        with torch.no_grad():

            # set the model in evaluation mode
            unet.eval()

            # loop over the validation set
            for (i, (x, y)) in enumerate(trainLoader):
                # send the input to the device
                (x, y) = (x.to(DEVICE), y.to(DEVICE))

                # make the predictions and calculate the validation loss
                pred = unet(x)
                # print(pred.shape, y.shape)
                totalTestLoss += lossFunc(pred, y)

        # calculate the average training and validation loss
        avgTrainLoss = totalTrainLoss / trainSteps
        avgTestLoss = totalTestLoss / testSteps

        if bestTestLoss >  avgTestLoss:
            bestModel = unet
            bestTestLoss = avgTestLoss

        # update our training history
        H["train_loss"].append(avgTrainLoss.detach().numpy())
        H["test_loss"].append(avgTestLoss.detach().numpy())

        # print the model training and validation information
        print("[INFO] EPOCH: {}/{}".format(e + 1, NUM_EPOCHS))
        print("Train loss: {:.6f}, Test loss: {:.4f}".format(
            avgTrainLoss, avgTestLoss))

    # display the total time needed to perform the training
    endTime = time.time()
    print("[INFO] total time taken to train the model: {:.2f}s".format(
        endTime - startTime))

    torch.save(bestModel, saveModelPath)

    # plot the training loss
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(H["train_loss"], label="train_loss")
    plt.plot(H["test_loss"], label="test_loss")
    plt.title("Training Loss on Dataset")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss")
    plt.legend(loc="lower left")
    plt.savefig(outputPath + 'lossPlot.jpeg')
    return bestModel