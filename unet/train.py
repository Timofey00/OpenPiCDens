from unet.dataset import SegmentationDataset
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from torchvision import transforms
from imutils import paths
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import time
import segmentation_models_pytorch as smp


def train(
    imgs_path: str,
    mask_path: str,
    output_path: str,
    save_model_path: str,
    input_image_width: int,
    input_image_height: int,
    test_split: float | int,
    batch_size: int,
    device: str,
    pin_memory: bool,
    init_lr: float | int,
    num_epochs: int,
) -> smp.Unet:
    """Train a U-Net segmentation model on wood micrograph image–mask pairs.

    Parameters
    ----------
    imgs_path : str
        Directory containing training images (.jpg).
    mask_path : str
        Directory containing binary mask images (.png).
    output_path : str
        Directory where the loss plot is saved.
    save_model_path : str
        File path for saving the best model checkpoint (.pth).
    input_image_width : int
        Width to which images are resized before training.
    input_image_height : int
        Height to which images are resized before training.
    test_split : float
        Fraction of data held out for validation (e.g. ``0.15``).
    batch_size : int
        Number of samples per gradient update.
    device : str
        PyTorch device string: ``"cuda"`` or ``"cpu"``.
    pin_memory : bool
        If ``True``, DataLoader copies tensors into CUDA pinned memory.
        Automatically disabled when running on CPU to avoid deadlocks on
        Windows (PyTorch attempts to allocate CUDA pinned memory even on
        CPU-only machines, which blocks indefinitely).
    init_lr : float
        Initial learning rate for the Adam optimiser.
    num_epochs : int
        Number of training epochs.

    Returns
    -------
    smp.Unet
        The model checkpoint with the lowest validation loss.
    """
    # pin_memory is only meaningful (and safe) when a CUDA device is available;
    # on CPU-only Windows machines it causes the DataLoader to deadlock.
    cuda_available = torch.cuda.is_available()
    if pin_memory and not cuda_available:
        print("[WARNING] pin_memory=True requested but CUDA is not available. "
              "Disabling pin_memory to prevent DataLoader deadlock on Windows.")
        pin_memory = False

    # load the image and mask filepaths in a sorted manner
    imagePaths = sorted(list(paths.list_images(imgs_path)))
    maskPaths = sorted(list(paths.list_images(mask_path)))

    # partition the data into training and testing splits
    split = train_test_split(imagePaths, maskPaths,
        test_size=test_split, random_state=42)

    # unpack the data split
    (trainImages, testImages) = split[:2]
    (trainMasks, testMasks) = split[2:]

    # Transforms for input images (grayscale numpy HxW -> tensor 1xHxW)
    img_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((input_image_height, input_image_width)),
        transforms.ToTensor(),
    ])

    # Transforms for masks: NEAREST interpolation preserves binary 0/255 values;
    # separate pipeline prevents ToPILImage from promoting (H,W) uint8 to RGB
    # in some torchvision versions, which would cause a channel mismatch.
    mask_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(
            (input_image_height, input_image_width),
            interpolation=transforms.InterpolationMode.NEAREST,
        ),
        transforms.ToTensor(),
    ])

    # create the train and test datasets
    trainDS = SegmentationDataset(
        imagePaths=trainImages, maskPaths=trainMasks,
        transforms=img_transforms, mask_transforms=mask_transforms,
    )
    testDS = SegmentationDataset(
        imagePaths=testImages, maskPaths=testMasks,
        transforms=img_transforms, mask_transforms=mask_transforms,
    )

    print(f"[INFO] found {len(trainDS)} examples in the training set...")
    print(f"[INFO] found {len(testDS)} examples in the test set...")

    # num_workers=0: load data in the main process.
    # On Windows, num_workers > 0 uses the "spawn" multiprocessing start
    # method, which requires all DataLoader usage to be inside
    # `if __name__ == "__main__"`. Since train() is called as a library
    # function, workers > 0 would cause a silent hang or recursive process
    # spawning. Use 0 for safety; increase only if profiling shows I/O is
    # the bottleneck and the call site is properly guarded.
    trainLoader = DataLoader(trainDS, shuffle=True,
        batch_size=batch_size, pin_memory=pin_memory, num_workers=0)
    testLoader = DataLoader(testDS, shuffle=False,
        batch_size=batch_size, pin_memory=pin_memory, num_workers=0)

    # initialize our UNet model
    # NOTE: encoder_weights="imagenet" triggers a ResNet-34 download on first
    # run (~90 MB). The model is kept on CPU until training starts; .to(device)
    # is called on tensors per-batch to avoid a slow CUDA init at startup.
    print(f"[INFO] loading U-Net (encoder weights may be downloaded on first run)...")
    unet = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=1,   # grayscale micrographs
        classes=1,       # binary segmentation: pore / not-pore
    )
    unet = unet.to(device)
    print(f"[INFO] model ready on {device}.")

    # initialize loss function and optimizer
    lossFunc = BCEWithLogitsLoss()
    opt = Adam(unet.parameters(), lr=init_lr)

    # calculate steps per epoch for training and test set
    trainSteps = len(trainDS) // batch_size
    testSteps = len(testDS) // batch_size

    # initialize a dictionary to store training history
    H = {"train_loss": [], "test_loss": []}

    # loop over epochs
    print("[INFO] training the network...")
    startTime = time.time()

    bestTestLoss = float("inf")
    bestModel = None

    for e in tqdm(range(num_epochs), desc="Epochs"):
        # set the model in training mode
        unet.train()

        totalTrainLoss = 0
        totalTestLoss = 0

        # loop over the training set
        trainBar = tqdm(trainLoader, desc=f"  Epoch {e+1}/{num_epochs} train",
                        leave=False, unit="batch")
        for (x, y) in trainBar:
            (x, y) = (x.to(device), y.to(device))
            pred = unet(x)
            loss = lossFunc(pred, y)

            opt.zero_grad()
            loss.backward()
            opt.step()

            totalTrainLoss += loss
            trainBar.set_postfix(loss=f"{loss.item():.4f}")

        # switch off autograd
        with torch.no_grad():
            unet.eval()

            # loop over the validation set
            valBar = tqdm(testLoader, desc=f"  Epoch {e+1}/{num_epochs} val  ",
                          leave=False, unit="batch")
            for (x, y) in valBar:
                (x, y) = (x.to(device), y.to(device))
                pred = unet(x)
                totalTestLoss += lossFunc(pred, y)

        # calculate the average training and validation loss
        avgTrainLoss = totalTrainLoss / trainSteps
        avgTestLoss = totalTestLoss / testSteps

        if avgTestLoss < bestTestLoss:
            bestModel = unet
            bestTestLoss = avgTestLoss

        # update our training history
        H["train_loss"].append(avgTrainLoss.detach().cpu().numpy())
        H["test_loss"].append(avgTestLoss.detach().cpu().numpy())

        print("[INFO] EPOCH: {}/{}".format(e + 1, num_epochs))
        print("Train loss: {:.6f}, Test loss: {:.4f}".format(
            avgTrainLoss, avgTestLoss))

    # display the total time needed to perform the training
    endTime = time.time()
    print("[INFO] total time taken to train the model: {:.2f}s".format(
        endTime - startTime))

    torch.save(bestModel, save_model_path)

    # plot the training loss
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(H["train_loss"], label="train_loss")
    plt.plot(H["test_loss"], label="test_loss")
    plt.title("Training Loss on Dataset")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss")
    plt.legend(loc="lower left")
    plt.savefig(output_path + "lossPlot.jpeg")
    return bestModel