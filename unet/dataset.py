from torch.utils.data import Dataset
import cv2


class SegmentationDataset(Dataset):
    def __init__(self, imagePaths, maskPaths, transforms, mask_transforms=None):
        self.imagePaths = imagePaths
        self.maskPaths = maskPaths
        self.transforms = transforms
        # If no dedicated mask pipeline is given, fall back to the image one.
        # Providing a separate pipeline is recommended so that mask pixels are
        # resized with NEAREST interpolation (preserving binary 0/255 values)
        # and so that ToPILImage cannot accidentally promote a (H,W) uint8
        # array to RGB mode, which would cause a channel mismatch with the
        # 1-channel model output.
        self.mask_transforms = mask_transforms if mask_transforms is not None else transforms

    def __len__(self):
        return len(self.imagePaths)

    def __getitem__(self, idx):
        # load image as grayscale (H, W)
        image = cv2.imread(self.imagePaths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # load mask as grayscale (H, W); flag 0 = IMREAD_GRAYSCALE
        mask = cv2.imread(self.maskPaths[idx], 0)

        if self.transforms is not None:
            image = self.transforms(image)
        if self.mask_transforms is not None:
            mask = self.mask_transforms(mask)

        return (image, mask)