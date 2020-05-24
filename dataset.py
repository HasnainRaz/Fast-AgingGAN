import os

from PIL import Image
from torch.utils.data import Dataset


class ImagetoImageDataset(Dataset):
    def __init__(self, domainA_dir, domainB_dir, transforms=None):
        self.imagesA = [os.path.join(domainA_dir, x) for x in os.listdir(domainA_dir) if
                        x.endswith('.png') or x.endswith('jpg')]
        self.imagesB = [os.path.join(domainB_dir, x) for x in os.listdir(domainB_dir) if
                        x.endswith('.png') or x.endswith('jpg')]

        self.transforms = transforms

    def __len__(self):
        return min(len(self.imagesA), len(self.imagesB))

    def __getitem__(self, idx):
        imageA = Image.open(self.imagesA[idx])
        imageB = Image.open(self.imagesB[idx])

        if self.transforms is not None:
            imageA = self.transforms(imageA)
            imageB = self.transforms(imageB)

        return imageA, imageB
