import os

from PIL import Image
from torch.utils.data import Dataset

import numpy as np

IMG_EXTENSIONS = ["png", "jpg"]

class ImagetoImageDataset(Dataset):
    def __init__(self, domainA_dir, domainB_dir, transforms=None):
        self.imagesA = [os.path.join(domainA_dir, x) for x in os.listdir(domainA_dir) if
                        x.lower().endswith(tuple(IMG_EXTENSIONS))]
        self.imagesB = [os.path.join(domainB_dir, x) for x in os.listdir(domainB_dir) if
                        x.lower().endswith(tuple(IMG_EXTENSIONS))]

        self.transforms = transforms

        self.lenA = len(self.imagesA)
        self.lenB = len(self.imagesB)

    def __len__(self):
        return max(self.lenA, self.lenB)

    def __getitem__(self, idx):
        idx_a = idx_b = idx
        if idx_a >= self.lenA:
            idx_a = np.random.randint(self.lenA)
        if idx_b >= self.lenB:
            idx_b = np.random.randint(self.lenB)
        
        imageA = np.array(Image.open(self.imagesA[idx_a]).convert("RGB"))
        imageB = np.array(Image.open(self.imagesB[idx_b]).convert("RGB"))

        if self.transforms is not None:
            imageA = self.transforms(imageA)
            imageB = self.transforms(imageB)

        return imageA, imageB
