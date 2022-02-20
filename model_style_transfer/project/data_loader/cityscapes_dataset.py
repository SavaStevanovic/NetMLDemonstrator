from torchvision import datasets
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import numpy as np


class CityscapesDataset(Dataset):
    def __init__(self, split, mode, directory):
        self.data = datasets.Cityscapes(os.path.join(
            '/Data/segmentation', directory, 'data'), split=split, mode=mode, target_type='semantic')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, _ = self.data[idx]
        return img
