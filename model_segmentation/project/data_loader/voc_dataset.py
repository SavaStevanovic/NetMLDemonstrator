from torchvision import datasets
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np

class VOCDataset(Dataset):
    def __init__(self, mode, directory):
       self.data = datasets.VOCSegmentation(os.path.join('/Data/segmentation', directory), image_set = mode, download = False)
       self.labels =[
            'airplane',
            'bicycle',
            'bird',
            'boat',
            'bottle',
            'bus',
            'car',
            'cat',
            'chair',
            'cow',
            'diningtable',
            'dog',
            'horse',
            'motorbike',
            'person',
            'pottedplant',
            'sheep',
            'sofa',
            'train',
            'tv'
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, smnt = self.data[idx]
        return img, smnt

