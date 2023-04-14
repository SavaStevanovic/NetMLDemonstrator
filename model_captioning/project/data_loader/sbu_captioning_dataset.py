import os
import torch
import torchvision.datasets as datasets
from PIL import Image


class SBUCaptioningDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir

        # Load SBU dataset
        self.dataset = datasets.SBU(
            root=self.root_dir, download=False)

    def __getitem__(self, index):
        img, label = self.dataset[index]
        return img, label, [label]

    def __len__(self):
        return len(self.dataset)
