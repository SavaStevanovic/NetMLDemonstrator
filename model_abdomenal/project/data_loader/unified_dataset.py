import numpy as np
import torch
from data_loader.hubman_dataset import RSNADataset
from torch.utils.data import Dataset
import torchvision


class TransformedDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self._dataset = dataset
        self._transform = transform

    def __getitem__(self, index):
        x, y = self._dataset[index]
        aug_data = self._transform(
            image=np.array(x)
        )
        x = torch.tensor(aug_data.pop("image").astype("float32"))
        return x, y

    def __len__(self):
        return len(self._dataset)


class UnifiedKeypointDataset(Dataset):
    def __init__(self, debug=False):
        self.debug = debug
        self.datasets = [
            RSNADataset("/Data/train_images", "/Data/train.csv"),
        ]
        self.labels = sorted(list(set(sum([x.labels for x in self.datasets], []))))
        self.data_ids = [
            (i, j)
            for i, _ in enumerate(self.datasets)
            for j in range(len(self.datasets[i]))
        ]

    def __len__(self):
        return len(self.data_ids)

    def __getitem__(self, idx):
        identifier = self.data_ids[idx]
        data = self.datasets[identifier[0]][identifier[1]]
        return data