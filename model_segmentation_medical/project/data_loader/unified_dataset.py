import torch
from data_loader.hubman_dataset import HubmapDataset
from data_loader import augmentation
from torch.utils.data import Dataset
import os
import random


class UnifiedKeypointDataset(Dataset):
    def __init__(self, train, depth, debug=False):
        self.debug = debug
        self.train = train
        train_datasets = [
            HubmapDataset("/Data/train", "/Data/polygons.jsonl"),
        ]

        self.labels = sorted(list(set(sum([x.labels for x in train_datasets], []))))
        if train:
            self.datasets = train_datasets

        if not train:
            self.datasets = [
                HubmapDataset("/Data/validation", "/Data/polygons.jsonl"),
            ]
        self.supported_labels = self.labels
        self.selector = [
            [
                self.supported_labels.index(u) + 1 if u in self.supported_labels else 0
                for u in x.labels
            ]
            for x in self.datasets
        ]
        if train:
            self.transforms = augmentation.PairCompose(
                [
                    augmentation.RandomHorizontalFlipTransform(),
                    augmentation.RandomCropTransform((448, 448)),
                    augmentation.RandomNoiseTransform(),
                    augmentation.RandomColorJitterTransform(),
                    augmentation.OneHotTransform(len(self.supported_labels)),
                    augmentation.OutputTransform(),
                ]
            )

        if not train:
            self.transforms = augmentation.PairCompose(
                [
                    augmentation.PaddTransform(2**depth),
                    augmentation.OneHotTransform(len(self.supported_labels)),
                    augmentation.OutputTransform(),
                ]
            )

        if self.debug == 1:
            self.data_ids = [
                (i, j) for i, dataset in enumerate(self.datasets) for j in range(50)
            ]
        else:
            self.data_ids = [
                (i, j)
                for i, _ in enumerate(self.datasets)
                for j in range(len(self.datasets[i]))
            ]

        random.seed(0)
        self.color_set = [
            [random.random() for _ in range(3)] for _ in range(len(self.labels))
        ]

    def __len__(self):
        return len(self.data_ids)

    def __getitem__(self, idx):
        identifier = self.data_ids[idx]
        data = self.datasets[identifier[0]][identifier[1]]
        if self.transforms:
            data = self.transforms(*data, identifier[0])
        return data
