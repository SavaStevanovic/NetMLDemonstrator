import torch
from data_loader.coco_dataset import CocoDataset
from data_loader.cityscapes_dataset import CityscapesDataset
from data_loader.voc_dataset import VOCDataset
from data_loader.ade_dataset import ADEChallengeData2016
import torchvision.transforms as transforms
from data_loader import augmentation
from visualization import output_transform
import multiprocessing as mu
from torch.utils.data import Dataset, DataLoader
from pycocotools.coco import COCO
import os


class UnifiedKeypointDataset(Dataset):
    def __init__(self, train, depth, debug=False):
        self.debug = debug
        self.train = train
        train_datasets = [
                VOCDataset('train', 'Voc'),
                ADEChallengeData2016('train', 'ADEChallengeData2016'),
                CityscapesDataset('train', 'fine', 'Cityscapes'),
                CityscapesDataset('train', 'coarse', 'Cityscapes'),
                CityscapesDataset('train_extra', 'coarse', 'Cityscapes'),
                CocoDataset('train2017', 'annotations_trainval2017/annotations/instances_train2017.json'),
            ]
        self.labels = sorted(list(set(sum([x.labels for x in train_datasets],[]))))
        if train:
            self.datasets = train_datasets
            
            self.transforms = augmentation.PairCompose([
                augmentation.RandomHorizontalFlipTransform(),
                augmentation.RandomResizeTransform(),
                augmentation.RandomCropTransform((384, 384)),
                augmentation.RandomNoiseTransform(),
                augmentation.RandomColorJitterTransform(),
                augmentation.RandomBlurTransform(),
                augmentation.JPEGcompression(95),
                augmentation.OneHotTransform(len(self.labels)+1),
                augmentation.OutputTransform()]
            )

        if not train:
            self.transforms = augmentation.PairCompose([
                augmentation.PaddTransform(2**depth),
                augmentation.OneHotTransform(len(self.labels)+1),
                augmentation.OutputTransform()]
            )
            self.datasets = [
                VOCDataset('val', 'Voc'),
                ADEChallengeData2016('val', 'ADEChallengeData2016'),
                CityscapesDataset('val', 'fine', 'Cityscapes'),
                CityscapesDataset('val', 'coarse', 'Cityscapes'),
                CocoDataset('val2017', 'annotations_trainval2017/annotations/instances_val2017.json'),
            ]

        self.data_ids = [(i, j) for i, dataset in enumerate(self.datasets) for j in range(len(self.datasets[i]))]
        self.selector = [[self.labels.index(u) for u in x.labels] for x in self.datasets]

    def __len__(self):
        if self.debug==1:
            return 5
        else:
            return len(self.data_ids)

    def __getitem__(self, idx):
        identifier = self.data_ids[idx]
        data = self.datasets[identifier[0]][identifier[1]]
        if self.transforms:
            data = self.transforms(*data, identifier[0])
        return data