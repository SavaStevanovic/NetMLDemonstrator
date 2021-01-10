import torch
from data_loader.coco_dataset import CocoDataset, clasess_inds
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
        self.clasess_inds = clasess_inds
        if train:
            self.transforms = augmentation.PairCompose([
                augmentation.RandomHorizontalFlipTransform(),
                augmentation.OneHotTransform(len(clasess_inds)+1),
                augmentation.RandomResizeTransform(),
                augmentation.RandomCropTransform((448, 448)),
                augmentation.RandomNoiseTransform(),
                augmentation.RandomColorJitterTransform(),
                augmentation.RandomBlurTransform(),
                augmentation.RandomJPEGcompression(95),
                augmentation.OutputTransform()]
            )
            self.datasets = [
                CocoDataset('train2017', 'annotations_trainval2017/annotations/instances_train2017.json')
            ]

        if not train:
            self.transforms = augmentation.PairCompose([
                augmentation.PaddTransform(2**depth),
                augmentation.OneHotTransform(len(clasess_inds)+1),
                augmentation.OutputTransform()]
            )
            self.datasets = [
                CocoDataset('val2017', 'annotations_trainval2017/annotations/instances_val2017.json')
            ]

        self.data_ids = [(i, j) for i, dataset in enumerate(self.datasets) for j in range(len(self.datasets[i]))]



    def __len__(self):
        if self.debug==1:
            return 5
        else:
            return len(self.data_ids)

    def __getitem__(self, idx):
        identifier = self.data_ids[idx]
        data = self.datasets[identifier[0]][identifier[1]]
        if self.transforms:
            data = self.transforms(*data)
        return data