import torch
from data_loader.coco_dataset import CocoDataset
from data_loader.cityscapes_dataset import CityscapesDataset
from data_loader.voc_dataset import VOCDataset
from data_loader.ade_dataset import ADEChallengeData2016
from data_loader import augmentation
from torch.utils.data import Dataset


class UnifiedKeypointDataset(Dataset):
    def __init__(self, train, depth, debug=False):
        self.debug = debug
        self.train = train
        train_datasets = [
            VOCDataset('train', 'Voc'),
            ADEChallengeData2016('train', 'ADEChallengeData2016'),
            CityscapesDataset('train', 'fine', 'Cityscapes'),
            CocoDataset(
                'train2017', 'annotations_trainval2017/annotations/instances_train2017.json'),
        ]

        if train:
            self.datasets = train_datasets

        if not train:
            self.datasets = [
                VOCDataset('val', 'Voc'),
                ADEChallengeData2016('val', 'ADEChallengeData2016'),
                CityscapesDataset('val', 'fine', 'Cityscapes'),
                CocoDataset(
                    'val2017', 'annotations_trainval2017/annotations/instances_val2017.json'),
            ]

        if train:
            self.transforms = augmentation.PairCompose([
                augmentation.RandomHorizontalFlipTransform(),
                augmentation.RandomResizeTransform(),
                augmentation.RandomCropTransform((256, 256)),
                augmentation.RandomNoiseTransform(),
                augmentation.RandomColorJitterTransform(),
                augmentation.RandomBlurTransform(),
                augmentation.JPEGcompression(95),
                augmentation.OutputTransform()]
            )

        if not train:
            self.transforms = augmentation.PairCompose([
                augmentation.CenterCropTransform((256, 256)),
                augmentation.JPEGcompression(95),
                augmentation.OutputTransform()]
            )

        if self.debug == 1:
            self.data_ids = [(i, j) for i, dataset in enumerate(
                self.datasets) for j in range(50)]
        else:
            self.data_ids = [(i, j) for i, dataset in enumerate(
                self.datasets) for j in range(len(self.datasets[i]))]

    def __len__(self):
        return len(self.data_ids)

    def __getitem__(self, idx):
        identifier = self.data_ids[idx]
        data = self.datasets[identifier[0]][identifier[1]]
        if self.transforms:
            data = self.transforms(data, identifier[0])
        return data
