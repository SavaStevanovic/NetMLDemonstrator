from data_loader import augmentation
from torch.utils.data import Dataset

from data_loader.indoors_dataset import IndoorDetection
from data_loader.class_dataset import ClassDataset


class UnifiedDataset(ClassDataset):
    def __init__(self, train, depth, debug=False):
        self.debug = debug
        self.train = train
        train_datasets = [
            IndoorDetection(
                "/Data/detection/IndoorObjectDetectionDataset/train"),
        ]

        if train:
            self.datasets = train_datasets

        if not train:
            self.datasets = [
                # VOCDataset('val', 'Voc'),
                # ADEChallengeData2016('val', 'ADEChallengeData2016'),
                # CityscapesDataset('val', 'fine', 'Cityscapes'),
                IndoorDetection(
                    "/Data/detection/IndoorObjectDetectionDataset/validation"),
            ]

        if train:
            self.transforms = augmentation.PairCompose([
                augmentation.RandomResizeTransform(),
                augmentation.RandomHorizontalFlipTransform(),
                augmentation.RandomCropTransform((416, 416)),
                augmentation.RandomNoiseTransform(),
                augmentation.RandomColorJitterTransform(),
                augmentation.RandomBlurTransform(),
                augmentation.RandomJPEGcompression(95),
                augmentation.OutputTransform()]
            )
        if not train:
            self.transforms = augmentation.PairCompose([
                augmentation.PaddTransform(pad_size=2**depth),
                augmentation.OutputTransform()]
            )

        if self.debug == 1:
            self.data_ids = [(i, j) for i, dataset in enumerate(
                self.datasets) for j in range(50)]
        else:
            self.data_ids = [(i, j) for i, dataset in enumerate(
                self.datasets) for j in range(len(self.datasets[i]))]

    @property
    def classes_map(self):
        return sorted(set(sum([x.classes_map for x in self.datasets], [])))
    
    def __len__(self):
        return len(self.data_ids)

    def __getitem__(self, idx):
        identifier = self.data_ids[idx]
        data = self.datasets[identifier[0]][identifier[1]]
        if self.transforms:
            data = self.transforms(data, identifier[0])
        return data
