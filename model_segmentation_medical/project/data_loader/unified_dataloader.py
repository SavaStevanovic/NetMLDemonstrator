from sklearn.model_selection import KFold
import torch
import multiprocessing as mu
from data_loader.unified_dataset import TransformedDataset, UnifiedKeypointDataset
from torch.utils.data import DataLoader, Subset
from data_loader import augmentation
import data_loader.detection_transforms as T
import albumentations as A


class KFoldCrossValidator:
    def __init__(self, dataset, num_folds, shuffle=True, random_state=None):
        self.dataset = dataset
        self.num_folds = num_folds
        self.shuffle = shuffle
        self.random_state = random_state

    def get_data_loaders(self):
        dataset_size = len(self.dataset)
        indices = list(range(dataset_size))
        kfold = KFold(
            n_splits=self.num_folds,
            shuffle=self.shuffle,
            random_state=self.random_state,
        )

        data_loaders = []
        for train_indices, val_indices in kfold.split(indices):
            train_dataset = Subset(self.dataset, train_indices)
            val_dataset = Subset(self.dataset, val_indices)
            data_loaders.append((train_dataset, val_dataset))

        return data_loaders


class UnifiedKeypointDataloader(object):
    def __init__(self, depth, batch_size=1, th_count=mu.cpu_count()):
        self.th_count = th_count
        self.batch_size = batch_size
        dataset = UnifiedKeypointDataset(debug=self.th_count)
        self._k_folder = KFoldCrossValidator(dataset, 6, random_state=42)
        self._labels = dataset.labels
        self._train_aug = A.Compose(
            [
                # A.RandomCrop(width=496, height=496),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                # A.ShiftScaleRotate(
                #     scale_limit=0.0,
                #     rotate_limit=0,
                # )
                # A.RandomBrightnessContrast(p=0.2),
            ],
            bbox_params=A.BboxParams(format="pascal_voc", label_fields=["labels"]),
        )
        self._val_aug = A.Compose(
            [],
            bbox_params=A.BboxParams(format="pascal_voc", label_fields=["labels"]),
        )

    def __iter__(self):
        for train_data, val_data in self._k_folder.get_data_loaders():
            train_data = TransformedDataset(
                train_data,
                self._train_aug,
            )
            val_data = TransformedDataset(
                val_data,
                self._val_aug,
            )
            trainloader = DataLoader(
                train_data,
                batch_size=1 if self.th_count == 1 else self.batch_size,
                shuffle=True,
                num_workers=self.th_count,
                collate_fn=UnifiedKeypointDataloader.collate_fn,
            )
            validationloader = DataLoader(
                val_data,
                batch_size=1,
                shuffle=False,
                num_workers=self.th_count,
                collate_fn=UnifiedKeypointDataloader.collate_fn,
            )
            yield trainloader, validationloader

    @staticmethod
    def collate_fn(batch):
        return tuple(zip(*batch))

    @property
    def labels(self):
        return self._labels
