from sklearn.model_selection import KFold
import torch
import multiprocessing as mu
from data_loader.unified_dataset import TransformedDataset, UnifiedKeypointDataset
from torch.utils.data import DataLoader, Subset
from data_loader import augmentation


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
        self._k_folder = KFoldCrossValidator(dataset, 6)
        self._labels = dataset.labels
        self._train_aug = augmentation.PairCompose(
            [
                augmentation.RandomHorizontalFlipTransform(),
                augmentation.RandomWidthFlipTransform(),
                augmentation.RandomCropTransform((448, 448)),
                augmentation.RandomNoiseTransform(),
                augmentation.RandomColorJitterTransform(),
                augmentation.OneHotTransform(len(self._labels)),
                augmentation.OutputTransform(),
            ]
        )
        self._val_aug = augmentation.PairCompose(
            [
                augmentation.PaddTransform(2**depth),
                augmentation.OneHotTransform(len(self._labels)),
                augmentation.OutputTransform(),
            ]
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
            )
            validationloader = DataLoader(
                val_data,
                batch_size=1,
                shuffle=False,
                num_workers=self.th_count,
            )
            yield trainloader, validationloader

    @property
    def labels(self):
        return self._labels
