from data_loader.hubman_dataset import HubmapDataset
from torch.utils.data import Dataset


class TransformedDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self._dataset = dataset
        self._transform = transform

    def __getitem__(self, index):
        x, y = self._dataset[index]
        if self._transform:
            x, y = self._transform(x, y)
        return x, y

    def __len__(self):
        return len(self._dataset)


class UnifiedKeypointDataset(Dataset):
    def __init__(self, debug=False):
        self.debug = debug
        self.datasets = [
            HubmapDataset("/Data/train", "/Data/polygons.jsonl"),
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
