import numpy as np
import torch
from data_loader.hubman_dataset import HubmapDataset, HubmapInstanceDataset
from torch.utils.data import Dataset
import torchvision


class TransformedDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self._dataset = dataset
        self._transform = transform

    def __getitem__(self, index):
        x, y = self._dataset[index]
        aug_data = self._transform(
            image=np.array(x),
            masks=y["masks"],
            bboxes=y["boxes"],
            labels=y["labels"],
        )
        y["labels"] = torch.tensor(aug_data["labels"], dtype=torch.int64)
        y["masks"] = torch.tensor(np.array(aug_data["masks"]), dtype=torch.uint8)
        y["iscrowd"] = torch.tensor(y["iscrowd"], dtype=torch.int64)
        if not len(aug_data["bboxes"]):
            aug_data["bboxes"] = np.zeros((0, 4))
        y["boxes"] = torch.tensor(aug_data["bboxes"], dtype=torch.float32)
        y["image_id"] = torch.tensor(y["image_id"])
        y["area"] = torch.tensor(y["area"])
        x = torchvision.transforms.ToTensor()(aug_data.pop("image"))
        return x, y

    def __len__(self):
        return len(self._dataset)


class UnifiedKeypointDataset(Dataset):
    def __init__(self, debug=False):
        self.debug = debug
        self.datasets = [
            HubmapInstanceDataset("/Data/train", "/Data/polygons.jsonl"),
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
