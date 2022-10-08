import os
from torch.utils.data import Dataset
from torchvision.datasets import CocoDetection

from data_loader.class_dataset import ClassDataset

class CocoDataset(ClassDataset):
    def __init__(self, directory):
        self._dataset = CocoDetection(
            root=os.path.join(directory, "images/train2017"), 
            annFile = os.path.join(directory, "annotations/instances_train2017.json")
        )
        self._label_map = {cat["id"]: cat["name"]for cat in self._dataset.coco.cats.values()}
     
    def __len__(self):
        return len(self._dataset)

    @property
    def classes_map(self):
        return sorted(self._label_map.values())

    def _convert_bounding_box(self, ann):
        bbox = ann["bbox"]
        ann = {"bbox": bbox, "category": self._label_map[ann["category_id"]]}
        
        return ann
    
    def __getitem__(self, idx):
        img, anns = self._dataset[idx]
        # anns = anns["annotation"]["object"]
        if isinstance(anns, dict):
            anns = [anns]
        anns = [self._convert_bounding_box(x) for x in anns]
        return img, anns

