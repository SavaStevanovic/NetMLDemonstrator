import json
import os
from data_loader.class_dataset import ClassDataset
from tqdm import tqdm


class SubsetDataset(ClassDataset):
    def __init__(self, dataset, maper: dict, save_path: str):
        self._dataset = dataset
        self._maper = maper
        if not os.path.exists(save_path):
            self._valid_ids = [i for i in tqdm(range(len(self._dataset))) if self._filter_annotations(self._dataset[i][1])]
            with open(save_path, "w") as out_file:
                json.dump(self._valid_ids, out_file) 
        else:
            with open(save_path, "r") as out_file:
                self._valid_ids = json.load(out_file) 
    
    def _filter_annotations(self, annotations: list):
        anns = [ann for ann in annotations if self._maper.get(ann["category"])]
        for ann in anns:
            ann["category"] = self._maper.get(ann["category"])
        return anns
    
    @property
    def classes_map(self):
        return sorted(self._maper.values())
    
    def __len__(self):
        return len(self._valid_ids)
    
    def __getitem__(self, idx):
        img, anns = self._dataset[self._valid_ids[idx]]
        return img, self._filter_annotations(anns)

