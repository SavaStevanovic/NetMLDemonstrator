from data_loader.class_dataset import ClassDataset

class SubsetDataset(ClassDataset):
    def __init__(self, dataset, maper: dict):
        self._dataset = dataset
        self._maper = maper
        self._valid_ids = [i for i in range(len(self._dataset)) if self._filter_annotations(self._dataset[i][1])]
    
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

