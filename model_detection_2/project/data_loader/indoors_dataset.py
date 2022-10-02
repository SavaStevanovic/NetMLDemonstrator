from torch.utils.data import Dataset
from PIL import Image
import os
import xmltodict
from data_loader.class_dataset import ClassDataset


class IndoorDetection(ClassDataset):
    def __init__(self, directory):
        ann_dir = os.path.join(directory, "annotation")
        anotations = [os.path.join(ann_dir, x) for x in os.listdir(ann_dir)]
        self._annotations = {}
        for ann_path in anotations:
            with open(ann_path, 'r', encoding='utf-8') as file:
                xml = file.read()
                image_data = xmltodict.parse(xml)
                image_data = {x["@file"]: x.get("box", []) for x in image_data["dataset"]["images"]["image"]}
                self._annotations = {**self._annotations, **image_data}
        self._annotations = {k: v if isinstance(v, list) else [v] for k, v in self._annotations.items()}
        image_folders = [os.path.join(directory, x) for x in os.listdir(directory) if x!="annotation"]
        self._image_anns = [(os.path.join(x, file_name), [self._convert_bounding_box(v) for v in (self._annotations[file_name])]) for x in image_folders for file_name in os.listdir(x)]

    def _convert_bounding_box(self, ann):
        return {"bbox": [ann["@top"], ann["@left"], ann["@width"], ann["@height"]], "category": ann["label"]}
    
    def __len__(self):
        return len(self._image_anns)

    @property
    def classes_map(self):
        anns = sum([x[1] for x in self._image_anns], [])
        return sorted(set([v["category"] for v in anns]))

    def __getitem__(self, idx):
        img_pth, anns = self._image_anns[idx]
        img = Image.open(img_pth)
        return img, anns