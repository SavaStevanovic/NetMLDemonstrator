import json
import os
from PIL import Image
import xmltodict

from data_loader.class_dataset import ClassDataset


class ManualDetection(ClassDataset):
    def __init__(self, directory):
        ann_dir = os.path.join(directory, "Annotations")
        anotations = [os.path.join(ann_dir, x) for x in os.listdir(ann_dir)]
        self._annotations = {}
        for ann_path in anotations:
            with open(ann_path, 'r', encoding='utf-8') as file:
                xml = file.read()
                image_data = xmltodict.parse(xml)
                image_data = image_data["annotation"]["object"]
                if isinstance(image_data, dict):
                    image_data = [image_data]
                self._annotations[ann_path.split("/")[-1].strip(".xml")] = image_data
        image_files = [x for x in os.listdir(os.path.join(directory, "images")) if ".webp" not in x]
        self._image_anns = [(os.path.join(directory, "images", file_name), [self._convert_bounding_box(v) for v in self._annotations[file_name.split(".")[0]]]) for file_name in image_files]

    def _convert_bounding_box(self, ann):
        bbox = ann["bndbox"]
        bbox = {k:float(v) for k, v in bbox.items()}
        ann = {"bbox": [bbox["xmin"], bbox["ymin"], bbox["xmax"] - bbox["xmin"], bbox["ymax"] - bbox["ymin"]], "category": ann["name"]}
        
        return ann
    
    def __len__(self):
        return len(self._image_anns)

    @property
    def classes_map(self):
        anns = sum([x[1] for x in self._image_anns], [])
        return sorted(set([v["category"] for v in anns if v]))

    def __getitem__(self, idx):
        img_pth, anns = self._image_anns[idx]
        img = Image.open(img_pth).convert("RGB")
        return img, anns