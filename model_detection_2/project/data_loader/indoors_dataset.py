from torch.utils.data import Dataset
from PIL import Image
import os
import xmltodict


class IndoorDetection(Dataset):
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
        image_folders = [os.path.join(directory, x) for x in os.listdir(directory) if x!="annotation"]
        self._image_paths = sum([os.listdir(x) for x in image_folders], [])

    def __len__(self):
        return len(self._image_paths)

    def __getitem__(self, idx):
        img_pth = self._image_paths[idx]
        img = Image.open(img_pth)
        return img
