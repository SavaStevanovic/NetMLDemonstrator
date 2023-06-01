import os
from torch.utils.data import Dataset
from PIL import Image
import ujson as json
import numpy as np
from libtiff import TIFF
import cv2


class HubmapDataset(Dataset):
    def __init__(self, image_dir, labels_file):
        with open(labels_file, "r") as json_file:
            json_labels = [json.loads(line) for line in json_file]
        image_files = os.listdir(image_dir)
        ids = [f.split(".")[0] for f in image_files]
        self._json_labels = [
            (os.path.join(image_dir, f"{x['id']}.tif"), x["annotations"])
            for x in json_labels
            if x["id"] in ids
        ]

        self.labels = ["blood_vessel"]

    def __len__(self):
        return len(self.json_labels)

    def __getitem__(self, idx):
        image_path, annots = self._json_labels[idx]
        image = TIFF.open(image_path).read_image()
        mask = np.zeros((512, 512), dtype=np.float32)

        for annot in annots:
            cords = annot["coordinates"]
            if annot["type"] == "blood_vessel":
                segment = np.array(cords)
                cv2.fillPoly(mask, segment, 1)

        return Image.fromarray(image), Image.fromarray(mask)
