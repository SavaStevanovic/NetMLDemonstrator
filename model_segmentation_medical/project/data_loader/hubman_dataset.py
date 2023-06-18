import os
from torch.utils.data import Dataset
import torch
from PIL import Image
import ujson as json
import numpy as np
from libtiff import TIFF
import cv2


class HubmapDataset(Dataset):
    def __init__(self, image_dir, labels_file):
        with open(labels_file, "r") as json_file:
            json_labels = [json.loads(line) for line in json_file]
        json_ids = {x["id"]: x["annotations"] for x in json_labels}
        image_files = os.listdir(image_dir)
        ids = [f.split(".")[0] for f in image_files]
        self._json_labels = [
            (
                os.path.join(image_dir, f"{x}.tif"),
                json_ids[x] if x in json_ids else [],
            )
            for x in ids
            if (x in json_ids)
            and any(ann["type"] == "blood_vessel" for ann in json_ids[x])
        ]

        self.labels = ["background", "blood_vessel"]

    def __len__(self):
        return len(self._json_labels)

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


class HubmapInstanceDataset(HubmapDataset):
    def __getitem__(self, idx):
        image_path, annots = self._json_labels[idx]
        image = TIFF.open(image_path).read_image()
        masks = HubmapInstanceDataset._generate_masks(annots)
        bboxes = HubmapInstanceDataset._generate_bboxes(masks)
        bboxes = np.array(bboxes)
        areas = HubmapInstanceDataset._generate_areas(bboxes)
        target = {
            "boxes": bboxes,
            "labels": np.ones((len(bboxes),)),
            "masks": masks,
            "image_id": np.array([idx]),
            "area": areas,
            "iscrowd": np.zeros((len(bboxes),)),
        }
        return Image.fromarray(image), target

    @staticmethod
    def _generate_bboxes(masks):
        bboxes = []
        for mask in masks:
            contours, _ = cv2.findContours(
                (mask * 255).astype(np.uint8),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )
            assert len(contours) == 1
            x, y, w, h = cv2.boundingRect(contours[0])
            bboxes.append([x, y, x + w, y + h])
        return bboxes

    @staticmethod
    def _generate_masks(annots):
        masks = []
        for annot in annots:
            cords = annot["coordinates"]
            if annot["type"] == "blood_vessel":
                segment = np.array(cords)
                mask = np.zeros((512, 512), dtype=np.float32)
                cv2.fillPoly(mask, segment, 1)
                masks.append(mask)

        return np.array(masks)

    @staticmethod
    def _generate_areas(boxes):
        if len(boxes):
            return (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        return torch.tensor([])
