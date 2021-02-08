from torch.utils.data import Dataset, DataLoader
import os
import json
from pycocotools.coco import COCO
import matplotlib.pyplot as plt
from skimage import io
from PIL import Image
from operator import itemgetter
import numpy as np

class CocoDataset(Dataset):
    def __init__(self, folder, ann_file):
        base_dir = '/Data/captioning/Coco'
        self.folder = os.path.join(base_dir, folder)
        self.ann_file = os.path.join(base_dir, ann_file)
        self.data = COCO(self.ann_file)
        self.ids = list(self.data.anns.keys())

    def get_vocab_list(self):

        return [self.data.anns[idx]['caption'] for idx in self.ids]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        item = self.data.anns[self.ids[idx]]
        img_id = item['image_id']
        img_path = self.data.loadImgs(img_id)[0]['file_name']
        image = Image.open(os.path.join(self.folder, img_path))

        # img_anns = self.data.loadAnns(ann_ids)
        label = item['caption']

        return image, label
