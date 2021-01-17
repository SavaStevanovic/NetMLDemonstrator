from torchvision import datasets
from torch.utils.data import Dataset, DataLoader
from PIL import Image 
import os
import numpy as np

class CityscapesDataset(Dataset):
    def __init__(self, split, mode, directory):
        self.data = datasets.Cityscapes(os.path.join('/Data/segmentation', directory, 'data'), split = split, mode = mode, target_type = 'semantic')
        self.labels = [
            'road',
            'sidewalk',
            'building',
            'wall',
            'fence',
            'pole',
            'traffic light',
            'traffic sign',
            'vegetation',
            'terrain',
            'sky',
            'person',
            'rider',
            'car',
            'truck',
            'bus',
            'train',
            'motorcycle',
            'bicycle',
        ]
        self.translater = {
            0:0,
            1:0,
            2:0,
            3:0,
            4:0,
            5:0,
            6:0,
            7:1,
            8:2,
            9:0,
            10:0,
            11:3,
            12:4,
            13:5,
            14:0,
            15:0,
            16:0,
            17:6,
            18:0,
            19:7,
            20:8,
            21:9,
            22:10,
            23:11,
            24:12,
            25:13,
            26:14,
            27:15,
            28:16,
            29:0,
            30:0,
            31:17,
            32:18,
            33:19,
            -1:0,
        }

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, smnt = self.data[idx]
        smnt = np.array(smnt)
        smnt = np.vectorize(self.translater.__getitem__)(smnt)
        smnt = Image.fromarray(smnt.astype(np.uint8))
        return img, smnt

