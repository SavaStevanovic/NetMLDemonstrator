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
        base_dir = '/Data/segmentation/Coco'
        self.folder = os.path.join(base_dir, folder)
        self.ann_file = os.path.join(base_dir, ann_file)
        self.data = COCO(self.ann_file)
        self.classes_inds = [x for x in self.data.cats]
        self.labels = [self.data.cats[x]['name'] for x in self.data.cats]
        self.ids = self.data.getImgIds()
        valid_ids = []
        for idx in self.ids:
            ann_ids = self.data.getAnnIds(imgIds=idx)
            img_anns   = self.data.loadAnns(ann_ids)
            if len(img_anns)>0:
                valid_ids.append(idx)
        self.ids = valid_ids

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        ann_ids = self.data.getAnnIds(imgIds=self.ids[idx])
        img_anns   = self.data.loadAnns(ann_ids)
        img_data    = self.data.loadImgs(ids = [self.ids[idx]])[0]
        img = Image.open(os.path.join(self.folder, img_data['file_name']))

        mask_anns = [self.data.annToMask(ann) for ann in img_anns]
        for i in range(len(mask_anns)):
            mask_anns[i] *= self.classes_inds.index(img_anns[i]['category_id']) + 1
        label = Image.fromarray(np.amax(mask_anns, axis=0)) 
        # plt.imshow(img); plt.axis('off')
        # self.data.showAnns(img_anns)
        # plt.imshow(label/len(self.labels), alpha=0.5); plt.axis('off')
        # plt.show()
        return img, label
