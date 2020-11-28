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
        base_dir = '/Data/keypoint/Coco'
        self.folder = os.path.join(base_dir, folder)
        self.ann_file = os.path.join(base_dir, ann_file)
        # with open(self.ann_file) as anns:
        #     self.data = json.load(anns)
        self.data = COCO(self.ann_file)
        # self.data.getCatIds()
        # print('sadsd')
        self.ids = self.data.getImgIds()
        self.keypoint_names = self.data.cats[1]['keypoints']
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

        person_keypoints = []
        mask_ids = []
        for i, keypoint_ann in enumerate(img_anns):
            ann = keypoint_ann['keypoints']
            person = {self.keypoint_names[k]:(ann[k*3], ann[1+k*3]) for k in range(len(self.keypoint_names)) if ann[2+k*3]}
            person_visible = len([0 for k in range(len(self.keypoint_names)) if ann[2+k*3] == 2])

            if not bool(person) or person_visible==0:
                mask_ids.append(i)
            elif bool(person):
                person_keypoints.append(person)
        if len(mask_ids)==0:
            img_anns = []
        elif len(mask_ids)<=1:
            img_anns = [itemgetter(*mask_ids)(img_anns)]
        else:
            img_anns = list(itemgetter(*mask_ids)(img_anns))
        mask_anns = [self.data.annToMask(ann) for ann in img_anns]
        mask_anns.append(np.zeros(img.size[:2][::-1], dtype=np.uint8))
        mask = np.bitwise_or.reduce(mask_anns)
        # plt.imshow(img); plt.axis('off')
        # self.data.showAnns(img_anns)
        # plt.imshow(Image.fromarray(mask, 'L'), alpha=0.5)
        # plt.show()
        return img, person_keypoints, Image.fromarray(mask*255, 'L')
