from torch.utils.data import Dataset, DataLoader
import os
import json
from pycocotools.coco import COCO
import matplotlib.pyplot as plt
from skimage import io
from PIL import Image
from operator import itemgetter
import numpy as np

classes = [(1, 'person'), (2, 'bicycle'), (3, 'car'), (4, 'motorcycle'), (5, 'airplane'), (6, 'bus'), (7, 'train'), (8, 'truck'), (9, 'boat'), (10, 'traffic light'), (11, 'fire hydrant'), (13, 'stop sign'), (14, 'parking meter'), (15, 'bench'), (16, 'bird'), (17, 'cat'), (18, 'dog'), (19, 'horse'), (20, 'sheep'), (21, 'cow'), (22, 'elephant'), (23, 'bear'), (24, 'zebra'), (25, 'giraffe'), (27, 'backpack'), (28, 'umbrella'), (31, 'handbag'), (32, 'tie'), (33, 'suitcase'), (34, 'frisbee'), (35, 'skis'), (36, 'snowboard'), (37, 'sports ball'), (38, 'kite'), (39, 'baseball bat'), (40, 'baseball glove'), (41, 'skateboard'), (42, 'surfboard'), (43, 'tennis racket'), (44, 'bottle'), (46, 'wine glass'), (47, 'cup'), (48, 'fork'), (49, 'knife'), (50, 'spoon'), (51, 'bowl'), (52, 'banana'), (53, 'apple'), (54, 'sandwich'), (55, 'orange'), (56, 'broccoli'), (57, 'carrot'), (58, 'hot dog'), (59, 'pizza'), (60, 'donut'), (61, 'cake'), (62, 'chair'), (63, 'couch'), (64, 'potted plant'), (65, 'bed'), (67, 'dining table'), (70, 'toilet'), (72, 'tv'), (73, 'laptop'), (74, 'mouse'), (75, 'remote'), (76, 'keyboard'), (77, 'cell phone'), (78, 'microwave'), (79, 'oven'), (80, 'toaster'), (81, 'sink'), (82, 'refrigerator'), (84, 'book'), (85, 'clock'), (86, 'vase'), (87, 'scissors'), (88, 'teddy bear'), (89, 'hair drier'), (90, 'toothbrush')]
clasess_inds = [x[0] for x in classes]
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
            mask_anns[i]*=clasess_inds.index(img_anns[i]['category_id'])+1
        label = Image.fromarray(np.amax(mask_anns, axis=0)) 
        # plt.imshow(img); plt.axis('off')
        # self.data.showAnns(img_anns)
        # plt.imshow(label/len(clasess_inds), alpha=0.5); plt.axis('off')
        # plt.show()
        return img, label
