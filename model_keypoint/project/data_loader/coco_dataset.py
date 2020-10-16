from torch.utils.data import Dataset, DataLoader
import os
import json
from pycocotools.coco import COCO
import matplotlib.pyplot as plt
from skimage import io

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

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        annIds = self.data.getAnnIds(imgIds=self.ids[idx])
        anns   = self.data.loadAnns(annIds)
        img_data    = self.data.loadImgs(ids = [self.ids[idx]])[0]
        img = io.imread(os.path.join(self.folder, img_data['file_name']))

        person_keypoints = []
        for keypoint_ann in anns:
            ann = keypoint_ann['keypoints']
            person = {self.keypoint_names[k]:(ann[k*3], ann[1+k*3]) for k in range(len(self.keypoint_names)) if ann[2+k*3]}
            if bool(person):
                person_keypoints.append(person)
        # plt.imshow(img); plt.axis('off')
        # self.data.showAnns(anns)
        # plt.show()

        return img, person_keypoints
