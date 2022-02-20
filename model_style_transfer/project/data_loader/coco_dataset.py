from torch.utils.data import Dataset
import os
from pycocotools.coco import COCO
from PIL import Image


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
            img_anns = self.data.loadAnns(ann_ids)
            if len(img_anns) > 0:
                valid_ids.append(idx)
        self.ids = valid_ids

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_data = self.data.loadImgs(ids=[self.ids[idx]])[0]
        img = Image.open(os.path.join(self.folder, img_data['file_name']))

        return img
