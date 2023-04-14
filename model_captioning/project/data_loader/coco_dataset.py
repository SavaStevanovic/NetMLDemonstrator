from torch.utils.data import Dataset
import os
from pycocotools.coco import COCO
from PIL import Image


class CocoDataset(Dataset):
    def __init__(self, folder, ann_file):
        base_dir = '/Data1/Data/captioning/Coco/'
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

        label = item['caption']

        ann_ids = self.data.getAnnIds(imgIds=img_id)
        img_anns = self.data.loadAnns(ann_ids)
        labels = [x['caption'] for x in img_anns]
        labels = labels[:5]
        return image, label, labels
