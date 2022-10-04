from torchvision import datasets
from torch.utils.data import Dataset

from data_loader.class_dataset import ClassDataset

class VOCDataset(ClassDataset):
    def __init__(self, mode, directory):
       self.data = datasets.VOCDetection(directory, image_set = mode, download = False)
       self.labels = [
            'aeroplane',
            'bicycle',
            'bird',
            'boat',
            'bottle',
            'bus',
            'car',
            'cat',
            'chair',
            'cow',
            'diningtable',
            'dog',
            'horse',
            'motorbike',
            'person',
            'pottedplant',
            'sheep',
            'sofa',
            'train',
            'tvmonitor'
        ]
    
    @property
    def classes_map(self):
        return self.labels
     
    def __len__(self):
        return len(self.data)

    def _convert_bounding_box(self, ann):
        bbox = ann["bndbox"]
        bbox = {k:float(v) for k, v in bbox.items()}
        ann = {"bbox": [bbox["xmin"], bbox["ymin"], bbox["xmax"] - bbox["xmin"], bbox["ymax"] - bbox["ymin"]], "category": ann["name"]}
        
        return ann
    
    def __getitem__(self, idx):
        img, anns = self.data[idx]
        anns = anns["annotation"]["object"]
        if isinstance(anns, dict):
            anns = [anns]
        anns = [self._convert_bounding_box(x) for x in anns]
        return img, anns

