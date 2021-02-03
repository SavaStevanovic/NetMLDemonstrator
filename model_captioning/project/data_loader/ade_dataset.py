from torch.utils.data import Dataset
from PIL import Image 
import glob 
import os
import numpy as np

class ADEChallengeData2016(Dataset):
    def __init__(self, mode, folder_path):
        super(ADEChallengeData2016, self).__init__()
        img_files = glob.glob(os.path.join('/Data/segmentation', folder_path, 'annotations', mode, '*.png'))
        self.data = [(x.replace('.png', '.jpg').replace('annotations', 'images'), x) for x in img_files]
        self.labels = [
            'wall',
            'building',
            'sky',
            'floor',
            'tree',
            'ceiling',
            'road',
            'bed',
            'windowpane',
            'grass',
            'cabinet',
            'sidewalk',
            'person',
            'earth',
            'door',
            'table',
            'mountain',
            'plant',
            'curtain',
            'chair',
            'car',
            'water',
            'painting',
            'sofa',
            'shelf',
            'house',
            'sea',
            'mirror',
            'rug',
            'field',
            'armchair',
            'seat',
            'fence',
            'desk',
            'rock',
            'wardrobe',
            'lamp',
            'bathtub',
            'railing',
            'cushion',
            'base',
            'box',
            'column',
            'signboard',
            'chest',
            'counter',
            'sand',
            'sink',
            'skyscraper',
            'fireplace',
            'refrigerator',
            'grandstand',
            'path',
            'stairs',
            'runway',
            'case',
            'pool',
            'pillow',
            'screen',
            'stairway',
            'river',
            'bridge',
            'bookcase',
            'blind',
            'coffee',
            'toilet',
            'flower',
            'book',
            'hill',
            'bench',
            'countertop',
            'stove',
            'palm',
            'kitchen',
            'computer',
            'swivel',
            'boat',
            'bar',
            'arcade',
            'hovel',
            'bus',
            'towel',
            'light',
            'truck',
            'tower',
            'chandelier',
            'awning',
            'streetlight',
            'booth',
            'television',
            'airplane',
            'dirt',
            'apparel',
            'pole',
            'land',
            'bannister',
            'escalator',
            'ottoman',
            'bottle',
            'buffet',
            'poster',
            'stage',
            'van',
            'ship',
            'fountain',
            'conveyer',
            'canopy',
            'washer',
            'plaything',
            'swimming',
            'stool',
            'barrel',
            'basket',
            'waterfall',
            'tent',
            'bag',
            'minibike',
            'cradle',
            'oven',
            'ball',
            'food',
            'step',
            'tank',
            'trade',
            'microwave',
            'pot',
            'animal',
            'bicycle',
            'lake',
            'dishwasher',
            'screen',
            'blanket',
            'sculpture',
            'hood',
            'sconce',
            'vase',
            'traffic',
            'tray',
            'ashcan',
            'fan',
            'pier',
            'crt',
            'plate',
            'monitor',
            'bulletin',
            'shower',
            'radiator',
            'glass',
            'clock',
            'flag'
        ]
       
    def __len__(self):
        return len(self.data)


    def __getitem__(self, index):
            img_path, segm_path = self.data[index]
            data =  Image.open(img_path , mode='r')
            segm = Image.open(segm_path)
            return data, segm