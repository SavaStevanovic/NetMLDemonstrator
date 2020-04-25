import torch
from torchvision.datasets import CocoDetection
import torchvision.transforms as transforms
from data_loader.augmentation import PairCompose, OutputTransform, TargetTransform, PaddTransform, TargetTransformToBoxes, RandomHorizontalFlipTransform, RandomResizeTransform, RandomCropTransform
import multiprocessing as mu
import os

class CocoDetectionDatasetProvider():
    def __init__(self, annDir='/Data/Coco/', train_transforms=None, val_transforms=None, classes=None, th_count=mu.cpu_count(), ratios=[1.0]):
        if classes is None:
            classes = [(1, 'person'), (2, 'bicycle'), (3, 'car'), (4, 'motorcycle'), (5, 'airplane'), (6, 'bus'), (7, 'train'), (8, 'truck'), (9, 'boat'), (10, 'traffic light'), (11, 'fire hydrant'), (13, 'stop sign'), (14, 'parking meter'), (15, 'bench'), (16, 'bird'), (17, 'cat'), (18, 'dog'), (19, 'horse'), (20, 'sheep'), (21, 'cow'), (22, 'elephant'), (23, 'bear'), (24, 'zebra'), (25, 'giraffe'), (27, 'backpack'), (28, 'umbrella'), (31, 'handbag'), (32, 'tie'), (33, 'suitcase'), (34, 'frisbee'), (35, 'skis'), (36, 'snowboard'), (37, 'sports ball'), (38, 'kite'), (39, 'baseball bat'), (40, 'baseball glove'), (41, 'skateboard'), (42, 'surfboard'), (43, 'tennis racket'), (44, 'bottle'), (46, 'wine glass'), (47, 'cup'), (48, 'fork'), (49, 'knife'), (50, 'spoon'), (51, 'bowl'), (52, 'banana'), (53, 'apple'), (54, 'sandwich'), (55, 'orange'), (56, 'broccoli'), (57, 'carrot'), (58, 'hot dog'), (59, 'pizza'), (60, 'donut'), (61, 'cake'), (62, 'chair'), (63, 'couch'), (64, 'potted plant'), (65, 'bed'), (67, 'dining table'), (70, 'toilet'), (72, 'tv'), (73, 'laptop'), (74, 'mouse'), (75, 'remote'), (76, 'keyboard'), (77, 'cell phone'), (78, 'microwave'), (79, 'oven'), (80, 'toaster'), (81, 'sink'), (82, 'refrigerator'), (84, 'book'), (85, 'clock'), (86, 'vase'), (87, 'scissors'), (88, 'teddy bear'), (89, 'hair drier'), (90, 'toothbrush')]
        self.classes = classes
        self.prior_box_size = 32
        if train_transforms is None:
            train_transforms = PairCompose([
                                            RandomResizeTransform(),
                                            RandomHorizontalFlipTransform(),
                                            RandomCropTransform((224, 224)),
                                            PaddTransform(pad_size=32), 
                                            TargetTransform(prior_box_size=self.prior_box_size, classes=classes, ratios=ratios, stride=32), 
                                            OutputTransform()])
        if val_transforms is None:
            val_transforms = PairCompose([PaddTransform(pad_size=32), 
                                          TargetTransform(prior_box_size=self.prior_box_size, classes=classes, ratios=ratios, stride=32), 
                                          OutputTransform()])

        self.target_to_box_transform = TargetTransformToBoxes(prior_box_size=self.prior_box_size, classes=classes, ratios=ratios, stride=32)

        train_dir           = os.path.join(annDir, 'train2017')  
        train_ann_file      = os.path.join(annDir, 'annotations_trainval2017/annotations/instances_train2017.json')
        validation_dir      = os.path.join(annDir, 'val2017')
        validation_add_file = os.path.join(annDir, 'annotations_trainval2017/annotations/instances_val2017.json')
        self.trainset       = CocoDetection(root = train_dir     , annFile = train_ann_file     , transforms=train_transforms)
        self.validationset  = CocoDetection(root = validation_dir, annFile = validation_add_file, transforms=val_transforms)

        if th_count==1:
            self.trainset.ids      = self.trainset.ids[:100]
            self.validationset.ids = self.validationset.ids[:100]

        self.trainloader      = torch.utils.data.DataLoader(self.trainset,      batch_size=1,   shuffle=True,  num_workers=th_count, pin_memory=True)
        self.validationloader = torch.utils.data.DataLoader(self.validationset, batch_size=1,   shuffle=False, num_workers=th_count, pin_memory=True)

        self.classes = classes
        self.trainloader.cats = classes
        self.validationloader.cats = classes