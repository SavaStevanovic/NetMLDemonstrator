import torch
from torchvision.datasets import CocoDetection
import torchvision.transforms as transforms
from data_loader import augmentation
import multiprocessing as mu
import os

class CocoDetectionDatasetProvider():
    def __init__(self, depth, feature_count, classes=None, annDir='/Data/Coco/', batch_size=1, train_transforms=None, val_transforms=None, th_count=mu.cpu_count(), ratios=[1.0]):
        self.classes = classes
        self.prior_box_sizes = [32*2**i for i in range(depth-feature_count, depth)]
        self.strides = [2**(i+1) for i in range(depth-feature_count, depth)]
        if train_transforms is None:
            train_transforms = augmentation.PairCompose([
                                            augmentation.RandomResizeTransform(),
                                            augmentation.RandomHorizontalFlipTransform(),
                                            augmentation.RandomCropTransform((416, 416)),
                                            augmentation.RandomNoiseTransform(),
                                            augmentation.RandomColorJitterTransform(),
                                            augmentation.RandomBlurTransform(),
                                            augmentation.TargetTransform(prior_box_sizes=self.prior_box_sizes, classes=classes, ratios=ratios, strides=self.strides), 
                                            augmentation.OutputTransform()])
        if val_transforms is None:
            val_transforms = augmentation.PairCompose([
                                          augmentation.PaddTransform(pad_size=32), 
                                          augmentation.TargetTransform(prior_box_sizes=self.prior_box_sizes, classes=classes, ratios=ratios, strides=self.strides), 
                                          augmentation.OutputTransform()])

        self.target_to_box_transform = augmentation.TargetTransformToBoxes(prior_box_sizes=self.prior_box_sizes, classes=classes, ratios=ratios, strides=self.strides)

        train_dir           = os.path.join(annDir, 'train2017')  
        train_ann_file      = os.path.join(annDir, 'annotations_trainval2017/annotations/instances_train2017.json')
        validation_dir      = os.path.join(annDir, 'val2017')
        validation_add_file = os.path.join(annDir, 'annotations_trainval2017/annotations/instances_val2017.json')
        self.trainset       = CocoDetection(root = train_dir     , annFile = train_ann_file     , transforms=train_transforms)
        self.validationset  = CocoDetection(root = validation_dir, annFile = validation_add_file, transforms=val_transforms)

        if th_count==1:
            self.trainset.ids      = self.trainset.ids[:100]
            self.validationset.ids = self.validationset.ids[:100]

        self.trainloader      = torch.utils.data.DataLoader(self.trainset,      batch_size=batch_size,   shuffle=True , num_workers=th_count, pin_memory=True)
        self.validationloader = torch.utils.data.DataLoader(self.validationset, batch_size=1,            shuffle=False, num_workers=th_count, pin_memory=True)

        self.classes = classes
        self.trainloader.cats = classes
        self.validationloader.cats = classes