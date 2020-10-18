import torch
from data_loader.coco_dataset import CocoDataset
import torchvision.transforms as transforms
from data_loader import augmentation
from visualization import output_transform
import multiprocessing as mu
from torch.utils.data import Dataset, DataLoader
from visualization.output_transform import PartAffinityFieldTransform
import os

skeleton = [
    ['left_ankle', 'left_knee'], 
    ['left_knee', 'left_hip'], 
    ['right_ankle', 'right_knee'], 
    ['right_knee', 'right_hip'], 
    ['left_hip', 'right_hip'], 
    ['left_shoulder', 'left_hip'], 
    ['right_shoulder', 'right_hip'], 
    ['left_shoulder', 'right_shoulder'], 
    ['left_shoulder', 'left_elbow'], 
    ['right_shoulder', 'right_elbow'], 
    ['left_elbow', 'left_wrist'], 
    ['right_elbow', 'right_wrist'], 
    ['left_eye', 'right_eye'], 
    ['nose', 'left_eye'], 
    ['nose', 'right_eye'], 
    ['left_eye', 'left_ear'], 
    ['right_eye', 'right_ear'], 
    ['left_ear', 'left_shoulder'], 
    ['right_ear', 'right_shoulder']
]

class UnifiedKeypointDataset(Dataset):
    def __init__(self, net, train=True, debug=False):
        self.debug = debug
        self.train = train
        self.skeleton = skeleton
        self.parts = list(set(sum(self.skeleton, [])))
        if train:
            self.transforms = augmentation.PairCompose([
                augmentation.RandomCropTransform((416, 416)),
                augmentation.PartAffinityFieldTransform(skeleton, 10, 6, self.parts),
                # augmentation.RandomResizeTransform(),
                # augmentation.RandomHorizontalFlipTransform(),
                # augmentation.RandomNoiseTransform(),
                # augmentation.RandomColorJitterTransform(),
                # augmentation.RandomBlurTransform(),
                # augmentation.TargetTransform(prior_box_sizes=net.prior_box_sizes, classes=net.classes, ratios=net.ratios, strides=net.strides), 
                augmentation.OutputTransform()]
            )
            self.datasets = [
                CocoDataset('train2017', 'annotations_trainval2017/annotations/person_keypoints_train2017.json')
            ]

        if not train:
            self.transforms = augmentation.PairCompose([
                augmentation.PartAffinityFieldTransform(skeleton, 10, 6, self.parts),
                #   augmentation.PaddTransform(pad_size=2**net.depth), 
                #   augmentation.TargetTransform(prior_box_sizes=net.prior_box_sizes, classes=net.classes, ratios=net.ratios, strides=net.strides), 
                augmentation.OutputTransform()]
            )
            self.datasets = [
                CocoDataset('val2017', 'annotations_trainval2017/annotations/person_keypoints_val2017.json')
            ]

        self.data_ids = [(i, j) for i, dataset in enumerate(self.datasets) for j in range(len(self.datasets[i]))]
        self.postprocessing = PartAffinityFieldTransform(skeleton, 6)



    def __len__(self):
        if self.debug==1:
            return 100
        else:
            return len(self.data_ids)

    def __getitem__(self, idx):
        identifier = self.data_ids[idx]
        data = self.datasets[identifier[0]][identifier[1]]
        if self.transforms:
            data = self.transforms(*data)
        return data