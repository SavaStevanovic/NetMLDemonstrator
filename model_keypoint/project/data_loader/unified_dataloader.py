import torch
from data_loader.coco_dataset import CocoDataset
import torchvision.transforms as transforms
from data_loader import augmentation
from visualization import output_transform
import multiprocessing as mu
from torch.utils.data import Dataset, DataLoader
from data_loader.unified_dataset import UnifiedKeypointDataset
import os

class UnifiedKeypointDataloader(object):
    def __init__(self, batch_size=1, th_count=mu.cpu_count()):
        self.th_count = th_count
        self.batch_size = batch_size 
        train_dataset = UnifiedKeypointDataset(train=True , debug=self.th_count)
        val_dataset   = UnifiedKeypointDataset(train=False, debug=self.th_count)

        self.trainloader      = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True , num_workers=th_count)
        self.validationloader = torch.utils.data.DataLoader(val_dataset  , batch_size=1         , shuffle=False, num_workers=th_count)
        self.postprocessing   = train_dataset.postprocessing
        self.trainloader.skeleton         = train_dataset.skeleton 
        self.trainloader.parts            = train_dataset.parts
        self.validationloader.skeleton    = val_dataset.skeleton 
        self.validationloader.parts       = val_dataset.parts