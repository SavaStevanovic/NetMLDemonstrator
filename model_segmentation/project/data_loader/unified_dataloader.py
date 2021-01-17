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
    def __init__(self, depth, batch_size=1, th_count=mu.cpu_count()):
        self.th_count = th_count
        self.batch_size = batch_size 
        train_dataset = UnifiedKeypointDataset(True , depth, debug=self.th_count)
        val_dataset   = UnifiedKeypointDataset(False, depth, debug=self.th_count)

        self.trainloader      = torch.utils.data.DataLoader(train_dataset, batch_size=(th_count>1)*(batch_size-1)+2, shuffle=th_count>1, num_workers=th_count)
        self.validationloader = torch.utils.data.DataLoader(val_dataset  , batch_size=1         , shuffle=False     , num_workers=th_count)
        self.labels = train_dataset.labels
        self.trainloader.selector = train_dataset.selector
        self.validationloader.selector = val_dataset.selector