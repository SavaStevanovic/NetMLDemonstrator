import torch
from data_loader.coco_dataset import CocoDataset
import torchvision.transforms as transforms
from data_loader import augmentation
from visualization import output_transform
import multiprocessing as mu
from torch.utils.data import Dataset, DataLoader
from data_loader.unified_dataset import UnifiedKeypointDataset
import os
from torch.nn.utils.rnn import pad_sequence

class UnifiedKeypointDataloader(object):
    def __init__(self, batch_size=1, th_count=mu.cpu_count()):
        self.th_count = th_count
        self.batch_size = batch_size 
        train_dataset = UnifiedKeypointDataset(True , debug=self.th_count)
        val_dataset   = UnifiedKeypointDataset(False, debug=self.th_count)

        self.trainloader      = torch.utils.data.DataLoader(train_dataset, batch_size=(th_count>1)*(batch_size-1)+1, shuffle=th_count>1, num_workers=th_count   , collate_fn = self.collate_fn)
        self.validationloader = torch.utils.data.DataLoader(val_dataset  , batch_size=1                            , shuffle=False     , num_workers=th_count//2, collate_fn = self.collate_fn)
        self.vectorizer = train_dataset.vectorizer

    def collate_fn(self, batch):
        batch = list(zip(*[(x[0].unsqueeze(0), x[1]) for x in batch if x[0]!= None]))
        if not batch:
            return None, None
        return torch.cat(batch[0]), pad_sequence(batch[1], True)