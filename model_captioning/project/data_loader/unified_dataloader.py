import torch
from data_loader.coco_dataset import CocoDataset
import torchvision.transforms as transforms
from data_loader import augmentation
import multiprocessing as mu
from torch.utils.data import Dataset, DataLoader
from data_loader.unified_dataset import UnifiedDataset
import os
from torch.nn.utils.rnn import pad_sequence
from itertools import groupby

class UnifiedDataloader(object):
    def __init__(self, batch_size=1, th_count=mu.cpu_count()):
        self.th_count = th_count
        self.batch_size = batch_size 
        train_dataset = UnifiedDataset(True , debug=self.th_count)
        val_dataset   = UnifiedDataset(False, debug=self.th_count)

        self.trainloader      = torch.utils.data.DataLoader(train_dataset, batch_size=(batch_size-1)+1, shuffle=th_count>1, num_workers=th_count   , collate_fn = self.collate_fn)
        self.validationloader = torch.utils.data.DataLoader(val_dataset  , batch_size=1                            , shuffle=False     , num_workers=th_count//2, collate_fn = self.collate_fn)
        self.vectorizer = train_dataset.vectorizer

    def collate_fn(self, batch):
        batch = sorted(batch, key=lambda x: len(x[1]))
        batch = list(zip(*[(x[0].unsqueeze(0), x[1], len(x[1])) for x in batch]))
        batch_per_length = [(key, len(list(group))) for key, group in groupby(batch[2])]
        return torch.cat(batch[0]), pad_sequence(batch[1], True), batch_per_length