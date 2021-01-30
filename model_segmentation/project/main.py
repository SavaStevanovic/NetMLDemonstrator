import torch
from model import blocks
from model import networks
from data_loader.unified_dataloader import UnifiedKeypointDataloader
from model_fitting.train import fit
import os

th_count = 24
block_counts = [3, 4, 6]
depth = len(block_counts)+2

dataloader = UnifiedKeypointDataloader(batch_size = 16, depth=depth, th_count=th_count)

net_backbone = networks.ResNetBackbone(block = blocks.BasicBlock, block_counts = block_counts, inplanes=64)
net = networks.DeepLabV3Plus(net_backbone, dataloader.labels)

fit(net, 
    dataloader.trainloader, 
    dataloader.validationloader, 
    epochs = 1000, 
    lower_learning_period = 3
)       