import torch
from model import blocks
from model import networks
from data_loader.unified_dataloader import UnifiedKeypointDataloader
from model_fitting.train import fit
import os

th_count = 12
depth = 4

dataloader = UnifiedKeypointDataloader(batch_size = 2, depth=4, th_count=th_count)

net = networks.Unet(block = blocks.ConvBlock, 
    inplanes = 64, 
    in_dim=3, 
    labels=dataloader.labels,
    depth=depth, 
    norm_layer=torch.nn.InstanceNorm2d
)

fit(net, 
    dataloader.trainloader, 
    dataloader.validationloader, 
    epochs = 1000, 
    lower_learning_period = 3
)       