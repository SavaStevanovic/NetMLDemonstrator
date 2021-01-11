import torch
from model import blocks
from model import networks
from data_loader.unified_dataloader import UnifiedKeypointDataloader
from model_fitting.train import fit
import os

th_count = 24
depth = 4

dataloader = UnifiedKeypointDataloader(batch_size = 6, depth=4, th_count=th_count)

net = networks.Unet(block = blocks.ConvBlock, 
    inplanes = 32, 
    in_dim=3, 
    out_dim=len(dataloader.clasess_inds)+1, 
    depth=depth, 
    norm_layer=torch.nn.InstanceNorm2d)

fit(net, 
    dataloader.trainloader, 
    dataloader.validationloader, 
    epochs = 1000, 
    lower_learning_period = 3
)       