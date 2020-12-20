import torch
from model import blocks
from model import networks
from data_loader.unified_dataloader import UnifiedKeypointDataloader
from model_fitting.train import fit
import os


th_count = 24

dataloader = UnifiedKeypointDataloader(batch_size = 6, th_count=th_count)
backbone = networks.VGGNetBackbone(inplanes = 64, block_counts = [2, 2, 4, 2])
net = networks.OpenPoseNet([backbone], 4, 1, blocks.PoseCNNStage, 10, len(dataloader.trainloader.skeleton)*2, len(dataloader.trainloader.parts)+1)
# net = networks.CocoPoseNet()

fit(net, 
    dataloader.trainloader, 
    dataloader.validationloader, 
    postprocessing = dataloader.postprocessing, 
    epochs = 1000, 
    lower_learning_period = 3
)      