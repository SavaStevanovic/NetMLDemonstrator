import torch
from model import blocks
from model import networks
from data_loader.unified_dataloader import UnifiedKeypointDataloader
from model_fitting.train import fit
import os

torch.backends.cudnn.benchmark = True

th_count = 24

dataloader = UnifiedKeypointDataloader(batch_size=10, th_count=th_count)
backbone = networks.VGGNetBackbone(planes = 64, blocks = [2, 2, 4, 2])
net = networks.OpenPoseNet([backbone], 1, 1, blocks.PoseCNNStage, 10, len(dataloader.trainloader.skeleton)*2, len(dataloader.trainloader.parts))


fit(net, 
    dataloader.trainloader, 
    dataloader.validationloader, 
    postprocessing = dataloader.postprocessing, 
    epochs = 1000, 
    lower_learning_period = 3
)      