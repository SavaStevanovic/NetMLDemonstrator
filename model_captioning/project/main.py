import torch
from model import blocks
from model import networks
from data_loader.unified_dataloader import UnifiedDataloader
from model_fitting.train import fit
import os

th_count = 12

dataloader = UnifiedDataloader(batch_size = 32, th_count=th_count)

net = networks.AttLSTM(512, (224, 224), dataloader.vectorizer)
net.grad_backbone(False)

fit(net, 
    dataloader.trainloader, 
    dataloader.validationloader, 
    epochs = 1000, 
    lower_learning_period = 3
)       