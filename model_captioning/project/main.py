import torch
from model import blocks
from model import networks
from data_loader.unified_dataloader import UnifiedKeypointDataloader
from model_fitting.train import fit
import os

th_count = 96

dataloader = UnifiedKeypointDataloader(batch_size = 16, th_count=th_count)

net = networks.LSTM(256, (224, 224), dataloader.vectorizer.vocab)

fit(net, 
    dataloader.trainloader, 
    dataloader.validationloader, 
    epochs = 1000, 
    lower_learning_period = 3
)       