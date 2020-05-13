import torch
from model import blocks
from model import networks
from model_fitting.train import fit
from data_loader.coco_dataset import CocoDetectionDatasetProvider
import os

th_count = 1
ratios = [0.5, 1.0, 2.0]
dataset_name = 'Coco'

coco_provider = CocoDetectionDatasetProvider(annDir=os.path.join('/Data', dataset_name), batch_size=16, th_count=th_count, ratios=ratios)

backbone = networks.ResNetBackbone(block = blocks.EfficientNetBlock, layers = [2, 2, 2, 2])

net = networks.YoloNet(backbone, classes = coco_provider.classes, ratios=ratios)

fit(net, coco_provider.trainloader, coco_provider.validationloader, 
    dataset_name = dataset_name, 
    box_transform = coco_provider.target_to_box_transform, 
    epochs=1000, 
    lower_learning_period=10)