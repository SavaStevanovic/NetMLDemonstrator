import torch
from model.networks import YoloNet, ResNetBackbone, PreActivationBlock
from model_fitting.train import fit
from data_loader.coco_dataset import CocoDetectionDatasetProvider
import os

th_count = 12
ratios = [1.0]
dataset_name = 'Coco'

coco_provider = CocoDetectionDatasetProvider(annDir=os.path.join('/Data', dataset_name), batch_size=24, th_count=th_count, ratios=ratios)

backbone = ResNetBackbone(block = PreActivationBlock, layers = [2, 2, 2, 2])

net = YoloNet(backbone, classes = coco_provider.classes, ratios=ratios)

fit(net, coco_provider.trainloader, coco_provider.validationloader, dataset_name = dataset_name, box_transform = coco_provider.target_to_box_transform, epochs=1000, lower_learning_period=20)