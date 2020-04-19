import torch
from model.networks import YoloNet, ResNetBackbone, PreActivationBlock
from model_fitting.train import fit
from data_loader.coco_dataset import CocoDetectionDatasetProvider

th_count = 12
ratios = [1.0]
preload = True

coco_provider = CocoDetectionDatasetProvider(annDir='/Data/Coco/', th_count=th_count, ratios=ratios)

backbone = ResNetBackbone(block = PreActivationBlock, layers = [2, 2, 2, 2])

if preload:
    net = torch.load('checkpoints/coco_checkpoints.pth')
else:
    net = YoloNet(backbone, classes = coco_provider.classes, ratios=ratios)
net.cuda()

fit(net, coco_provider.trainloader, coco_provider.validationloader, chp_prefix="coco", box_transform = coco_provider.target_to_box_transform, epochs=1000, lower_learning_period=20)