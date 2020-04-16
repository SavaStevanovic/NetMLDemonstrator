import torch
from model.resnet import YoloNet, ResNetBackbone, PreActivationBlock
from model_fitting.train import fit
from data_loader.coco_dataset import CocoDetectionDatasetProvider
from data_loader.augmentation import TargetTransformToBoxes

th_count = 1
ratios = [1.0]
coco_provider = CocoDetectionDatasetProvider(annDir='/Data/Coco/', th_count=th_count, ratios=ratios)

backbone = ResNetBackbone(block = PreActivationBlock, layers = [2, 2, 2, 2])
net = YoloNet(backbone, classes = coco_provider.classes, ratios=ratios)
net.cuda()

target_to_box_transform = TargetTransformToBoxes(prior_box_size=32, classes=coco_provider.classes, ratios=ratios, stride=32)
fit(net, coco_provider.trainloader, coco_provider.validationloader, chp_prefix="coco", box_transform = target_to_box_transform, epochs=1000, lower_learning_period=20)