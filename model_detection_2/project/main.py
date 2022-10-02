import torch
from data_loader.unified_dataloader import UnifiedDataloader
from model import blocks
from model import networks
from model_fitting.train import fit
from visualization.output_transform import TargetTransformToBoxes

torch.backends.cudnn.benchmark = True

th_count = 1
ratios = [0.5, 1.0, 2.0]
dataset_name = 'Coco'
block_size = [3, 4, 6, 3]
backbone = networks.ResNetBackbone(
    inplanes=64, block=blocks.BasicBlock, block_counts=block_size)
data_provider = UnifiedDataloader(len(block_size), batch_size=8, th_count=th_count)
net = networks.RetinaNet(backbone=[
                         networks.FeaturePyramidBackbone, backbone], classes=data_provider.classes_map, ratios=ratios)


fit(net, data_provider.trainloader, data_provider.validationloader,
    dataset_name=dataset_name,
    box_transform=TargetTransformToBoxes(prior_box_sizes=net.prior_box_sizes, classes=net.classes, ratios=net.ratios, strides=net.strides),
    epochs=1000,
    lower_learning_period=3)
