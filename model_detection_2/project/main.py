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
classes = [(1, 'person'), (2, 'bicycle'), (3, 'car'), (4, 'motorcycle'), (5, 'airplane'), (6, 'bus'), (7, 'train'), (8, 'truck'), (9, 'boat'), (10, 'traffic light'), (11, 'fire hydrant'), (13, 'stop sign'), (14, 'parking meter'), (15, 'bench'), (16, 'bird'), (17, 'cat'), (18, 'dog'), (19, 'horse'), (20, 'sheep'), (21, 'cow'), (22, 'elephant'), (23, 'bear'), (24, 'zebra'), (25, 'giraffe'), (27, 'backpack'), (28, 'umbrella'), (31, 'handbag'), (32, 'tie'), (33, 'suitcase'), (34, 'frisbee'), (35, 'skis'), (36, 'snowboard'), (37, 'sports ball'), (38, 'kite'), (39, 'baseball bat'), (40, 'baseball glove'), (41, 'skateboard'), (42, 'surfboard'), (43, 'tennis racket'),
           (44, 'bottle'), (46, 'wine glass'), (47, 'cup'), (48, 'fork'), (49, 'knife'), (50, 'spoon'), (51, 'bowl'), (52, 'banana'), (53, 'apple'), (54, 'sandwich'), (55, 'orange'), (56, 'broccoli'), (57, 'carrot'), (58, 'hot dog'), (59, 'pizza'), (60, 'donut'), (61, 'cake'), (62, 'chair'), (63, 'couch'), (64, 'potted plant'), (65, 'bed'), (67, 'dining table'), (70, 'toilet'), (72, 'tv'), (73, 'laptop'), (74, 'mouse'), (75, 'remote'), (76, 'keyboard'), (77, 'cell phone'), (78, 'microwave'), (79, 'oven'), (80, 'toaster'), (81, 'sink'), (82, 'refrigerator'), (84, 'book'), (85, 'clock'), (86, 'vase'), (87, 'scissors'), (88, 'teddy bear'), (89, 'hair drier'), (90, 'toothbrush')]

backbone = networks.ResNetBackbone(
    inplanes=64, block=blocks.BasicBlock, block_counts=[3, 4, 6, 3])
net = networks.RetinaNet(backbone=[
                         networks.FeaturePyramidBackbone, backbone], classes=classes, ratios=ratios)

coco_provider = UnifiedDataloader(net, batch_size=8, th_count=th_count)

fit(net, coco_provider.trainloader, coco_provider.validationloader,
    dataset_name=dataset_name,
    box_transform=TargetTransformToBoxes(prior_box_sizes=net.prior_box_sizes, classes=net.classes, ratios=net.ratios, strides=net.strides),
    epochs=1000,
    lower_learning_period=3)
