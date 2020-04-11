import torch
from model.resnet import YoloNet, ResNetBackbone, PreActivationBlock
from model_fitting.train import fit
from torchvision.datasets import CocoDetection
import torchvision.transforms as transforms
from data_loader.augmentation import PairCompose, OutputTransform, TargetTransform, TargetTransformToBoxes, PaddTransform

th_count = 1

classes = [(1, 'person'), (2, 'bicycle'), (3, 'car'), (4, 'motorcycle'), (5, 'airplane'), (6, 'bus'), (7, 'train'), (8, 'truck'), (9, 'boat'), (10, 'traffic light'), (11, 'fire hydrant'), (13, 'stop sign'), (14, 'parking meter'), (15, 'bench'), (16, 'bird'), (17, 'cat'), (18, 'dog'), (19, 'horse'), (20, 'sheep'), (21, 'cow'), (22, 'elephant'), (23, 'bear'), (24, 'zebra'), (25, 'giraffe'), (27, 'backpack'), (28, 'umbrella'), (31, 'handbag'), (32, 'tie'), (33, 'suitcase'), (34, 'frisbee'), (35, 'skis'), (36, 'snowboard'), (37, 'sports ball'), (38, 'kite'), (39, 'baseball bat'), (40, 'baseball glove'), (41, 'skateboard'), (42, 'surfboard'), (43, 'tennis racket'), (44, 'bottle'), (46, 'wine glass'), (47, 'cup'), (48, 'fork'), (49, 'knife'), (50, 'spoon'), (51, 'bowl'), (52, 'banana'), (53, 'apple'), (54, 'sandwich'), (55, 'orange'), (56, 'broccoli'), (57, 'carrot'), (58, 'hot dog'), (59, 'pizza'), (60, 'donut'), (61, 'cake'), (62, 'chair'), (63, 'couch'), (64, 'potted plant'), (65, 'bed'), (67, 'dining table'), (70, 'toilet'), (72, 'tv'), (73, 'laptop'), (74, 'mouse'), (75, 'remote'), (76, 'keyboard'), (77, 'cell phone'), (78, 'microwave'), (79, 'oven'), (80, 'toaster'), (81, 'sink'), (82, 'refrigerator'), (84, 'book'), (85, 'clock'), (86, 'vase'), (87, 'scissors'), (88, 'teddy bear'), (89, 'hair drier'), (90, 'toothbrush')]
ratios = [1.0]
# transforms = PairCompose([TargetTransform(prior_box_size=32, classes=classes, ratios=ratios, stride=32), TargetTransformToBoxes(prior_box_size=32, classes=classes, ratios=ratios, stride=32), OutputTransform()])
transforms = PairCompose([PaddTransform(pad_size=32), TargetTransform(prior_box_size=32, classes=classes, ratios=ratios, stride=32), OutputTransform()])
trainset      = CocoDetection(root = '/Data/Coco/train2017', annFile = '/Data/Coco/annotations_trainval2017/annotations/instances_train2017.json', transforms=transforms)
validationset = CocoDetection(root = '/Data/Coco/val2017'  , annFile = '/Data/Coco/annotations_trainval2017/annotations/instances_val2017.json'  , transforms=transforms)

if th_count==1:
    trainset.ids      = trainset.ids[:100]
    validationset.ids = validationset.ids[:100]

trainloader      = torch.utils.data.DataLoader(trainset,      batch_size=1,   shuffle=True,  num_workers=th_count, pin_memory=False)
validationloader = torch.utils.data.DataLoader(validationset, batch_size=1,   shuffle=False, num_workers=th_count, pin_memory=False)

trainloader.cats = classes
validationloader.cats = classes

backbone = ResNetBackbone(block = PreActivationBlock, layers = [2, 2, 2, 2])
net = YoloNet(backbone, classes = len(trainset.coco.cats), ratios=ratios)
net.cuda()

fit(net, trainloader, validationloader, chp_prefix="coco", box_transform = TargetTransformToBoxes(prior_box_size=32, classes=classes, ratios=ratios, stride=32), epochs=1000, lower_learning_period=100)