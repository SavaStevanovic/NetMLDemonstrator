import torch.nn as nn
from model import utils
import itertools
import functools
import torch
import numpy as np
import torchvision.models as models
from torchvision.models import ResNet50_Weights

class ResNetBackbone(nn.Module, utils.Identifier):

    def __init__(self, block, block_counts, inplanes, norm_layer=nn.InstanceNorm2d):
        super(ResNetBackbone, self).__init__()

        self.feature_count = len(block_counts)
        self.block_counts = block_counts
        self.feature_start_layer = 1
        self.depth = self.feature_start_layer + self.feature_count
        self.block = block
        self._norm_layer = norm_layer
        self.inplanes = inplanes
        self.first_layer = nn.Sequential(
            nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False),
            norm_layer(self.inplanes),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.layers = nn.ModuleList([self._make_layer(block, int(i>0)+1, layer_count, int(i>0)+1) for i, layer_count in enumerate(block_counts)])
           
    def _make_layer(self, block, expansion, blocks, stride=1):
        outplanes = self.inplanes*expansion
        layers = []
        layers.append(block(self.inplanes, outplanes, stride))
        self.inplanes *= expansion
        for _ in range(1, blocks):
            layers.append(block(outplanes, outplanes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.first_layer(x)

        outputs = [x]
        for l in self.layers:
            outputs.append(l(outputs[-1]))

        return tuple(outputs[1:])


class FeaturePyramidNet(nn.Module, utils.Identifier):
    def __init__(self, backbone, classes, ratios):
        super(FeaturePyramidNet, self).__init__()

        self.classes = classes
        self.ratios = ratios
        self.backbone = functools.reduce(lambda b,m : m(b),backbone[::-1])
        self.inplanes = self.backbone.inplanes
        self.feature_count = self.backbone.feature_count
        self.feature_start_layer = self.backbone.feature_start_layer
        self.depth = self.backbone.depth

        feature_range = range(self.feature_start_layer, self.feature_start_layer + self.feature_count)
        self.prior_box_sizes = [32*2**i for i in feature_range]
        self.strides = [2**(i+1) for i in feature_range]

        self.ranges = DetectionRanges(len(classes), len(ratios))
        self.head = nn.Conv2d(256, self.ranges.output_size, kernel_size=1)  

    def forward(self, x):
        features = self.backbone(x)
        features = tuple(self.ranges.activate_output(self.head(feature)) for feature in features)

        return features
        

class RetinaNet(nn.Module, utils.Identifier):
    def __init__(self, backbone, classes, ratios, scales=[2 ** (i / 3) for i in range(3)]):
        super(RetinaNet, self).__init__()

        self.classes = classes
        self.ratios = ratios
        self.backbone = functools.reduce(lambda b,m : m(b),backbone[::-1])
        self.inplanes = self.backbone.inplanes
        self.feature_count = self.backbone.feature_count
        self.feature_start_layer = self.backbone.feature_start_layer
        self.depth = self.backbone.depth

        feature_range = range(self.feature_start_layer, self.feature_start_layer + self.feature_count)
        self.prior_box_sizes = [32*2**i for i in feature_range]
        self.strides = [2**(i+1) for i in feature_range]

        self.ranges = DetectionRanges(len(classes), len(ratios))
        regression_layers = list(itertools.chain.from_iterable([[nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True)] for _ in range(4)]))
        classification_layers = list(itertools.chain.from_iterable([[nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256),nn.ReLU(inplace=True)] for _ in range(4)]))

        self.regression_head = nn.Sequential(
            *regression_layers,
            nn.Conv2d(256, self.ranges.object_per_cell * len(self.ranges.size + self.ranges.offset), kernel_size=3, padding=1),
        )
        o_size = self.ranges.object_per_cell * len(self.ranges.objectness + self.ranges.classes)
        biased_conv = nn.Conv2d(256, o_size, kernel_size=3, padding=1)
        biased_conv.bias.data = torch.autograd.Variable(torch.from_numpy(np.array([-1.995635195 for _ in range(o_size)])).float())
        self.classification_head = nn.Sequential(
            *classification_layers,
            biased_conv,
        )

    def forward(self, x):
        features = self.backbone(x)
        
        regression_features = tuple(self.regression_head(feature) for feature in features)
        regression_features = tuple(x.view(x.shape[0], self.ranges.object_per_cell, -1, x.shape[2], x.shape[3]) for x in regression_features)
        classification_features = tuple(self.classification_head(feature) for feature in features)
        classification_features = tuple(x.view(x.shape[0], self.ranges.object_per_cell, -1, x.shape[2], x.shape[3]) for x in classification_features)
        features = tuple(torch.cat(
            (classification_features[i][:, :, :len(self.ranges.objectness)], 
            regression_features[i], 
            classification_features[i][:, :, len(self.ranges.objectness):]), 2) for i in range(len(features)))

        features = tuple(self.ranges.activate_output(feature) for feature in features)

        return features

class YoloV2(nn.Module, utils.Identifier):
    def __init__(self, classes, ratios=[1.0]):
        super(YoloV2, self).__init__()

        self.feature_count = 1
        self.depth = 5
        self.feature_start_layer = 4
        self.inplanes = 64
        self.classes = classes
        self.ratios = ratios

        feature_range = range(self.feature_start_layer, self.feature_start_layer + self.feature_count)
        self.prior_box_sizes = [32*2**i for i in feature_range]
        self.strides = [2**(i+1) for i in feature_range]
        self.ranges = DetectionRanges(len(classes), len(ratios))

        modules = list(models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1).children())[:-2]
        self._backbone = nn.Sequential(*modules)
        self.grad_backbone(False)
        
        self.stage3_conv2 = nn.Conv2d(2048, len(self.ratios) * (5 + len(self.classes)), 1, 1, 0, bias=False)

    def grad_backbone(self, freeze):
        for param in self._backbone.parameters():
            param.requires_grad = freeze
            
    
    def unlock_layer(self):
        for i, param in reversed(list(enumerate(self._backbone.parameters()))):
            if not param.requires_grad:
                param.requires_grad = True
                print(f"Layer {i} unlocked")
                return
        
        
        
    def forward(self, input):
        output = self._backbone(input)
        output = self.stage3_conv2(output)

        return self.ranges.activate_output(output),

class YoloNet(nn.Module, utils.Identifier):
    def __init__(self, backbone, classes, ratios=[1.0]):
        super(YoloNet, self).__init__()
        
        self.feature_count = 1
        self.backbone = functools.reduce(lambda b,m : m(b),backbone[::-1])
        self.depth = self.backbone.depth
        self.feature_start_layer = self.backbone.feature_start_layer + self.backbone.feature_count - self.feature_count
        self.inplanes = self.backbone.inplanes
        self.classes = classes
        self.ratios = ratios

        feature_range = range(self.feature_start_layer, self.feature_start_layer + self.feature_count)
        self.prior_box_sizes = [32*2**i for i in feature_range]
        self.strides = [2**(i+1) for i in feature_range]

        self.ranges = DetectionRanges(len(classes), len(ratios))
        self.head = nn.Conv2d(self.inplanes, self.ranges.output_size, kernel_size=1)  

    def forward(self, x):
        boutput = self.backbone(x)[-1]
        output = self.head(boutput)
        return self.ranges.activate_output(output),


class FeaturePyramidBackbone(nn.Module, utils.Identifier):
    def __init__(self, backbone):
        super(FeaturePyramidBackbone, self).__init__()

        self.backbone = backbone
        self.inplanes = backbone.inplanes
        self.feature_count = backbone.feature_count - 1
        self.feature_start_layer = backbone.feature_start_layer
        self.depth = backbone.depth
        # Top layer
        self.toplayer = nn.Conv2d(self.inplanes, 256, kernel_size=1)  # Reduce channels

        self.up = nn.Upsample(scale_factor=2)

        self.lat_layers = nn.ModuleList([nn.Conv2d(self.inplanes // 2**i, 256, kernel_size=1) for i in range(self.feature_count, 0, -1)])

        self.smooth_layers = nn.ModuleList([nn.Conv2d(256, 256, kernel_size=1) for i in range(self.feature_count, 0, -1)]) 

    def forward(self, x):
        boutputs = self.backbone(x)
        top_layer = self.toplayer(boutputs[-1])

        features = [top_layer]
        for i in range(self.feature_count-1, -1 , -1):
            features.append(self.lat_layers[i](boutputs[i]) + self.up(features[-1]))
        features.pop(0)

        features = [self.smooth_layers[i](features[i]) for i in range(self.feature_count)]

        return tuple(features[::-1])

        
class DetectionRanges(object):
    def __init__(self, classes_len, ratios_len):
        self.object_per_cell = ratios_len
        self.object_range = 5+classes_len
        self.output_size = self.object_per_cell * self.object_range
        self.objectness = [0]
        self.size       = [1,2]
        self.offset     = [3,4]
        self.classes    = list(range(5, self.object_range))

    def activate_output(self, x):
        if len(x.size())==4:
            x = x.view(x.shape[0], self.object_per_cell, self.object_range, x.shape[2], x.shape[3])
        x[:, :, self.objectness] = x[:, :, self.objectness].sigmoid()
        x[:, :, self.offset] = x[:, :,self.offset].sigmoid()
        x[:, :, self.classes] = x[:, :,self.classes].log_softmax(2)
        return x