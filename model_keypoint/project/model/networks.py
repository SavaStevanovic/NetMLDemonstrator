import torch.nn as nn
from model import utils
import itertools
import functools
import torch
import numpy as np


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
        
class OpenPoseNet(nn.Module, utils.Identifier):
    def __init__(self, backbone, paf_stages, map_stages, block, block_count, paf_planes, map_planes):
        super(OpenPoseNet, self).__init__()
        self.backbone = functools.reduce(lambda b,m : m(b),backbone[::-1])
        self.block = block
        self.adapter = nn.Sequential(
            nn.Conv2d(self.backbone.channels   , self.backbone.channels//2, kernel_size=1, bias=True, padding=0),
            nn.Conv2d(self.backbone.channels//2, self.backbone.channels//4, kernel_size=1, bias=True, padding=0),
        )
        self.channels = self.backbone.channels//4
        self.first_paf = block(self.channels, self.channels, paf_planes, block_count)
        self.pafs = [block(self.channels + paf_planes) for _ in range(paf_stages-1)]
        self.first_map = block(self.channels + paf_planes, self.channels, map_planes, block_count)
        self.maps = [block(self.channels + map_planes) for _ in range(map_stages-1)]
    
    def forward(self, x):
        feature = self.adapter(self.backbone(x))
        out = self.first_paf(feature)
        pafs_features = [out]
        for paf in self.pafs:
            out = paf(pafs_features[-1])
            pafs_features.append(out)
            out = torch.cat([out, feature], 1)

        out = self.first_map(feature)
        maps_features = [out]
        for mapf in self.maps:
            out = mapf(maps_features[-1])
            maps_features.append(out)
            out = torch.cat([out, feature], 1)
        
        return pafs_features, maps_features

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

        self.stage1_conv1 = nn.Sequential(nn.Conv2d(3, 32, 3, 1, 1, bias=False), nn.BatchNorm2d(32),
                                          nn.LeakyReLU(0.1, inplace=True), nn.MaxPool2d(2, 2))
        self.stage1_conv2 = nn.Sequential(nn.Conv2d(32, 64, 3, 1, 1, bias=False), nn.BatchNorm2d(64),
                                          nn.LeakyReLU(0.1, inplace=True), nn.MaxPool2d(2, 2))
        self.stage1_conv3 = nn.Sequential(nn.Conv2d(64, 128, 3, 1, 1, bias=False), nn.BatchNorm2d(128),
                                          nn.LeakyReLU(0.1, inplace=True))
        self.stage1_conv4 = nn.Sequential(nn.Conv2d(128, 64, 1, 1, 0, bias=False), nn.BatchNorm2d(64),
                                          nn.LeakyReLU(0.1, inplace=True))
        self.stage1_conv5 = nn.Sequential(nn.Conv2d(64, 128, 3, 1, 1, bias=False), nn.BatchNorm2d(128),
                                          nn.LeakyReLU(0.1, inplace=True), nn.MaxPool2d(2, 2))
        self.stage1_conv6 = nn.Sequential(nn.Conv2d(128, 256, 3, 1, 1, bias=False), nn.BatchNorm2d(256),
                                          nn.LeakyReLU(0.1, inplace=True))
        self.stage1_conv7 = nn.Sequential(nn.Conv2d(256, 128, 1, 1, 0, bias=False), nn.BatchNorm2d(128),
                                          nn.LeakyReLU(0.1, inplace=True))
        self.stage1_conv8 = nn.Sequential(nn.Conv2d(128, 256, 3, 1, 1, bias=False), nn.BatchNorm2d(256),
                                          nn.LeakyReLU(0.1, inplace=True), nn.MaxPool2d(2, 2))
        self.stage1_conv9 = nn.Sequential(nn.Conv2d(256, 512, 3, 1, 1, bias=False), nn.BatchNorm2d(512),
                                          nn.LeakyReLU(0.1, inplace=True))
        self.stage1_conv10 = nn.Sequential(nn.Conv2d(512, 256, 1, 1, 0, bias=False), nn.BatchNorm2d(256),
                                           nn.LeakyReLU(0.1, inplace=True))
        self.stage1_conv11 = nn.Sequential(nn.Conv2d(256, 512, 3, 1, 1, bias=False), nn.BatchNorm2d(512),
                                           nn.LeakyReLU(0.1, inplace=True))
        self.stage1_conv12 = nn.Sequential(nn.Conv2d(512, 256, 1, 1, 0, bias=False), nn.BatchNorm2d(256),
                                           nn.LeakyReLU(0.1, inplace=True))
        self.stage1_conv13 = nn.Sequential(nn.Conv2d(256, 512, 3, 1, 1, bias=False), nn.BatchNorm2d(512),
                                           nn.LeakyReLU(0.1, inplace=True))

        self.stage2_a_maxpl = nn.MaxPool2d(2, 2)
        self.stage2_a_conv1 = nn.Sequential(nn.Conv2d(512, 1024, 3, 1, 1, bias=False),
                                            nn.BatchNorm2d(1024), nn.LeakyReLU(0.1, inplace=True))
        self.stage2_a_conv2 = nn.Sequential(nn.Conv2d(1024, 512, 1, 1, 0, bias=False), nn.BatchNorm2d(512),
                                            nn.LeakyReLU(0.1, inplace=True))
        self.stage2_a_conv3 = nn.Sequential(nn.Conv2d(512, 1024, 3, 1, 1, bias=False), nn.BatchNorm2d(1024),
                                            nn.LeakyReLU(0.1, inplace=True))
        self.stage2_a_conv4 = nn.Sequential(nn.Conv2d(1024, 512, 1, 1, 0, bias=False), nn.BatchNorm2d(512),
                                            nn.LeakyReLU(0.1, inplace=True))
        self.stage2_a_conv5 = nn.Sequential(nn.Conv2d(512, 1024, 3, 1, 1, bias=False), nn.BatchNorm2d(1024),
                                            nn.LeakyReLU(0.1, inplace=True))
        self.stage2_a_conv6 = nn.Sequential(nn.Conv2d(1024, 1024, 3, 1, 1, bias=False), nn.BatchNorm2d(1024),
                                            nn.LeakyReLU(0.1, inplace=True))
        self.stage2_a_conv7 = nn.Sequential(nn.Conv2d(1024, 1024, 3, 1, 1, bias=False), nn.BatchNorm2d(1024),
                                            nn.LeakyReLU(0.1, inplace=True))

        self.stage2_b_conv = nn.Sequential(nn.Conv2d(512, 64, 1, 1, 0, bias=False), nn.BatchNorm2d(64),
                                           nn.LeakyReLU(0.1, inplace=True))

        self.stage3_conv1 = nn.Sequential(nn.Conv2d(256 + 1024, 1024, 3, 1, 1, bias=False), nn.BatchNorm2d(1024),
                                          nn.LeakyReLU(0.1, inplace=True))
        self.stage3_conv2 = nn.Conv2d(1024, len(self.ratios) * (5 + len(self.classes)), 1, 1, 0, bias=False)

    def forward(self, input):
        output = self.stage1_conv1(input)
        output = self.stage1_conv2(output)
        output = self.stage1_conv3(output)
        output = self.stage1_conv4(output)
        output = self.stage1_conv5(output)
        output = self.stage1_conv6(output)
        output = self.stage1_conv7(output)
        output = self.stage1_conv8(output)
        output = self.stage1_conv9(output)
        output = self.stage1_conv10(output)
        output = self.stage1_conv11(output)
        output = self.stage1_conv12(output)
        output = self.stage1_conv13(output)

        residual = output

        output_1 = self.stage2_a_maxpl(output)
        output_1 = self.stage2_a_conv1(output_1)
        output_1 = self.stage2_a_conv2(output_1)
        output_1 = self.stage2_a_conv3(output_1)
        output_1 = self.stage2_a_conv4(output_1)
        output_1 = self.stage2_a_conv5(output_1)
        output_1 = self.stage2_a_conv6(output_1)
        output_1 = self.stage2_a_conv7(output_1)

        output_2 = self.stage2_b_conv(residual)
        batch_size, num_channel, height, width = output_2.data.size()
        output_2 = output_2.view(batch_size, int(num_channel / 4), height, 2, width, 2).contiguous()
        output_2 = output_2.permute(0, 3, 5, 1, 2, 4).contiguous()
        output_2 = output_2.view(batch_size, -1, int(height / 2), int(width / 2))

        output = torch.cat((output_1, output_2), 1)
        output = self.stage3_conv1(output)
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

class VGGNetBackbone(nn.Sequential, utils.Identifier):
    def __init__(self, planes, blocks):
        self.channels = planes
        self.planes = planes
        layers = [nn.Conv2d(3, self.channels, kernel_size=3, bias=True, padding=1)]
        layers.extend([nn.Conv2d(self.channels, self.channels, kernel_size=3, bias=True, padding=1) for _ in range(blocks[0]-1)])
        for b in blocks[1:]:
            layers.append(nn.MaxPool2d(2, 2))
            layers.append(nn.Conv2d(self.channels, self.channels*2, kernel_size=3, bias=True, padding=1))
            self.channels*=2
            layers.extend([nn.Conv2d(self.channels, self.channels, kernel_size=3, bias=True, padding=1) for _ in range(b-1)])
           
        super(VGGNetBackbone, self).__init__(*layers)

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