import torch
import torch.nn as nn
from model import utils

class ResNetBackbone(nn.Module, utils.Identifier):

    def __init__(self, block_wrapper, block, block_counts, inplanes, norm_layer=nn.InstanceNorm2d):
        super(ResNetBackbone, self).__init__()

        self.feature_count = len(block_counts)
        self.block_counts = block_counts
        self.feature_start_layer = 1
        self.depth = self.feature_start_layer + self.feature_count
        self.block_wrapper = block_wrapper
        self.block = block
        self._norm_layer = norm_layer
        self.inplanes = inplanes
        self.first_layer = nn.Sequential(
            nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False),
            norm_layer(self.inplanes),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.layers = nn.ModuleList([self._make_layer(block_wrapper, block, int(i>0)+1, layer_count, int(i>0)+1) for i, layer_count in enumerate(block_counts)])
           
    def _make_layer(self, block_wrapper, block, expansion, blocks, stride=1):
        downsample = None
        outplanes = self.inplanes*expansion
        if stride != 1 or expansion != 1:
            downsample = nn.Sequential(
                utils.conv1x1(self.inplanes, outplanes, stride), 
                self._norm_layer(outplanes)
            )
        layers = []
        layers.append(block_wrapper(block(self.inplanes, outplanes, stride, self._norm_layer), downsample=downsample))
        self.inplanes *= expansion
        for _ in range(1, blocks):
            layers.append(block_wrapper(block(outplanes, outplanes, norm_layer=self._norm_layer)))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.first_layer(x)

        outputs = [x]
        for l in self.layers:
            outputs.append(l(outputs[-1]))

        return tuple(outputs[1:])

class FeaturePyramidNet(nn.Module, utils.Identifier):
    def __init__(self, backbone, classes=None, ratios=None):
        super(FeaturePyramidNet, self).__init__()

        self.classes = classes
        self.ratios = ratios
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
        
        self.head=None
        if classes is not None and ratios is not None:
            self.ranges = DetectionRanges(len(classes), len(ratios))
            self.head = nn.Conv2d(256, self.ranges.output_size, kernel_size=1)  

    def forward(self, x):
        boutputs = self.backbone(x)
        top_layer = self.toplayer(boutputs[-1])

        features = [top_layer]
        for i in range(self.feature_count-1, -1 , -1):
            features.append(self.lat_layers[i](boutputs[i]) + self.up(features[-1]))
        features.pop(0)

        features = [self.smooth_layers[i](features[i]) for i in range(self.feature_count)]

        if self.head is not None:
            features = [self.ranges.activate_output(self.head(features[i])) for i in range(self.feature_count)]

        return tuple(features[::-1])

class RetinaNet(nn.Module, utils.Identifier):
    def __init__(self, backbone, classes = 80, ratios=[0.5, 1.0, 2.0], scales=[2 ** (i / 3) for i in range(3)]):
        super(FPN, self).__init__()

        self.backbone = backbone
        self.multiplier = backbone.multiplier
        self.inplanes = backbone.inplanes
        # Top layer
        self.toplayer = nn.Conv2d(self.inplanes * 8 * multiplier, 256, kernel_size=1)  # Reduce channels

        # Smooth layers
        self.smooth1 = utils.convUp(self.inplanes * 1 * multiplier, 256)
        self.smooth2 = utils.convUp(self.inplanes * 2 * multiplier, 256)
        self.smooth3 = utils.convUp(self.inplanes * 4 * multiplier, 256)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        # Bottom-up
        l1, l2, l3, l4 = self.backbone(x)
        # Top-down
        p4 = self.toplayer(l4)
        p3 = p4 + self.smooth3(l3)
        p2 = p3 + self.smooth2(l2)
        p1 = p2 + self.smooth1(l1)
        return p1, p2, p3

class YoloNet(nn.Module, utils.Identifier):
    def __init__(self, backbone, classes, ratios=[1.0]):
        super(YoloNet, self).__init__()
        
        self.feature_count = 1
        self.depth = backbone.depth
        self.feature_start_layer = backbone.feature_start_layer + backbone.feature_count - self.feature_count
        self.backbone = backbone
        self.inplanes = backbone.inplanes
        self.classes = classes
        self.ratios = ratios
        self.ranges = DetectionRanges(len(classes), len(ratios))
        self.head = nn.Conv2d(self.inplanes, self.ranges.output_size, kernel_size=1)  

    def forward(self, x):
        boutput = self.backbone(x)[-1]
        output = self.head(boutput)
        return self.ranges.activate_output(output),

class DetectionRanges(object):
    def __init__(self, classes_len, ratios_len):
        self.object_per_cell = ratios_len
        self.object_range = 5+classes_len
        self.output_size = self.object_per_cell * self.object_range
        self.objectness = [0]
        self.size       = [1,2]
        self.offset     = [3,4]
        self.classes    = range(5, self.object_range)

    def activate_output(self, x):
        x = x.view(x.shape[0], self.object_per_cell, self.object_range, x.shape[2], x.shape[3])
        x[:, :, self.objectness] = x[:, :, self.objectness].sigmoid()
        x[:, :, self.offset] = x[:, :,self.offset].sigmoid()
        x[:, :, self.classes] = x[:, :,self.classes].log_softmax(2)
        return x