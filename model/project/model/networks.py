import torch
import torch.nn as nn

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def convUp(in_planes, out_planes):
    """deconvolution convolution"""
    return nn.ConvTranspose2d(in_channels , out_planes, kernel_size=2, stride=2)


class Identifier(object):
    def __init__(self):
        pass

    def get_identifier(self):
        idt = self.__class__.__name__
        if hasattr(self, 'backbone') and hasattr(self.backbone, 'get_identifier'): 
            idt+='/'+self.backbone.get_identifier() 
        return idt

class PreActivationBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, norm_layer=nn.InstanceNorm2d):
        super(PreActivationBlock, self).__init__()
        self.bn1 = norm_layer(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = conv3x3(inplanes, planes, stride)

        self.bn2 = norm_layer(planes)
        self.conv2 = conv3x3(planes, planes)

        self.downsample = downsample
        self.stride = stride
        self.planes = planes

    def forward(self, x):
        identity = x

        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity

        return out

class ResNetBackbone(nn.Module, Identifier):

    def __init__(self, block, layers, norm_layer=nn.InstanceNorm2d, multiplier=2):
        super(ResNetBackbone, self).__init__()

        self.groups = 1
        self._norm_layer = norm_layer
        self.dilation = 1
        self.inplanes = 32
        self.multiplier = multiplier
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.layer1 = self._make_layer(block, self.inplanes * 2, layers[0])
        self.layer2 = self._make_layer(block, self.inplanes * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(block, self.inplanes * 2, layers[2], stride=2)
        self.layer4 = self._make_layer(block, self.inplanes * 2, layers[3], stride=2)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(conv1x1(self.inplanes, planes * block.expansion, stride), norm_layer(planes * block.expansion),)
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups, norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        return x1, x2, x3, x4

class FPN(nn.Module, Identifier):
    def __init__(self, backbone):
        super(FPN, self).__init__()

        self.backbone = backbone
        self.multiplier = backbone.multiplier
        self.inplanes = backbone.inplanes
        # Top layer
        self.toplayer = nn.Conv2d(self.inplanes * 8 * multiplier, 256, kernel_size=1)  # Reduce channels

        # Smooth layers
        self.smooth1 = convUp(self.inplanes * 1 * multiplier, 256)
        self.smooth2 = convUp(self.inplanes * 2 * multiplier, 256)
        self.smooth3 = convUp(self.inplanes * 4 * multiplier, 256)

    def forward(self, x):
        # Bottom-up
        l1, l2, l3, l4 = self.backbone(x)
        # Top-down
        p4 = self.toplayer(l4)
        p3 = p4 + self.smooth3(l3)
        p2 = p3 + self.smooth2(l2)
        p1 = p2 + self.smooth1(l1)
        return p1, p2, p3

class RetinaNet(nn.Module, Identifier):
    def __init__(self, backbone, classes = 80, ratios=[0.5, 1.0, 2.0], scales=[2 ** (i / 3) for i in range(3)]):
        super(FPN, self).__init__()

        self.backbone = backbone
        self.multiplier = backbone.multiplier
        self.inplanes = backbone.inplanes
        # Top layer
        self.toplayer = nn.Conv2d(self.inplanes * 8 * multiplier, 256, kernel_size=1)  # Reduce channels

        # Smooth layers
        self.smooth1 = convUp(self.inplanes * 1 * multiplier, 256)
        self.smooth2 = convUp(self.inplanes * 2 * multiplier, 256)
        self.smooth3 = convUp(self.inplanes * 4 * multiplier, 256)

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

class YoloNet(nn.Module, Identifier):
    def __init__(self, backbone, classes, ratios=[1.0]):
        super(YoloNet, self).__init__()

        self.backbone = backbone
        self.inplanes = backbone.inplanes
        self.classes = classes
        self.ratios = ratios
        self.object_per_cell = len(ratios)
        self.object_range = 5+len(self.classes)
        classes_range = range(5, self.object_range)
        self.output_size = self.object_per_cell * self.object_range
        self.ranges = YoloRanges(objectness = [0], size = [1,2], offset = [3,4], classes = classes_range, object_range = self.object_range)
        self.toplayer = nn.Conv2d(self.inplanes, self.output_size, kernel_size=1)  

    def output_actrivations(self, x):
        x = x.view(x.shape[0], self.object_per_cell, self.object_range, x.shape[2], x.shape[3])
        x[:, :, self.ranges.objectness] = x[:, :, self.ranges.objectness].sigmoid()
        x[:, :, self.ranges.offset] = x[:, :,self.ranges.offset].sigmoid()
        x[:, :, self.ranges.classes] = x[:, :,self.ranges.classes].log_softmax(2)
        return x

    def forward(self, x):
        _, _, _, boutput = self.backbone(x)
        output = self.toplayer(boutput)
        return self.output_actrivations(output)

class YoloRanges(object):
    def __init__(self, objectness, size, offset, classes, object_range):
        self.objectness = objectness
        self.size       = size
        self.offset     = offset
        self.classes    = classes
        self.object_range = object_range

