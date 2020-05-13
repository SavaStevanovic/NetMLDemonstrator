import torch.nn as nn
from model import utils

class PreActivationBlock(nn.Module, utils.Identifier):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, norm_layer=nn.InstanceNorm2d):
        super(PreActivationBlock, self).__init__()
        self.bn1 = norm_layer(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = utils.conv3x3(inplanes, planes, stride)

        self.bn2 = norm_layer(planes)
        self.conv2 = utils.conv3x3(planes, planes)

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

class InvertedResidualBlock(nn.Module, utils.Identifier):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, norm_layer=nn.InstanceNorm2d, expand_ratio=6):
        super(InvertedResidualBlock, self).__init__()
        self.expanded_dim = inplanes * expand_ratio
        self.stride = stride
        self.planes = planes
        self.downsample = downsample
        self.use_res_connect = self.stride == 1 and inplanes == planes

        self.conv1 = utils.conv1x1(inplanes, self.expanded_dim)
        self.bn1 = norm_layer(self.expanded_dim)
        self.relu = nn.ReLU6(inplace=True)

        self.conv2 = utils.conv3x3(self.expanded_dim,  self.expanded_dim, stride, groups=self.expanded_dim)
        self.bn2 = norm_layer(inplanes)

        self.conv3 = utils.conv1x1(self.expanded_dim, planes)
        self.bn3 = norm_layer(planes)


    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(x)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None and self.use_res_connect:
            identity = self.downsample(x)
            out += identity

        return out