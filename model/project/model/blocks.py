import torch.nn as nn
from model import utils


class ResidualBlock(nn.Module, utils.Identifier):
    expansion = 1

    def __init__(self, block, downsample=None):
        super(ResidualBlock, self).__init__()

        self.block = block
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.block(x)

        if self.downsample:
            identity = self.downsample(x)
        out += identity

        return out


class PreActivationBlock(nn.Module, utils.Identifier):

    def __init__(self, inplanes, planes, stride=1, norm_layer=nn.InstanceNorm2d):
        super(PreActivationBlock, self).__init__()

        self.sequential = nn.Sequential(
            norm_layer(inplanes),
            nn.ReLU(inplace=True),
            utils.conv3x3(inplanes, planes, stride),
            norm_layer(planes),
            nn.ReLU(inplace=True),
            utils.conv3x3(planes, planes)
        )

    def forward(self, x):
        return self.sequential(x)


class InvertedBlock(nn.Module, utils.Identifier):

    def __init__(self, inplanes, planes, stride=1, norm_layer=nn.InstanceNorm2d, expand_ratio=6):
        super(InvertedBlock, self).__init__()
        self.expanded_dim = inplanes * expand_ratio

        self.sequential = nn.Sequential(
            utils.conv1x1(inplanes, self.expanded_dim),
            norm_layer(self.expanded_dim),
            nn.ReLU6(inplace=True),

            utils.conv3x3(self.expanded_dim,  self.expanded_dim, stride, groups=self.expanded_dim),
            norm_layer(self.expanded_dim),
            nn.ReLU6(inplace=True),

            utils.conv1x1(self.expanded_dim, planes),
            norm_layer(planes)
        )

    def forward(self, x):
        return self.sequential(x)


class SqueezeExcitationBlock(nn.Module, utils.Identifier):

    def __init__(self, channel, reduction=16):
        super(SqueezeExcitationBlock, self).__init__()
        self.sequential = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channel, channel // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.sequential(x)
        return x * out


class EfficientNetBlock(nn.Module, utils.Identifier):

    def __init__(self, inplanes, planes, stride=1, norm_layer=nn.InstanceNorm2d, expand_ratio=6, reduction=16):
        super(EfficientNetBlock, self).__init__()

        self.sequential = nn.Sequential(
            InvertedBlock(inplanes, planes, stride, norm_layer, expand_ratio),
            SqueezeExcitationBlock(planes, reduction)
        )

    def forward(self, x):
        return self.sequential(x)