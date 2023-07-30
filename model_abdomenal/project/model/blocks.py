import torch.nn as nn
from model import utils
import torch.nn.functional as F
import torch

class BasicBlock(nn.Module):

    def __init__(self, inplanes, planes, stride=1, downsample=None, norm_layer=nn.InstanceNorm2d):
        super(BasicBlock, self).__init__()
        self.sequential = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False),
            norm_layer(planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False),
            norm_layer(planes),
        )

        self.relu = nn.ReLU(inplace=True)   
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.sequential(x)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out



class PreActivationBlock(nn.Module, utils.Identifier):

    def __init__(self, inplanes, planes, stride=1, norm_layer=nn.InstanceNorm2d):
        super(PreActivationBlock, self).__init__()
        self.planes = planes
        self.inplanes = inplanes
        self.stride = stride

        self.sequential = nn.Sequential(
            norm_layer(inplanes),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, planes, kernel_size=3, stride = stride, padding=1, bias=False),
            norm_layer(planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(planes, planes, kernel_size=3, stride = 1, padding=1, bias=False),
        )

    def forward(self, x):
        return self.sequential(x)


class ResidualBlock(nn.Module, utils.Identifier):

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


class InvertedBlock(nn.Module, utils.Identifier):

    def __init__(self, inplanes, planes, stride=1, norm_layer=nn.InstanceNorm2d, expand_ratio=6):
        super(InvertedBlock, self).__init__()
        self.expanded_dim = inplanes * expand_ratio

        self.sequential = nn.Sequential(
            conv1x1(inplanes, self.expanded_dim),
            norm_layer(self.expanded_dim),
            nn.ReLU6(inplace=True),

            conv3x3(self.expanded_dim,  self.expanded_dim, stride, groups=self.expanded_dim),
            norm_layer(self.expanded_dim),
            nn.ReLU6(inplace=True),

            conv1x1(self.expanded_dim, planes),
            norm_layer(planes)
        )

    def forward(self, x):
        return self.sequential(x)


class SqueezeExcitationLayer(nn.Module, utils.Identifier):

    def __init__(self, channel, reduction=16):
        super(SqueezeExcitationLayer, self).__init__()
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


class SqueezeExcitationBlock(nn.Module, utils.Identifier):

    def __init__(self, inplanes, planes, norm_layer=nn.InstanceNorm2d, reduction=16):
        super(SqueezeExcitationBlock, self).__init__()

        self.sequential = nn.Sequential(
            ConvBlock(inplanes, planes, norm_layer),
            SqueezeExcitationLayer(planes, reduction)
        )

    def forward(self, x):
        return self.sequential(x)


class EfficientNetBlock(nn.Module, utils.Identifier):

    def __init__(self, inplanes, planes, stride=1, norm_layer=nn.InstanceNorm2d, reduction=16, expand_ratio=6):
        super(EfficientNetBlock, self).__init__()

        self.sequential = nn.Sequential(
            InvertedBlock(inplanes, planes, stride, norm_layer, expand_ratio=expand_ratio),
            SqueezeExcitationLayer(planes, reduction),
        )

    def forward(self, x):
        return self.sequential(x)

class ConvBlock(nn.Module, utils.Identifier):

    def __init__(self, inplanes, planes, norm_layer=None):
        super(ConvBlock, self).__init__()
        self.inplanes = inplanes
        self.planes = planes

        layers = []
        layers.append(nn.Conv2d(inplanes, planes, 3, padding=1, bias=norm_layer is not None))
        if norm_layer is not None:
            layers.append(norm_layer(planes))
        layers.append(nn.ReLU(True))

        layers.append(nn.Conv2d(planes, planes, 3, padding=1, bias=norm_layer is not None))
        if norm_layer is not None:
            layers.append(norm_layer(planes))
        layers.append(nn.ReLU(True))

        self.sequential = nn.Sequential(*layers)

    def forward(self, x):
        return self.sequential(x)

class UpConvBlock(nn.Module, utils.Identifier):

    def __init__(self, inplanes, planes):
        super(UpConvBlock, self).__init__()

        layers = []
        layers.append(nn.ConvTranspose2d(inplanes, planes, 2, 2, output_padding=1, bias=True))
        layers.append(nn.ReLU(True))

        self.sequential = nn.Sequential(*layers)

    def forward(self, x):
        return self.sequential(x)


class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        ]
        super(ASPPConv, self).__init__(*modules)


class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        self.in_channels = in_channels
        self.out_channels = out_channels
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())

    def forward(self, x):
        size = x.shape[-2:]
        for mod in self:
            x = mod(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)


class ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates, out_channels=256):
        super(ASPP, self).__init__()
        modules = []
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()))

        rates = tuple(atrous_rates)
        for rate in rates:
            modules.append(ASPPConv(in_channels, out_channels, rate))

        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)