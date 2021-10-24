import torch.nn as nn
from model import utils
import torch.nn.functional as F

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class BasicLinearLayer(nn.Sequential):
    def __init__(self, inplanes, planes):
        modules = [
            nn.Linear(inplanes, planes, bias=True),
            # nn.BatchNorm1d(planes),
            nn.ReLU(),
        ]
        super(BasicLinearLayer, self).__init__(*modules)

class BasicLinearBlock(nn.Module):

    def __init__(self, inplanes, planes, downsample=None):
        super(BasicLinearBlock, self).__init__()
        self.planes = planes
        self.inplanes = inplanes
        self.sequential = BasicLinearLayer(inplanes, planes)

        self.downsample = None
        if inplanes != planes:
            self.downsample = nn.Sequential(
                nn.Linear(inplanes, planes, bias=True),
                # nn.BatchNorm1d(planes)
            )

    def forward(self, x):
        residual = x

        out = self.sequential(x)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = F.relu(out, True)

        return out

class BasicBlock(nn.Module):

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.planes = planes
        self.inplanes = inplanes
        self.stride = stride
        self.sequential = BasicLayer(inplanes, planes, stride)

        self.downsample = None
        if stride != 1 or inplanes != planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False),
                nn.InstanceNorm2d(planes)
            )

    def forward(self, x):
        residual = x

        out = self.sequential(x)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = F.relu(out, True)

        return out

class PreActivationBlock(nn.Module, utils.Identifier):

    def __init__(self, inplanes, planes, stride=1):
        super(PreActivationBlock, self).__init__()
        self.planes = planes
        self.inplanes = inplanes
        self.stride = stride

        self.sequential = PreActivationLayer(inplanes, planes, stride)

        self.downsample = None
        if stride != 1 or inplanes != planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        residual = x

        out = self.sequential(x)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        return out


class InvertedBlock(nn.Module, utils.Identifier):

    def __init__(self, inplanes, planes, stride=1, expand_ratio=6):
        super(InvertedBlock, self).__init__()
        self.expanded_dim = inplanes * expand_ratio
        self.stride = stride
        self.inplanes = inplanes
        self.planes = planes

        self.sequential = nn.Sequential(
            nn.Conv2d(inplanes, self.expanded_dim, kernel_size=1, bias=False),
            nn.InstanceNorm2d(self.expanded_dim),
            nn.ReLU6(inplace=True),

            nn.Conv2d(self.expanded_dim, self.expanded_dim, kernel_size=3, stride=stride, padding=1, bias=False, groups=self.expanded_dim),
            nn.InstanceNorm2d(self.expanded_dim),
            nn.ReLU6(inplace=True),

            nn.Conv2d(self.expanded_dim, planes, kernel_size=1, bias=False),
            nn.InstanceNorm2d(planes)
        )

    def forward(self, x):
        if self.stride != 1 or self.inplanes != self.planes:
            return self.sequential(x)
        return x + self.sequential(x)


class BasicLayer(nn.Sequential):
    def __init__(self, inplanes, planes, stride):
        modules = [
            nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.InstanceNorm2d(planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(planes)
        ]
        super(BasicLayer, self).__init__(*modules)


class PreActivationLayer(nn.Sequential):
    def __init__(self, inplanes, planes, stride):
        modules = [
            nn.InstanceNorm2d(inplanes),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.InstanceNorm2d(planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, planes, kernel_size=3, padding=1, bias=False)
        ]
        super(PreActivationLayer, self).__init__(*modules)


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

    def __init__(self, inplanes, planes, stride=1, norm_layer=nn.InstanceNorm2d, reduction=16):
        super(SqueezeExcitationBlock, self).__init__()
        
        self.sequential = BasicLayer(inplanes, planes, stride)
        
        self.se_layer = SqueezeExcitationLayer(planes, reduction)

        self.downsample = None
        if stride != 1 or inplanes != planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False),
                nn.InstanceNorm2d(planes),
            )

    def forward(self, x):
        residual = x
        res_out = self.sequential(x)
        out  = self.se_layer(res_out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        return out


class EfficientNetBlock(nn.Module, utils.Identifier):

    def __init__(self, inplanes, planes, stride=1, norm_layer=nn.InstanceNorm2d, reduction=16, expand_ratio=6):
        super(EfficientNetBlock, self).__init__()
        self.expanded_dim = inplanes * expand_ratio
        self.sequential = nn.Sequential(
            nn.Conv2d(inplanes, self.expanded_dim, kernel_size=1, bias=False),
            nn.InstanceNorm2d(self.expanded_dim),
            nn.ReLU6(inplace=True),

            nn.Conv2d(self.expanded_dim, self.expanded_dim, kernel_size=3, stride=stride, padding=1, bias=False, groups=self.expanded_dim),
            nn.InstanceNorm2d(self.expanded_dim),
            nn.ReLU6(inplace=True),

            nn.Conv2d(self.expanded_dim, planes, kernel_size=1, bias=False),
            nn.InstanceNorm2d(planes)
        )
        
        self.se_layer = SqueezeExcitationLayer(planes, reduction)

        self.downsample = None
        if stride != 1 or inplanes != planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False),
                nn.InstanceNorm2d(planes),
            )

    def forward(self, x):
        residual = x
        res_out = self.sequential(x)
        out  = self.se_layer(res_out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        return out

