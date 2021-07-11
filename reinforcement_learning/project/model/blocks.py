from torch import nn
from model import utils
import torch.nn.functional as F
import torch

class PoseCNNStage(nn.Sequential, utils.Identifier):
    def __init__(self, inplanes, planes, outplanes, block_count):
        self.inplanes = inplanes
        self.planes = planes
        self.outplanes = outplanes
        self.block_count = block_count
        layers = []
        layers.append(PoseConvBlock(inplanes, planes, 3))
        layers.extend([PoseConvBlock(planes, planes, 3) for _ in range(block_count-1)])
        layers.append(nn.Conv2d(planes, planes, kernel_size=1, bias=True))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(planes, outplanes, kernel_size=1, bias=True))

        super(PoseCNNStage, self).__init__(*layers)


class PoseConvBlock(nn.Module, utils.Identifier):
    def __init__(self, inplanes, planes, layer_count):
        super(PoseConvBlock, self).__init__()
        self.inplanes = inplanes
        self.planes = planes
        self.layer_count = layer_count
        layers = []
        layers.append(nn.Sequential(nn.Conv2d(inplanes, planes, kernel_size=3, bias=True, padding=1), nn.ReLU(inplace=True)))
        layers.extend([nn.Sequential(nn.Conv2d(planes, planes, kernel_size=3, bias=True, padding=1), nn.ReLU(inplace=True)) for _ in range(layer_count-1)])
        
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        # out = x
        # l = self.layers[0]
        x = self.layers[0](x)
        outputs = x
        for l in self.layers[1:]:
            x = l(x)
            outputs = outputs + x
        return outputs
