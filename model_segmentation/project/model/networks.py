import torch.nn as nn
from model import utils, blocks
import itertools
import functools
import torch
import torch.nn.functional as F


class Unet(nn.Module, utils.Identifier):

    def __init__(self, block, inplanes, in_dim, out_dim, depth=4, norm_layer=None):
        super(Unet, self).__init__()

        self.depth = depth
        self.block = block
        self._norm_layer = norm_layer
        self.inplanes = inplanes
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.first_layer = block(in_dim, inplanes, norm_layer=norm_layer)
        self.down_layers = nn.ModuleList([self._make_down_layer(block, inplanes*2**i) for i in range(depth)])
        self.downsample = nn.MaxPool2d(2, 2)
        mid_planes = inplanes*2**depth
        self.mid_layer = self._make_mid_layer(block, mid_planes)
        self.up_layers = nn.ModuleList([self._make_up_layer(block, inplanes*2**i) for i in range(depth)])
        self.lateral_layers = nn.ModuleList([self._make_lateral_layer(block, inplanes*2**i) for i in range(depth)])
        self.out_layer = nn.Conv2d(inplanes, out_dim, 1)
           
    def _make_down_layer(self, block, planes):
        return block(planes, planes*2, norm_layer=self._norm_layer)

    def _make_mid_layer(self, block, planes):
        return block(planes, planes, norm_layer=self._norm_layer)

    def _make_up_layer(self, block, planes):
        return nn.ConvTranspose2d(planes*2, planes, 2, 2)
    
    def _make_lateral_layer(self, block, planes):
        return block(planes*2, planes, norm_layer=self._norm_layer)

    def forward(self, x):
        x = self.first_layer(x)
        
        outputs = [x]
        for l in self.down_layers:
            x = self.downsample(x)
            x = l(x)
            outputs.append(x)
    
        x = self.mid_layer(x)

        for i in range(self.depth-1, -1, -1):
            x = self.up_layers[i](x)
            x = torch.cat((x, outputs[i]), 1)
            x = self.lateral_layers[i](x)

        x = self.out_layer(x) 
        return x


class ResNetBackbone(nn.Module, utils.Identifier):

    def __init__(self, block, block_counts, inplanes, norm_layer=nn.InstanceNorm2d):
        super(ResNetBackbone, self).__init__()

        self.feature_count = len(block_counts)
        self.block_counts = block_counts
        self.feature_start_layer = 2
        self.depth = self.feature_start_layer + self.feature_count
        self.block = block
        self._norm_layer = norm_layer
        self.inplanes = inplanes
        self.first_layer = nn.Sequential(
            nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False),
            self._norm_layer(self.inplanes),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2)
        )

        self.layers = nn.ModuleList([self._make_layer(block, int(i>0)+1, layer_count, int(i>0)+1) for i, layer_count in enumerate(block_counts)])
           
    def _make_layer(self, block, expansion, blocks, stride=1):
        downsample = None
        outplanes = self.inplanes*expansion
        if stride != 1 or expansion != 1:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, outplanes, kernel_size = 1, stride=stride, bias=False), 
                nn.BatchNorm2d(outplanes)
            )
        layers = []
        layers.append(block(self.inplanes, outplanes, stride, downsample=downsample, norm_layer=self._norm_layer))
        self.inplanes *= expansion
        for _ in range(1, blocks):
            layers.append(block(outplanes, outplanes, norm_layer=self._norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.first_layer(x)

        outputs = [x]
        for l in self.layers:
            outputs.append(l(outputs[-1]))

        return tuple(outputs[1:])


class DeepLabV3Plus(nn.Module, utils.Identifier):
    def __init__(self, backbone, classes):
        super(DeepLabV3Plus, self).__init__()
        
        self.feature_count = 1
        self.backbone = backbone
        self.depth = self.backbone.depth
        self.feature_start_layer = 1
        self.inplanes = self.backbone.inplanes
        self.classes = classes
        self.head = DeepLabHead(self.inplanes, 256)
        self.mid_layer = nn.Conv2d(self.inplanes//4, 256, kernel_size=1, bias=False)
        self.lateral_layers = nn.Sequential(nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                            nn.BatchNorm2d(256),
                                            nn.ReLU(),
                                            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                            nn.BatchNorm2d(256),
                                            nn.ReLU(),
                                            nn.Conv2d(256, classes, kernel_size=1, stride=1))


    def forward(self, x):
        features = self.backbone(x)
        bmid_out = features[-3]
        bmid_out = self.mid_layer(bmid_out)
        boutput = features[-1]
        head = self.head(boutput)
        m_out = F.upsample(head, size=[x for x in bmid_out.size()[-2:]], mode='bilinear', align_corners=True)
        m_out = torch.cat((m_out, bmid_out), dim = 1)
        m_out = self.lateral_layers(m_out)
        output = F.upsample(m_out, size=[x*4 for x in bmid_out.size()[-2:]], mode='bilinear', align_corners=True)
        return output


class DeepLabHead(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(DeepLabHead, self).__init__(
            blocks.ASPP(in_channels, [6, 12, 18]),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, out_channels, 1)
        )