import torch.nn as nn
from model import utils, blocks
import itertools
import functools
import torch
import numpy as np
import typing
from torchsummary import summary


class ResNetBackbone(nn.Module, utils.Identifier):

    def __init__(self, block, block_counts, inplanes, norm_layer=nn.InstanceNorm2d):
        super(ResNetBackbone, self).__init__()

        self._feature_count = len(block_counts)
        self._block_counts = block_counts
        self._feature_start_layer = 1
        self._depth = self._feature_start_layer + self._feature_count
        self._block = block
        self._norm_layer = norm_layer
        self._inplanes = inplanes
        self._first_layer = nn.Sequential(
            nn.Conv2d(3, self._inplanes, kernel_size=7,
                      stride=2, padding=3, bias=False),
            norm_layer(self._inplanes),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self._layers = nn.ModuleList([self._make_layer(block, int(
            i > 0)+1, layer_count, int(i > 0)+1) for i, layer_count in enumerate(block_counts)])

    @property
    def inplanes(self):
        return self._inplanes

    @property
    def block_counts(self):
        return self._block_counts

    def _make_layer(self, block, expansion, blocks, stride=1):
        outplanes = self._inplanes*expansion
        layers = []
        layers.append(block(self._inplanes, outplanes, stride))
        self._inplanes *= expansion
        for _ in range(1, blocks):
            layers.append(block(outplanes, outplanes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self._first_layer(x.transpose(3, 1))

        for l in self._layers:
            x = l(x)

        return x


class LinearResNetBackbone(nn.Module, utils.Identifier):

    def __init__(self, block, block_counts, inplanes, norm_layer=nn.BatchNorm1d):
        super(LinearResNetBackbone, self).__init__()

        self._feature_count = len(block_counts)
        self._block_counts = block_counts
        self._feature_start_layer = 1
        self._depth = self._feature_start_layer + self._feature_count
        self._block = block
        self._norm_layer = norm_layer
        self._inplanes = inplanes

        self._layers = nn.ModuleList([self._make_layer(block, int(
            i > 0)+1, layer_count) for i, layer_count in enumerate(block_counts)])

    @property
    def inplanes(self):
        return self._inplanes

    @property
    def block_counts(self):
        return self._block_counts

    def _make_layer(self, block, expansion, blocks):
        outplanes = self._inplanes*expansion
        layers = []
        layers.append(block(self._inplanes, outplanes))
        self._inplanes *= expansion
        for _ in range(1, blocks):
            layers.append(block(outplanes, outplanes))

        return nn.Sequential(*layers)

    def forward(self, x):
        outputs = [x]
        for l in self._layers:
            outputs.append(l(outputs[-1]))

        return outputs[-1]


class LinearNet(nn.Module, utils.Identifier):
    def __init__(self, adapter_network, backbone: utils.Identifier, output_size: int, input_size: list):
        super(LinearNet, self).__init__()
        self._output_size = output_size
        self._backbone = backbone
        self._first_layer = adapter_network
        input = torch.zeros(input_size).unsqueeze(0)
        in_size = self._first_layer(input).flatten().shape[0]
        self._inplanes = self._backbone.inplanes
        self._adapter = nn.Sequential(
            nn.Linear(in_size, self._inplanes),
            nn.ReLU()
        )
        self._head = nn.Linear(self._inplanes, self._output_size)

        summary(self.cuda(), torch.Size(input_size))

    @property
    def inplanes(self):
        return self._inplanes

    @property
    def block_counts(self):
        return [1]

    def forward(self, x):
        features = self._first_layer(x).flatten(1)
        features = self._adapter(features)
        features = self._backbone(features)

        return self._head(features)
