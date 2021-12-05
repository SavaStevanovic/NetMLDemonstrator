import torch.nn as nn
from model import utils, blocks
import itertools
import functools
import torch
import numpy as np
import typing


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

        self._layers = nn.ModuleList([self.__make_layer(block, int(
            i > 0)+1, layer_count, int(i > 0)+1) for i, layer_count in enumerate(block_counts)])

    def _make_layer(self, block, expansion, blocks, stride=1):
        outplanes = self._inplanes*expansion
        layers = []
        layers.append(block(self._inplanes, outplanes, stride))
        self._inplanes *= expansion
        for _ in range(1, blocks):
            layers.append(block(outplanes, outplanes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self._first_layer(x)

        outputs = [x]
        for l in self._layers:
            outputs.append(l(outputs[-1]))

        return tuple(outputs[1:])


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

        return tuple(outputs[1:])


class LinearNet(nn.Module, utils.Identifier):
    def __init__(self, backbone: utils.Identifier, input_size: int, output_size: int):
        super(LinearNet, self).__init__()

        self._output_size = output_size
        self._backbone = functools.reduce(lambda b, m: m(b), backbone[::-1])
        self._inplanes = self._backbone.inplanes
        self._first_layer = nn.Sequential(
            nn.Linear(input_size, self._inplanes),
            nn.ReLU(inplace=True)
        )
        self._head = nn.Linear(self._inplanes, self._output_size)

    @property
    def inplanes(self):
        return self._inplanes

    @property
    def block_counts(self):
        return [1]

    def forward(self, x):
        x = self._first_layer(x)
        features = self._backbone(x)[-1]

        return self._head(features)
