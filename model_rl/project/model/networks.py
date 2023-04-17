from tensorboard import summary
import torch.nn as nn
import torch

from model.identifier import Identifier


class LinearNet(nn.Module, Identifier):
    def __init__(self, input_size: list):
        Identifier.__init__(self, len(input_size), sum(input_size, 0))
        nn.Module.__init__(self)

        layers = [
            nn.Linear(input_size[i], input_size[i+1]) for i in range(len(input_size)-1) 
        ]
        layers = [layer for elem in layers for layer in (elem, nn.ReLU())][:-1]
        self._layers = nn.Sequential(*layers)

    def forward(self, x):
        return self._layers(x)