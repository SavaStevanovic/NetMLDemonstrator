import torch.nn as nn
from model import utils
import torch.nn.functional as F
import torch


class AttentionBlock(nn.Module, utils.Identifier):
    def __init__(self, encoder_size, hidden_size, inplanes):
        super(AttentionBlock, self).__init__()
        self.encoder_size = encoder_size
        self.hidden_size = hidden_size
        self.inplanes = inplanes

        self.encoder_layer = nn.Conv2d(self.encoder_size, self.inplanes, 1)
        self.hidden_layer = nn.Linear(self.hidden_size, self.inplanes, 1)
        self.attention_layer = nn.Conv2d(self.inplanes, 1, 1)


    def forward(self, encoder_input, hidden_input):
        encod_out = self.encoder_layer(encoder_input)
        hidden_out = self.hidden_layer(hidden_input)
        attention = self.attention_layer(F.relu(encod_out + hidden_out.unsqueeze(-1).unsqueeze(-1))).flatten(1).softmax(1)
        output = (encoder_input.flatten(2) * attention.unsqueeze(1)).sum(dim = 2)

        return output, attention