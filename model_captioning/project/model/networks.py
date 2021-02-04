import torch.nn as nn
from model import utils, blocks
import itertools
import functools
import torch
import torch.nn.functional as F
import torchvision.models as models

class LSTM(nn.Module, utils.Identifier):

    def __init__(self, inplanes, input_size, vectorizer):
        super(LSTM, self).__init__()
        self.inplanes = inplanes
        self.input_size = input_size
        self.vectorizer = vectorizer

        modules = list(models.resnet50().children())[:-1]
        self.backbone = nn.Sequential(*modules)
        self.depth = 5
        self.encoder_layer = nn.Linear(2048, self.inplanes)
        self.sequence_cell = nn.LSTMCell(self.inplanes, self.inplanes)
        self.word_encoder = nn.Embedding(len(self.vectorizer.vocab), self.inplanes)

        self.out_layer = nn.Linear(self.inplanes, len(self.vectorizer.vocab))

    def forward(self, x, state = None):
        if state:
            x = self.word_encoder(x)
        else:
            x = self.backbone(x)
            x = self.encoder_layer(x.flatten(1))
        state = self.sequence_cell(x, state)
        return self.out_layer(state[0]), state
           
    

class DeepLabV3Plus(nn.Module, utils.Identifier):
    def __init__(self, backbone, labels):
        super(DeepLabV3Plus, self).__init__()
        
        self.feature_count = 1
        self.backbone = backbone
        self.depth = self.backbone.depth
        self.feature_start_layer = 1
        self.inplanes = self.backbone.inplanes
        self.out_dim = len(labels)
        self.head = DeepLabHead(self.inplanes, 256)
        self.mid_layer = nn.Conv2d(self.inplanes//4, 256, kernel_size=1, bias=True)
        self.lateral_layers = nn.Sequential(nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                            nn.BatchNorm2d(256),
                                            nn.ReLU(),
                                            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                            nn.BatchNorm2d(256),
                                            nn.ReLU(),
                                            nn.Conv2d(256, self.out_dim, kernel_size=1, stride=1))


    def forward(self, x):
        features = self.backbone(x)
        bmid_out = features[-3]
        bmid_out = self.mid_layer(bmid_out)
        boutput = features[-1]
        head = self.head(boutput)
        m_out = F.interpolate(head, size=[x for x in bmid_out.size()[-2:]], mode='bilinear', align_corners=True)
        m_out = torch.cat((m_out, bmid_out), dim = 1)
        m_out = self.lateral_layers(m_out)
        output = F.interpolate(m_out, size=[x*4 for x in bmid_out.size()[-2:]], mode='bilinear', align_corners=True)
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