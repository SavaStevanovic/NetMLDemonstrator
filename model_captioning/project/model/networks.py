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
        self.word_encoder = nn.Embedding(len(self.vectorizer.vocab), 300)
        self.word_compresser = nn.Linear(300, self.inplanes)

        self.out_layer = nn.Linear(self.inplanes, len(self.vectorizer.vocab))

    def grad_backbone(self, freeze):
        for param in self.backbone.parameters():
            param.requires_grad = freeze

    def forward(self, encoder_input, x, state = None):
        state = None
        outputs = torch.zeros((labels.shape[0], len(net.vectorizer.vocab), labels.shape[1]), device = 'cuda')
        d = image.cuda()
        start = 0
        init_axis = 0
        for l in label_lens:
            end = l[0]
            for j in range(start, end):
                output, state = net(d, state)
                outputs[-len(output):, :, j] = output
                d = labels_cuda[init_axis:, j] 
            state = (state[0][l[1]:], state[1][l[1]:])
            init_axis += l[1]
            d = labels_cuda[init_axis:, j]
            start = end

        # if state:
        #     x = self.word_encoder(x)
        #     x = self.word_compresser(x)
        # else:    
        #     x = self.encoder_layer(encoder_input.flatten(1))
        # state = self.sequence_cell(x, state)

        # return self.out_layer(state[0]), state
           

class AttLSTM(nn.Module, utils.Identifier):

    def __init__(self, inplanes, input_size, vectorizer):
        super(AttLSTM, self).__init__()
        self.inplanes = inplanes
        self.input_size = input_size
        self.vectorizer = vectorizer
        self.embed_layer = 300
        self.depth = 5

        self.dropout = nn.Dropout(p=0.5)

        modules = list(models.resnet50().children())[:-2]
        self.backbone = nn.Sequential(*modules)

        self.attention = blocks.AttentionBlock(2048, self.inplanes, self.inplanes)
        self.state_layer  = nn.Linear(2048, self.inplanes, 1)
        self.hidden_layer = nn.Linear(2048, self.inplanes, 1)
        self.beta_layer   = nn.Linear(self.inplanes, 2048, 1)

        self.sequence_cell = nn.LSTMCell(2048 + self.embed_layer, self.inplanes)

        self.word_encoder = nn.Embedding(len(self.vectorizer.vocab), self.embed_layer)

        self.out_layer = nn.Linear(self.inplanes, len(self.vectorizer.vocab))

    def grad_backbone(self, freeze):
        for param in self.backbone.parameters():
            param.requires_grad = freeze

    def forward(self, images, labels, label_lens = None):
        #for summary
        if label_lens is None:
            label_lens = [(1, 1)]
            labels = labels.long()
        start = 0
        init_axis = 0
        img_encoded = self.backbone(images)
        outputs    = torch.zeros((labels.shape[0], len(self.vectorizer.vocab)                 , labels.shape[1]+1), device = 'cuda')
        attentions = torch.zeros((labels.shape[0], img_encoded.shape[2] * img_encoded.shape[3], labels.shape[1]+1), device = 'cuda')
        image_enc_mean = img_encoded.mean((2, 3))
        state = self.hidden_layer(image_enc_mean), self.state_layer(image_enc_mean)
        # word_enc = torch.zeros((labels.shape[0], self.embed_layer), device = 'cuda')
        for l in label_lens:
            end = l[0]
            for j in range(start, end):
                d = labels[init_axis:, j] 
                word_enc = self.word_encoder(d)
                # word_enc = word_enc + word_enc_c
                beta = self.beta_layer(state[0]).sigmoid()
                att_output, att = self.attention(img_encoded, state[0])
                attention = beta * att_output
                state = self.sequence_cell(torch.cat((word_enc, attention), 1), state)
                # state = state[0] + state_c[0], state[1] + state_c[1]
                output = self.out_layer(self.dropout(state[0]))
                # output, state = net(d, state)
                outputs[-len(output):, :, j+1]    = output
                attentions[-len(output):, :, j+1] = att
            state = (state[0][l[1]:], state[1][l[1]:])
            word_enc = word_enc[l[1]:]
            img_encoded = img_encoded[l[1]:]
            init_axis += l[1]
            start = end

        return outputs[..., :-1], attentions[..., :-1]