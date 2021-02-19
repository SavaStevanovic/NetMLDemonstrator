import torch.nn as nn
from model import utils, blocks
import itertools
import functools
import torch
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
from operator import itemgetter
from itertools import compress

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

        modules = list(models.resnet152(pretrained=True).children())[:-2]
        self.backbone = nn.Sequential(*modules)
        self.attention = blocks.AttentionBlock(2048, self.inplanes, self.inplanes)
        self.state_layer  = nn.Linear(2048, self.inplanes, 1)
        self.hidden_layer = nn.Linear(2048, self.inplanes, 1)
        self.beta_layer   = nn.Linear(self.inplanes, 2048, 1)

        self.sequence_cell = nn.LSTMCell(2048 + self.embed_layer, self.inplanes)

        self.word_encoder = nn.Embedding(len(self.vectorizer.vocab), self.embed_layer)

        self.out_layer = nn.Linear(self.inplanes, len(self.vectorizer.vocab))

    def grad_backbone(self, enable):
        for param in self.backbone.parameters():
            param.requires_grad = enable

    def forward(self, images, labels):
        #for summary
        if labels.type()=='torch.cuda.FloatTensor':
            labels = torch.ones((len(images), 1)).long().cuda()

        img_encoded = self.backbone(images)
        outputs    = []
        attentions = []
        image_enc_mean = img_encoded.mean((2, 3))
        state = self.hidden_layer(image_enc_mean), self.state_layer(image_enc_mean)
        for j in range(labels.shape[1]):
            d = labels[:, j] 
            word_enc = self.word_encoder(d)
            beta = self.beta_layer(state[0]).sigmoid()
            att_output, att = self.attention(img_encoded, state[0])
            attention = beta * att_output
            state = self.sequence_cell(torch.cat((word_enc, attention), 1), state)
            output = self.out_layer(self.dropout(state[0]))
            outputs.append(output)
            attentions.append(att)

        return torch.stack(outputs, -1), torch.stack(attentions, -1)

    def forward_single_inference(self, images, beam_size):
        outs = [(torch.tensor([]), 0) for _ in range(len(images))]
        cur_indexes = list(range(len(images)))
        img_encoded = self.backbone(images)
        outputs   = torch.zeros((len(images), beam_size, 0), dtype=torch.long, device='cuda')
        sos_index = np.where(self.vectorizer.vocab == self.vectorizer.sos_token)[0][0]
        eos_index = np.where(self.vectorizer.vocab == self.vectorizer.eos_token)[0][0]
        labels = torch.tensor([[sos_index for _ in range(1)] for _ in range(len(images))], dtype=torch.long, device='cuda')
        scores = torch.zeros_like(labels, dtype=torch.float, device='cuda')
        image_enc_mean = img_encoded.mean((2, 3))
        state = self.hidden_layer(image_enc_mean), self.state_layer(image_enc_mean)
        state = list(zip(*[[s for s in state] for _ in range(1)]))
        state = [torch.stack(s).permute(1,0,2) for s in state]
        for j in range(50):
            word_enc = self.word_encoder(labels)
            beta = self.beta_layer(state[0]).sigmoid()
            att_output, att = list(zip(*[self.attention(img_encoded, state[0][:, p]) for p in range(state[0].shape[1])]))
            att_output = torch.stack(att_output).permute(1,0,2)
            attention = beta * att_output
            state = list(zip(*[self.sequence_cell(torch.cat((word_enc[:, p], attention[:, p]), -1), [s[:, p] for s in state]) for p in range(state[0].shape[1])]))
            state = [torch.stack(s).permute(1,0,2) for s in state]
            output = self.out_layer(self.dropout(state[0])) 
            ts, ti = (scores.unsqueeze(-1) + output.softmax(-1)).flatten(1).topk(beam_size)
            labels =  ti.remainder(len(self.vectorizer.vocab))
            indx = ti // (len(self.vectorizer.vocab))
            state = [torch.stack([s[p, idx] for p, idx in enumerate(indx)]) for s in state]
            outputs = torch.stack([outputs[p, idx] for p, idx in enumerate(indx)])
            scores = ts
            outputs = torch.cat((outputs, labels.unsqueeze(-1)), -1)

            finished = (labels == eos_index).nonzero(as_tuple = False)
            for f in finished:
                if outs[cur_indexes[f[0]]][1] != 0 and ts[f[0], f[1]]>0:
                    oo=0
                if outs[cur_indexes[f[0]]][1] == 0:
                    outs[cur_indexes[f[0]]] = (outputs[f[0], f[1]], ts[f[0], f[1]].item())
                    ts[f[0], f[1]] = -100

            # indeces = [outs[p][1] == 0 for p in cur_indexes]
            # outputs = outputs[indeces]
            # images = images[indeces]
            # scores = scores[indeces]
            # labels = labels[indeces]
            # img_encoded = img_encoded[indeces]
            # state = [s[indeces] for s in state]
            # cur_indexes = list(compress(cur_indexes, indeces))

            if sum([outs[p][1] == 0 for p in cur_indexes])==0:
                break
            
        return [o[0].detach().cpu().numpy() for o in outs]