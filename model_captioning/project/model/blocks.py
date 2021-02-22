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


class ScaledDotProductAttentionBlock(nn.Module, utils.Identifier):
    def __init__(self, heads = 8):
        super(ScaledDotProductAttentionBlock, self).__init__()
        self.heads = heads

    def forward(self, query, key, value):
        qs = query.split(query.shape[-1] // self.heads, -1)
        ks = key.split(key.shape[-1] // self.heads, -1)
        vs = value.split(value.shape[-1] // self.heads, -1)
        
        scores = [(q.bmm(k.transpose(1,2)) / q.shape[-1]**0.5).softmax(-1).bmm(v) for q, k, v in zip(qs, ks, vs)]
        output = torch.cat(scores, -1)

        return output

class AoAttentionBlock(nn.Module, utils.Identifier):
    def __init__(self, inplanes):
        super(AoAttentionBlock, self).__init__()
        self.inplanes = inplanes
        self.input_attention = ScaledDotProductAttentionBlock()
        self.pixel_attention = nn.Sequential(nn.Linear(2 * self.inplanes, 2 * self.inplanes), nn.GLU())

    def forward(self, query, key, value):

        v = self.input_attention(query, key, value)
        x = torch.cat((v, query), -1)
        output = self.pixel_attention(x)

        return output

class RefiningBlock(nn.Module, utils.Identifier):
    def __init__(self, inplanes):
        super(RefiningBlock, self).__init__()
        self.inplanes = inplanes
        self.q_layer = nn.Linear(self.inplanes, self.inplanes)
        self.k_layer = nn.Linear(self.inplanes, self.inplanes)
        self.v_layer = nn.Linear(self.inplanes, self.inplanes)
        self.aoa = AoAttentionBlock(self.inplanes)  

    def forward(self, a):
        query = self.q_layer(a)
        key   = self.v_layer(a)
        value = self.k_layer(a)
        x = self.aoa(query, key, value)
        a = F.layer_norm(a + x, (self.inplanes, ))

        return a

class DecoderBlock(nn.Module, utils.Identifier):
    def __init__(self, inplanes, output_dim):
        super(DecoderBlock, self).__init__()
        self.inplanes = inplanes
        self.output_dim = output_dim
        self.dropout = nn.Dropout(p=0.5)
        self.word_encoder = nn.Embedding(self.output_dim, self.inplanes)
        self.out_layer = nn.Linear(self.inplanes, self.output_dim)
        self.sequence_cell = nn.LSTMCell(2 * self.inplanes, self.inplanes)

        self.aoa = AoAttentionBlock(inplanes)  

    def forward(self, a, l, c, state):
        af = a.mean((2, 3)) + c
        e = self.word_encoder(l)
        state = self.sequence_cell(torch.cat((af, e), -1), state)
        a = a.flatten(2).permute(0, 2, 1)
        c = self.aoa(state[0].unsqueeze(1), a, a).squeeze(1)
        out = self.out_layer(self.dropout(c))

        # split for summary
        return out, c, state[0], state[1]