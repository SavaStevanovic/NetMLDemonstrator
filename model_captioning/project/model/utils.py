import re
from operator import itemgetter

class Identifier(object):
    def __init__(self):
        pass

    def get_identifier(self):
        idt = self.__class__.__name__
        if hasattr(self, 'inplanes'): 
            idt+='/'+str(self.inplanes)
        if hasattr(self, 'paf_stages'): 
            idt+='/'+str(self.paf_stages)
        if hasattr(self, 'map_stages'): 
            idt+='/'+str(self.map_stages)
        if hasattr(self, 'paf_planes'): 
            idt+='/'+str(self.paf_planes)
        if hasattr(self, 'map_planes'): 
            idt+='/'+str(self.map_planes)
        if hasattr(self, 'block_counts'): 
            idt+='/'+'-'.join([str(x) for x in self.block_counts])
        if hasattr(self, 'ratios'): 
            idt+='/'+'-'.join([str(x).replace('.',',') for x in self.ratios])
        if hasattr(self, 'block') and hasattr(self.block, 'get_identifier'): 
            idt+='/'+self.block.__name__
        if hasattr(self, 'backbone') and hasattr(self.backbone, 'get_identifier'): 
            idt+='/'+self.backbone.get_identifier() 
        return idt

class WordVocabulary(object):
    def __init__(self, max_vocab_size):
        self.words = {}
        self.max_vocab_size = max_vocab_size
        self.eos_token = '<eos>'
        self.sos_token = '<sos>'
        self.unk_token = '<unk>'
        self.pad_token = '<pad>'

    def __preprocess__(self, sentence):
        sentence = sentence.lower()
        sentence = re.sub(r'[^a-zA-Z0-9_\s]+', '', sentence)
        sentence = sentence.split()

        return sentence

    def build_vocab(self, sentences):
        for sentence in sentences:
            sentence = self.__preprocess__(sentence)
            for w in sentence:
                if w not in self.words:
                    self.words[w] = 0
                self.words[w] += 1

        vocab = sorted(list(self.words.items()), key=itemgetter(1), reverse=True)
        tokens = [self.pad_token, self.sos_token, self.eos_token, self.unk_token]
        vocab_size = min(self.max_vocab_size, len(vocab)) - len(tokens)
        self.vocab = tokens + [x[0] for x in vocab[:vocab_size]]

    def __call__(self, sentence):
        sentence = [self.sos_token] + self.__preprocess__(sentence) + [self.eos_token]
        sentence = [self.vocab.index(x) if x in self.vocab else self.vocab.index(self.unk_token) for x in sentence]

        return sentence