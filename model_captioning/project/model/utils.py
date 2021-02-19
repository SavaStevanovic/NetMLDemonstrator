import re
from operator import itemgetter
import numpy as np

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
    def __init__(self, max_vocab_size, min_word_count = 6):
        self.min_word_count = 5
        self.words = {}
        self.max_vocab_size = max_vocab_size
        self.eos_token = '<eos>'
        self.sos_token = '<sos>'
        self.unk_token = '<unk>'
        self.pad_token = '<pad>'
        self.vec_numerizer = None

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
        self.vocab = np.array(tokens + [x[0] for x in vocab[:vocab_size] if x[1]>=self.min_word_count])

    def __call__(self, sentence):
        if self.vec_numerizer is None:
            numerizer = numerizer = lambda x: np.where(self.vocab == x)[0][0] if x in self.vocab else np.where(self.vocab == self.unk_token)[0][0]
            self.vec_numerizer = np.vectorize(numerizer)
        sentence = [self.sos_token] + self.__preprocess__(sentence) + [self.eos_token]
        sentence = self.vec_numerizer(sentence)

        return sentence

def get_acc(output, label):
    label_trim = label[label!=0]
    acc = (label_trim == output[:len(label_trim)]).float().sum() / len(label_trim)
    return acc.item()

def get_output_text(vectorizer, output):
    eos_index = np.where(vectorizer.vocab == vectorizer.eos_token)[0][0]
    if eos_index in output:
        output = output[:np.where(output == eos_index)[0][0]]
    output_text = ' '.join([vectorizer.vocab[x] for x in output])
    return output_text