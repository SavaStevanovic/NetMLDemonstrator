import os 
import numpy as np
from sklearn.model_selection import train_test_split
from data_loader.iterator import StringDatasetTrain, StringDatasetValidation
from torchvision import transforms
from data_loader.data_parser import getNamingDictFromFile, getUniversalNames
from data_loader.augmentation import RandomShuffle, SequencePadder, StringVectorizer, RandomStarPlace, RandomCharDelete, RandomWordShuffle

class DatasetCreator:
    def __init__(self, root_dir, names_file):
        self.universal_names = getUniversalNames(names_file)
        self.train_data, self.leagues  = self.parse_data(os.path.join(root_dir, 'train_data'))
        self.validation_data, _ = self.parse_data(os.path.join(root_dir, 'validation_data'))
        self.corpus, self.ocurences = self.get_corpus(list(self.train_data.values()))

    def get_corpus(self, data):
        corp_list = []
        for x in data:
            corp_list += x
        joined_text = "".join(corp_list)
        corpus = set(joined_text)
        ocurences = {x:joined_text.count(x) for x in corpus}
        return corpus, ocurences

    def parse_data(self, root_dir):
        mapping_dict = {}
        league_dict = {}
        for x in os.listdir(root_dir):
            getNamingDictFromFile(os.path.join(root_dir, x), self.universal_names, mapping_dict, league_dict) 
        for x in mapping_dict:
            mapping_dict[x] = list(set(mapping_dict[x]))

        for x in league_dict:
            league_dict[x] = list(set(league_dict[x]))

        return mapping_dict, league_dict

    def get_train_iterator(self, transform=None):
        if transform is None:
            transform = transforms.Compose([
                # RandomShuffle(),
                RandomWordShuffle(),
                RandomCharDelete(),
                RandomStarPlace(),
                StringVectorizer(self.corpus),
                SequencePadder(35, self.corpus),

            ])
        return StringDatasetTrain(self.train_data, transform)

    def get_validation_iterator(self, transform=None):        
        if transform is None:
            transform = transforms.Compose([
                StringVectorizer(self.corpus),
                SequencePadder(35, self.corpus),

            ])
        return StringDatasetValidation(self.validation_data, self.leagues, transform) 