import torch
from data_loader.conceptual_dataset import ConceptualDataset
from data_loader.sbu_dataset import SBUDataset
import torchvision.transforms as transforms
from data_loader import augmentation
import multiprocessing as mu
from torch.utils.data import Dataset, DataLoader
from pycocotools.coco import COCO
import os
import random
import io 
import pickle
from model.utils import WordVocabulary
import warnings
import numpy as np
from PIL import Image


class UnifiedKeypointDataset(Dataset):
    def __init__(self, train, debug=False):
        self.debug = debug
        self.train = train
        train_datasets = [
                SBUDataset(False),
                ConceptualDataset('train', 'google', False),
            ]

        if train:
            self.datasets = train_datasets
            
        if not train:
            self.datasets = [
                ConceptualDataset('val', 'google', False),
            ]
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        if train:
            self.transforms = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                augmentation.OutputTransform(),
                normalize,
                ]
            )

        if not train:
            self.transforms = transforms.Compose([
                transforms.Resize((224, 224)),
                augmentation.OutputTransform(),
                normalize,
                ]
            )

        vectorizer_path = 'vectorizer.p'
        if os.path.exists(vectorizer_path):
            self.vectorizer = pickle.load(open(vectorizer_path, "rb"))
        else:
            self.vectorizer = WordVocabulary(32768)
            texts = np.concatenate([x.get_vocab_list() for x in self.datasets])
            self.vectorizer.build_vocab(texts)
            with open(vectorizer_path, 'wb') as handle:
                pickle.dump(self.vectorizer, handle)

        if self.debug==1:
            self.data_ids = [(i, j) for i, dataset in enumerate(self.datasets) for j in range(50)]
        else:
            self.data_ids = [(i, j) for i, dataset in enumerate(self.datasets) for j in range(len(self.datasets[i]))]
        
        self.data_ids = np.array(self.data_ids)

    def __len__(self):
        return len(self.data_ids)

    def __getitem__(self, idx):
        warnings.filterwarnings("error")
        identifier = self.data_ids[idx]
        image, label = self.datasets[identifier[0]][identifier[1]]
        if image.mode not in ("RGB", "L", "RGBA", "P", "1", "LA"):
            print(image.mode)
        if image.mode=='LA':
            image = Image.fromarray(np.array(image)[..., 0])
        if image.mode=='P':
            image = image.convert('RGBA')
        if image.mode=='RGBA':
            image = Image.fromarray(np.array(image)[..., :3])
        if any([m == image.mode for m in ("RGB", "1", "P")]):
            image = image.convert('RGB')

        if self.transforms:
            image = self.transforms(image)

        label_tokenized = torch.tensor(self.vectorizer(label), dtype=torch.long)
        return image, label_tokenized