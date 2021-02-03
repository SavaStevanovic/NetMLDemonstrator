from torchvision import datasets
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
import pandas as pd
import requests
import shutil 
from PIL import Image 
import torchvision.transforms as transforms

class ConceptualDataset(Dataset):
    def __init__(self, mode, directory):
        dataset_dir = os.path.join('/Data/captioning', directory)
        self.image_dir = os.path.join(dataset_dir, 'images')
        data_path = os.path.join(dataset_dir, mode + '.tsv')
        os.makedirs(self.image_dir, exist_ok = True)
        data = pd.read_csv(data_path, sep = '\t')
        self.data = data
        self.vocab_list = self.data.iloc[:, 0].values.tolist()

    def __len__(self):
        return len(self.data)

    def __download_image__(self, filename, image_url):
        try:
            im = Image.open(requests.get(image_url, stream=True, timeout=2.5).raw)
            im = transforms.functional.resize(im, 256, Image.ANTIALIAS)
            im.save(filename)
        except Exception as e:
            pass

    def __getitem__(self, idx):
        label, image_url = self.data.iloc[idx]
        filename = os.path.join(self.image_dir, str(idx)+'.png')
        if not os.path.exists(filename):
            self.__download_image__(filename, image_url)
        try:
            image = Image.open(filename, mode='r')
            image.convert("RGB")
        except Exception as e:
            # print(e)
            image = None
        return image, label

