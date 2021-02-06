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
    def __init__(self, mode, directory, download = True):
        dataset_dirs = [os.path.join('/Data/captioning', directory), os.path.join('/Data1/captioning', directory)]
        # dataset_dirs = [os.path.join('/Data/captioning', directory)]
        self.download = download
        self.image_dirs = [os.path.join(d, mode, 'images') for d in dataset_dirs]
        
        for d in self.image_dirs:
            os.makedirs(d, exist_ok = True)

        for d in dataset_dirs:
            data_path = os.path.join(d, mode + '.tsv')    
            data = pd.read_csv(data_path, sep = '\t')
            self.data = data
            break

        self.data['index'] = self.data.index
        if not download:
            restricted_data_path = data_path.replace('.tsv', '_restricted.csv')
            if os.path.exists(restricted_data_path):
                self.data = pd.read_csv(restricted_data_path, sep = '\t')
            else:
                self.data = self.data[self.data['index'].map(lambda x: len(self.get_file_index(x))>0)]
                self.data.to_csv(restricted_data_path, sep = '\t', index = False)
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

    def get_file_index(self, idx):
        filenames = [os.path.join(d, str(idx)+'.png') for d in self.image_dirs]
        filename = ''
        for f in filenames:
            if os.path.exists(f):
                filename = f
        # try:
        #     image = Image.open(filename, mode='r')
        # except Exception as e:
        #     print(idx, e)
        #     if any([x in str(e) for x in ["cannot identify image file", "Decompressed Data Too Large"]]):
        #         os.remove(filename)

        return filename

    def __getitem__(self, index):
        label, image_url, idx = self.data.iloc[index]
        filenames = [os.path.join(d, str(idx)+'.png') for d in self.image_dirs]
        for filename in filenames:
            if os.path.exists(filename):
                break
        if self.download and not os.path.exists(filename):
            self.__download_image__(filename, image_url)
        try:
            image = Image.open(filename)
        except Exception as e:
            print(e)
            image = None
        
        if ':' in label:
            label = label.split(':')[-1]
        return image, label

