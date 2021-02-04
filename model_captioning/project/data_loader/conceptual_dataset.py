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
        dataset_dirs = [os.path.join('/Data/captioning', directory), os.path.join('/Data1/captioning', directory)]
        self.image_dirs = [os.path.join(d, mode, 'images') for d in dataset_dirs]
        
        for d in self.image_dirs:
            os.makedirs(d, exist_ok = True)

        for d in dataset_dirs:
            data_path = os.path.join(d, mode + '.tsv')    
            data = pd.read_csv(data_path, sep = '\t')
            self.data = data
            break

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
        filenames = [os.path.join(d, str(idx)+'.png') for d in self.image_dirs]
        file_index = np.where([os.path.exists(f) for f in filenames])[0]
        filename = filenames[-1]
        if len(file_index):
            filename = filenames[file_index[0]]
        if not os.path.exists(filename):
            self.__download_image__(filename, image_url)
        try:
            image = Image.open(filename, mode='r')
            image.convert("RGB")
        except Exception as e:
            # print(e)
            image = None
        return image, label

