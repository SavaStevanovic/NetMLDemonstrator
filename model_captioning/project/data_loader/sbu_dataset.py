from torchvision import datasets
from torch.utils.data import Dataset, DataLoader
from PIL import Image 
import os
import numpy as np
from tqdm import tqdm

class SBUDataset(Dataset):
    def __init__(self, clean = False):
        dataset_dir = '/Data2/SBUCaptionedPhotoDataset/dataset'
        captions_file = open(os.path.join(dataset_dir, 'SBU_captioned_photo_dataset_captions.txt'), 'r')
        self.captions = np.array([x.replace('\n', '') for x in captions_file.readlines()])
        images_dir = os.path.join(dataset_dir, 'images')
        self.images_paths = [os.path.join(images_dir, img_path) for img_path in os.listdir(images_dir)]
        if clean:
            for filename in tqdm(self.images_paths):
                try:
                    image = Image.open(filename, mode='r')
                except Exception as e:
                    print(filename, e)
                    if any([x in str(e) for x in ["cannot identify image file", "Decompressed Data Too Large"]]):
                        os.remove(filename)
            self.images_paths = np.array([os.path.join(images_dir, img_path) for img_path in os.listdir(images_dir)])
    
    def get_vocab_list(self):
        return self.captions

    def __len__(self):
        return len(self.images_paths)

    def __getitem__(self, idx):
        image_path = self.images_paths[idx]
        image = Image.open(image_path)
        label_index = int(image_path.split('.png')[0].split('/')[-1])
        label = self.captions[label_index]
        return image, label
