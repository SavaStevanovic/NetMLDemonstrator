from torch.utils.data import Dataset
from PIL import Image
import glob
import os


class ADEChallengeData2016(Dataset):
    def __init__(self, mode, folder_path):
        super(ADEChallengeData2016, self).__init__()
        img_files = glob.glob(os.path.join(
            '/Data/segmentation', folder_path, 'annotations', mode, '*.png'))
        self.data = [(x.replace('.png', '.jpg').replace(
            'annotations', 'images'), x) for x in img_files]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_path, segm_path = self.data[index]
        data = Image.open(img_path, mode='r')
        return data
