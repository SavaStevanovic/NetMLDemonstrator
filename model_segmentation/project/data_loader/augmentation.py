import random                     
import torchvision.transforms as transforms
import numpy as np
from PIL import Image, ImageFilter
from skimage import util
import torch
import matplotlib.pyplot as plt
import cv2
import io
from skimage.draw import polygon
from itertools import compress
# from scipy.stats import multivariate_

class PairCompose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, label):
        data = [img, label]
        for t in self.transforms:
            data = t(*data)
        return tuple(data)

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string

class RandomCropTransform(object):
    def __init__(self, crop_size):
        self.crop_size = crop_size

    def __call__(self, image, label):
        padding_x = max(self.crop_size[0]-image.size[0],0)
        padding_y = max(self.crop_size[1]-image.size[1],0)
        image = transforms.functional.pad(image, (0, 0, padding_x, padding_y))
        label = transforms.functional.pad(label, (0, 0, padding_x, padding_y))
        x_radius = self.crop_size[0]//2
        y_radius = self.crop_size[1]//2
        x = random.randint(x_radius, image.size[0]-x_radius)
        y = random.randint(y_radius, image.size[1]-y_radius)
        start_point = [x-x_radius,y-y_radius]
        end_point = [x+x_radius, y+y_radius]
        image = image.crop((*start_point, *end_point))
        label = label.crop((*start_point, *end_point))
        
        return image, label

class RandomHorizontalFlipTransform(object):
    def __call__(self, image, label):
        p = random.random()
        if p>=0.5:
            image = transforms.functional.hflip(image)
            label = transforms.functional.hflip(label)

        return image, label

class RandomResizeTransform(object):
    def __call__(self, image, label):
        p = random.random()/2+0.5
        new_size = [np.round(s*p).astype(np.int32) for s in list(image.size)[::-1]]
        image = transforms.functional.resize(image, new_size, Image.ANTIALIAS)
        label = transforms.functional.resize(label, new_size, Image.NEAREST)
        return image, label

class RandomBlurTransform(object):
    def __init__(self):
        self.blur = ImageFilter.GaussianBlur(2)

    def __call__(self, image, label):
        p = random.random()
        if p>0.9:
            image = image.filter(self.blur)
        return image, label

class RandomColorJitterTransform(object):
    def __init__(self):
        self.jitt = transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
        )

    def __call__(self, image, label):
        p = random.random()
        if p>0.75:
            image = self.jitt(image)
        return image, label

class RandomNoiseTransform(object):
    def __init__(self):
        pass

    def __call__(self, image, label):
        p = random.random()
        if p>0.9:
            image = np.array(image)
            image = util.random_noise(image, mode='gaussian', seed=None, clip=True)*255
            image = Image.fromarray(image.astype(np.uint8))
        return image, label

class PaddTransform(object):
    def __init__(self, pad_size = 32):
        self.pad_size = pad_size

    def __call__(self, image, label):
        padding_x = self.pad_size-image.size[0]%self.pad_size
        padding_x = (padding_x!=self.pad_size) * padding_x
        padding_y = self.pad_size-image.size[1]%self.pad_size
        padding_y = (padding_y!=self.pad_size) * padding_y
        image = transforms.functional.pad(image, (0, 0, padding_x, padding_y))
        label = transforms.functional.pad(label, (0, 0, padding_x, padding_y))
        return image, label

class OutputTransform(object):
    def __call__(self, image, label):
        image = transforms.functional.to_tensor(image)
        if image.shape[0]==1:
            image = torch.cat([image]*3)
        label = transforms.functional.to_tensor(label)
        return image, label

class OneHotTransform(object):
    def __init__(self, cat_count):
        self.cat_transform = np.eye(cat_count)

    def __call__(self, image, label):
        label = np.array(label)
        label = self.cat_transform[label]
        return image, label

# added to reflect inference state
class RandomJPEGcompression(object):
    def __init__(self, quality):
        self.quality = quality

    def __call__(self, image, label):
        outputIoStream = io.BytesIO()
        image.save(outputIoStream, "JPEG", quality=self.quality, optimice=True)
        outputIoStream.seek(0)
        image = Image.open(outputIoStream)
        return image, label