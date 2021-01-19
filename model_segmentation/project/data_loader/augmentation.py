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

class PairCompose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, label, dataset_id):
        data = [img, label, dataset_id]
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

    def __call__(self, image, label, dataset_id):
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
        
        return image, label, dataset_id

class RandomHorizontalFlipTransform(object):
    def __call__(self, image, label, dataset_id):
        p = random.random()
        if p>=0.5:
            image = transforms.functional.hflip(image)
            label = transforms.functional.hflip(label)

        return image, label, dataset_id

class RandomResizeTransform(object):
    def __call__(self, image, label, dataset_id):
        p = random.random()/2+0.5
        new_size = [np.round(s*p).astype(np.int32) for s in list(image.size)[::-1]]
        image = transforms.functional.resize(image, new_size, Image.ANTIALIAS)
        label = transforms.functional.resize(label, new_size, Image.NEAREST)
        return image, label, dataset_id

class ResizeTransform(object):
    def __init__(self, max_size):
        self.max_size = max_size

    def __call__(self, image, label, dataset_id):
        if min(image.size)>self.max_size:
            image = transforms.functional.resize(image, self.max_size, Image.ANTIALIAS)
            label = transforms.functional.resize(label, self.max_size, Image.NEAREST)
        return image, label, dataset_id

class RandomBlurTransform(object):
    def __init__(self):
        self.blur = ImageFilter.GaussianBlur(2)

    def __call__(self, image, label, dataset_id):
        p = random.random()
        if p>0.9:
            image = image.filter(self.blur)
        return image, label, dataset_id

class RandomColorJitterTransform(object):
    def __init__(self):
        self.jitt = transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
        )

    def __call__(self, image, label, dataset_id):
        p = random.random()
        if p>0.75:
            image = self.jitt(image)
        return image, label, dataset_id

class RandomNoiseTransform(object):
    def __init__(self):
        pass

    def __call__(self, image, label, dataset_id):
        p = random.random()
        if p>0.9:
            image = np.array(image)
            image = util.random_noise(image, mode='gaussian', seed=None, clip=True)*255
            image = Image.fromarray(image.astype(np.uint8))
        return image, label, dataset_id

class PaddTransform(object):
    def __init__(self, pad_size = 32):
        self.pad_size = pad_size

    def __call__(self, image, label, dataset_id):
        padding_x = self.pad_size-image.size[0]%self.pad_size
        padding_x = (padding_x!=self.pad_size) * padding_x
        padding_y = self.pad_size-image.size[1]%self.pad_size
        padding_y = (padding_y!=self.pad_size) * padding_y
        image = transforms.functional.pad(image, (0, 0, padding_x, padding_y))
        label = transforms.functional.pad(label, (0, 0, padding_x, padding_y))
        return image, label, dataset_id

class OutputTransform(object):
    def __call__(self, image, label, dataset_id):
        image = transforms.functional.to_tensor(image)
        if image.shape[0]==1:
            image = torch.cat([image]*3)
        label = transforms.functional.to_tensor(label)
        return image, label, dataset_id

class OneHotTransform(object):
    def __init__(self, cat_count, selector):
        self.cat_transform = np.eye(cat_count+1)
        self.cat_count = cat_count
        self.selector_mapper = [{i+1:x+1 for i, x in enumerate(m)} for m in selector]
        for x in self.selector_mapper:
            x[0] = 0
            x[255] = cat_count

        
    def __call__(self, image, label, dataset_id):
        label = np.array(label)
        label = np.vectorize(self.selector_mapper[dataset_id].__getitem__)(label)
        label = self.cat_transform[label]
        return image, label, dataset_id

# added to reflect inference state
class JPEGcompression(object):
    def __init__(self, quality):
        self.quality = quality

    def __call__(self, image, label, dataset_id):
        outputIoStream = io.BytesIO()
        image.save(outputIoStream, "JPEG", quality=self.quality, optimice=True)
        outputIoStream.seek(0)
        image = Image.open(outputIoStream)
        return image, label, dataset_id