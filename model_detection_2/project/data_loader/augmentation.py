import random                     
import torchvision.transforms as transforms
import numpy as np
from PIL import Image, ImageFilter
from skimage import util
import io

class PairCompose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, label):
        for t in self.transforms:
            img, label = t(img, label)
        return img, label

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
        x = random.randint(self.crop_size[0]//2, image.size[0]-self.crop_size[0]//2)
        y = random.randint(self.crop_size[1]//2, image.size[1]-self.crop_size[1]//2)
        image = image.crop((x-self.crop_size[0]//2, y-self.crop_size[1]//2, x+self.crop_size[0]//2, y+self.crop_size[1]//2))
        for i, l in enumerate(label):
            bbox = l['bbox']
            bbox_center = bbox[0]+bbox[2]/2, bbox[1]+bbox[3]/2
            if abs(bbox_center[0]-x)<self.crop_size[0]//2 and abs(bbox_center[1]-y)<self.crop_size[1]//2:
                bbox[0]=max(self.crop_size[0]//2 + bbox_center[0] - x - bbox[2]/2, 0)
                bbox[1]=max(self.crop_size[1]//2 + bbox_center[1] - y - bbox[3]/2, 0)
                bbox[2]=min(bbox[2], self.crop_size[0]-bbox[0])
                bbox[3]=min(bbox[3], self.crop_size[1]-bbox[1])
                label[i]['bbox'] = bbox
            else:
                label[i]=None
        label = [l for l in label if l is not None]
        return image, label

class RandomHorizontalFlipTransform(object):
    def __call__(self, image, label):
        p = random.random()
        if p>0.5:
            image = transforms.functional.hflip(image)
            for i, l in enumerate(label):
                bbox = l['bbox']
                bbox_center = bbox[0]+bbox[2]/2, bbox[1]+bbox[3]/2
                bbox_center = image.size[0]-bbox_center[0], bbox_center[1]
                bbox = [bbox_center[0]-bbox[2]/2, bbox_center[1]-bbox[3]/2, bbox[2], bbox[3]]
                label[i]['bbox'] = bbox
        return image, label

class RandomResizeTransform(object):
    def __call__(self, image, label):
        p = random.random()/2+0.5
        new_size = [np.round(s*p).astype(np.int32) for s in list(image.size)[::-1]]
        image = transforms.functional.resize(image, new_size, transforms.functional.InterpolationMode.LANCZOS)
        for i, l in enumerate(label):
            bbox = l['bbox']
            bbox = [b*p for b in bbox]
            label[i]['bbox'] = bbox
        return image, label

class RandomBlurTransform(object):
    def __init__(self):
        self.blur = ImageFilter.GaussianBlur(2)

    def __call__(self, image, label):
        p = random.random()
        if p>0.95:
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
        if p>0.95:
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
        image_padded = transforms.functional.pad(image, (0, 0, padding_x, padding_y))
        return image_padded, label

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

class OutputTransform(object):
    def __call__(self, image, label):
        image_padded = transforms.functional.to_tensor(image)
        return image_padded, label


