import random
import torchvision.transforms as transforms
import numpy as np
from PIL import Image, ImageFilter
import io
import torch
import typing


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


class RandomHorizontalFlipTransform(object):
    def __call__(self, image, label):
        p = random.random()
        if p > 0.5:
            image = transforms.functional.hflip(image)
            for i, l in enumerate(label):
                bbox = l['bbox']
                bbox_center = bbox[0]+bbox[2]/2, bbox[1]+bbox[3]/2
                bbox_center = image.size[0]-bbox_center[0], bbox_center[1]
                bbox = [bbox_center[0]-bbox[2]/2,
                        bbox_center[1]-bbox[3]/2, bbox[2], bbox[3]]
                label[i]['bbox'] = bbox
        return image, label


class ResizeTransform(object):
    def __init__(self, size: typing.Tuple[int]):
        self._size = size

    def __call__(self, image):
        image = torch.nn.functional.interpolate(
            image.unsqueeze(0), self._size, mode="area")

        return image


class RandomResizeTransform(object):
    def __call__(self, image, label):
        p = random.random()/2+0.5
        new_size = [np.round(s*p).astype(np.int32)
                    for s in list(image.size)[::-1]]
        image = transforms.functional.resize(image, new_size, Image.ANTIALIAS)
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
        if p > 0.95:
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
        if p > 0.75:
            image = self.jitt(image)
        return image, label


class PaddTransform(object):
    def __init__(self, pad_size=32):
        self.pad_size = pad_size

    def __call__(self, image, label):
        padding_x = self.pad_size-image.size[0] % self.pad_size
        padding_x = (padding_x != self.pad_size) * padding_x
        padding_y = self.pad_size-image.size[1] % self.pad_size
        padding_y = (padding_y != self.pad_size) * padding_y
        image_padded = transforms.functional.pad(
            image, (0, 0, padding_x, padding_y))
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


class TargetTransform(object):
    def __init__(self, prior_box_sizes, classes, ratios, strides):
        self.classes = [x[0] for x in classes]
        self.classes_len = len(classes)
        self.prior_box_sizes = prior_box_sizes
        self.ratios = ratios
        self.strides = strides

    def __call__(self, image, labels):
        targets = []
        for i in range(min(len(self.strides), len(self.prior_box_sizes))):
            stride = self.strides[i]
            prior_box_size = self.prior_box_sizes[i]
            target = np.zeros((len(self.ratios), (5 + self.classes_len),
                              image.size[1]//stride, image.size[0]//stride), dtype=np.float32)
            for l in labels:
                id = self.classes.index(l['category_id']) + 5
                bbox = l['bbox']
                box_center = (bbox[0] + bbox[2]/2) / \
                    stride, (bbox[1] + bbox[3]/2)/stride
                box_position = np.floor(box_center[0]).astype(
                    np.int), np.floor(box_center[1]).astype(np.int)
                i = min(range(len(self.ratios)), key=lambda i: abs(
                    self.ratios[i]-bbox[2]/(bbox[3]+1e-9)))
                if target[i,  0, box_position[1], box_position[0]] == 0:
                    target[i,  0, box_position[1], box_position[0]] = 1
                    target[i,  1, box_position[1], box_position[0]] = np.log(
                        max(bbox[2], 1)/prior_box_size*self.ratios[i])
                    target[i,  2, box_position[1], box_position[0]
                           ] = np.log(max(bbox[3], 1)/prior_box_size)
                    target[i,  3, box_position[1], box_position[0]
                           ] = box_center[0] - box_position[0]
                    target[i,  4, box_position[1], box_position[0]
                           ] = box_center[1] - box_position[1]
                    target[i, id, box_position[1], box_position[0]] = 1
            targets.append(target)
        return image, targets
