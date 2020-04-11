import random                     
import torchvision.transforms as transforms
import numpy as np

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

class PaddTransform(object):
    def __init__(self, pad_size = 32):
        self.pad_size = pad_size

    def __call__(self, image, label):
        padding_x = self.pad_size-image.size[0]%self.pad_size
        padding_x = (padding_x!=self.pad_size) * padding_x
        padding_y = self.pad_size-image.size[1]%self.pad_size
        padding_y = (padding_y!=self.pad_size) * padding_y
        image_padded = transforms.Pad((0, 0, padding_x, padding_y))(image)
        return image_padded, label

class OutputTransform(object):
    def __init__(self):
        pass

    def __call__(self, image, label):
        padding_x = 32-image.size[0]%32
        padding_x = (padding_x!=32) * padding_x
        padding_y = 32-image.size[1]%32
        padding_y = (padding_y!=32) * padding_y
        image_padded = transforms.Pad((0, 0, padding_x, padding_y))(image)
        image_padded = transforms.ToTensor()(image_padded)
        return image_padded, label

class TargetTransform(object):
    def __init__(self, prior_box_size, classes, ratios, stride):
        self.classes = [x[0] for x in classes]
        self.classes_len = len(classes)
        self.prior_box_size = prior_box_size
        self.ratios = ratios
        self.stride = stride

    def __call__(self, image, labels):
        target = np.zeros((len(self.ratios)*5 + self.classes_len, image.size[1]//self.stride, image.size[0]//self.stride))
        for l in labels:
            id = self.classes.index(l['category_id']) + 5
            bbox = l['bbox']
            box_center = (bbox[0] + bbox[2]/2)/self.stride, (bbox[1] + bbox[3]/2)/self.stride  
            box_position = np.floor(box_center[0]).astype(np.int), np.floor(box_center[1]).astype(np.int)
            i = min(range(len(self.ratios)), key=lambda i: abs(self.ratios[i]-bbox[2]/bbox[3]))
            target[i * (len(self.ratios)*5+self.classes_len)+ 0, box_position[1], box_position[0]] = 1
            target[i * (len(self.ratios)*5+self.classes_len)+ 1, box_position[1], box_position[0]] = np.log(bbox[2]/self.prior_box_size*self.ratios[i])
            target[i * (len(self.ratios)*5+self.classes_len)+ 2, box_position[1], box_position[0]] = np.log(bbox[3]/self.prior_box_size)
            target[i * (len(self.ratios)*5+self.classes_len)+ 3, box_position[1], box_position[0]] = box_center[0] - box_position[0]
            target[i * (len(self.ratios)*5+self.classes_len)+ 4, box_position[1], box_position[0]] = box_center[1] - box_position[1]
            target[i * (len(self.ratios)*5+self.classes_len)+id, box_position[1], box_position[0]] = 1
        return image, target

class TargetTransformToBoxes(object):
    def __init__(self, prior_box_size, classes, ratios, stride):
        self.classes = classes
        self.classes_len = len(classes)
        self.prior_box_size = prior_box_size
        self.ratios = ratios
        self.stride = stride

    def __call__(self, target):
        first_target = target[0,...]
        objects = np.argwhere(first_target[::(5+self.classes_len)]>0.1).T
        labels = []
        for cords in objects:
            l = {}
            class_id = np.argmax(first_target[cords[0]+5:cords[0]+5+self.classes_len, cords[1], cords[2]])
            l['category_id'] = class_id
            box_center = first_target[cords[0]+3, cords[1], cords[2]], first_target[cords[0]+4, cords[1], cords[2]]
            box_size = first_target[cords[0]+1, cords[1], cords[2]], first_target[cords[0]+2, cords[1], cords[2]]
            box_size = np.exp(box_size)*self.prior_box_size
            box_size = box_size[0]/self.ratios[cords[0]], box_size[1]
            bbox = (cords[2]+box_center[0])*self.stride - box_size[0]/2, (cords[1]+box_center[1])*self.stride - box_size[1]/2, box_size[0], box_size[1]
            l['bbox'] = bbox
            labels.append(l)
        return labels
