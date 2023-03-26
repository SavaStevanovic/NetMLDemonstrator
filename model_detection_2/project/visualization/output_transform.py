from PIL import ImageDraw
import numpy as np

class TargetTransformToBoxes(object):
    def __init__(self, prior_box_sizes, classes, ratios, strides):
        self.classes = classes
        self.classes_len = len(classes)
        self.prior_box_sizes = prior_box_sizes
        self.ratios = ratios
        self.strides = strides

    def __call__(self, targets, threshold = 0.5, scale = None, depth = 0):
        labels = []
        for i in range(min(len(targets), len(self.strides), len(self.prior_box_sizes))):
            stride = self.strides[i]
            prior_box_size = self.prior_box_sizes[i]
            target = targets[i]
            for j, first_target in enumerate(target):
                objects = np.argwhere(first_target[::(5+self.classes_len)]>=threshold)
                labels += [self.convert_cords(first_target, cords, i, depth, scale, self.ratios[j]) for cords in objects]
        return labels

    def convert_cords(self, first_target, cords, i, depth, scale, ratio):
        l = {}
        l['category'] = self.classes[np.argmax(first_target[cords[0]+5:cords[0]+5+self.classes_len, cords[1], cords[2]]).item()]
        box_center = first_target[cords[0]+3, cords[1], cords[2]], first_target[cords[0]+4, cords[1], cords[2]]
        box_size = first_target[cords[0]+1, cords[1], cords[2]], first_target[cords[0]+2, cords[1], cords[2]]
        box_size = np.exp(box_size)*self.prior_box_sizes[i]
        box_size = box_size[0]/ratio, box_size[1]
        l['bbox'] = ((cords[2]+box_center[0])*self.strides[i] - box_size[0]/2).item(), ((cords[1]+box_center[1])*self.strides[i] - box_size[1]/2).item(), box_size[0], box_size[1]
        l['bbox'] = [b*2**depth for b in l['bbox']]
        l['confidence'] = first_target[cords[0], cords[1], cords[2]].item()
        if scale:
            l['bbox'] = [min(max(x/scale[i%2], 0), 1) for i, x in enumerate(l['bbox'])]
        return l

class TargetTransform(object):
    def __init__(self, prior_box_sizes, classes, ratios, strides):
        self.classes = classes
        self.classes_len = len(classes)
        self.prior_box_sizes = prior_box_sizes
        self.ratios = ratios
        self.strides = strides

    def __call__(self, image, labels):
        targets = []
        for i in range(min(len(self.strides), len(self.prior_box_sizes))):
            stride = self.strides[i]
            prior_box_size = self.prior_box_sizes[i]
            target = np.zeros((len(self.ratios), (5 + self.classes_len), image.shape[1]//stride, image.shape[2]//stride), dtype=np.float32)
            for l in labels:
                id = self.classes.index(l['category']) + 5
                bbox = l['bbox']
                box_center = (bbox[0] + bbox[2]/2)/stride, (bbox[1] + bbox[3]/2)/stride  
                box_position = np.floor(box_center[0]).astype(np.int), np.floor(box_center[1]).astype(np.int)
                i = min(range(len(self.ratios)), key=lambda i: abs(self.ratios[i]-bbox[2]/(bbox[3]+1e-9)))
                if  target[i,  0, box_position[1], box_position[0]]== 0:
                    target[i,  0, box_position[1], box_position[0]] = 1
                    target[i,  1, box_position[1], box_position[0]] = np.log(max(bbox[2], 1)/prior_box_size*self.ratios[i])
                    target[i,  2, box_position[1], box_position[0]] = np.log(max(bbox[3], 1)/prior_box_size)
                    target[i,  3, box_position[1], box_position[0]] = box_center[0] - box_position[0]
                    target[i,  4, box_position[1], box_position[0]] = box_center[1] - box_position[1]
                    target[i, id, box_position[1], box_position[0]] = 1
            targets.append(target)
        return image, targets