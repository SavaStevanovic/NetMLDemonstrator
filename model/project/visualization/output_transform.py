from PIL import ImageDraw
import numpy as np


class TargetTransformToBoxes(object):
    def __init__(self, prior_box_sizes, classes, ratios, strides):
        self.classes = classes
        self.classes_len = len(classes)
        self.prior_box_sizes = prior_box_sizes
        self.ratios = ratios
        self.strides = strides

    def __call__(self, targets, threshold = 0.5):
        labels = []
        for i in range(min(len(targets), len(self.strides), len(self.prior_box_sizes))):
            stride = self.strides[i]
            prior_box_size = self.prior_box_sizes[i]
            target = targets[i]
            for i, first_target in enumerate(target):
                objects = np.argwhere(first_target[::(5+self.classes_len)]>=threshold)
                for cords in objects:
                    l = {}
                    l['category_id'] = np.argmax(first_target[cords[0]+5:cords[0]+5+self.classes_len, cords[1], cords[2]]).item()
                    box_center = first_target[cords[0]+3, cords[1], cords[2]], first_target[cords[0]+4, cords[1], cords[2]]
                    box_size = first_target[cords[0]+1, cords[1], cords[2]], first_target[cords[0]+2, cords[1], cords[2]]
                    box_size = np.exp(box_size)*prior_box_size
                    box_size = box_size[0]/self.ratios[i], box_size[1]
                    l['bbox'] = ((cords[2]+box_center[0])*stride - box_size[0]/2).item(), ((cords[1]+box_center[1])*stride - box_size[1]/2).item(), box_size[0], box_size[1]
                    l['confidence'] = first_target[cords[0], cords[1], cords[2]].item()
                    labels.append(l)
        return labels
