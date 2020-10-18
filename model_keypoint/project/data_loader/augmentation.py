import random                     
import torchvision.transforms as transforms
import numpy as np
from PIL import Image, ImageFilter
from skimage import util
import torch
import matplotlib.pyplot as plt
import cv2

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
        x_radius = self.crop_size[0]//2
        y_radius = self.crop_size[1]//2
        x = random.randint(x_radius, image.size[0]-x_radius)
        y = random.randint(y_radius, image.size[1]-y_radius)
        start_point = [x-x_radius,y-y_radius]
        end_point = [x+x_radius, y+y_radius]
        image = image.crop((*start_point, *end_point))
        start_point = np.array(start_point)
        end_point = np.array(end_point)
        for i, l in enumerate(label):
            l = {k:np.array(v) for k, v in l.items()}
            l = {k:v-start_point for k, v in l.items() if  (start_point <= v).all() and (end_point > v).all()}
            label[i] = {k:tuple(v) for k, v in l.items()}
        label = [l for l in label if bool(l)]
        return image, label

class RandomHorizontalFlipTransform(object):
    def __call__(self, image, label):
        p = random.random()
        if p<0.5:
            return image, label
        image = transforms.functional.hflip(image)
        end_point = np.array([image.size[0], 0])
        for i, l in enumerate(label):
            l = {k:np.array(v) for k, v in l.items()}
            l = {k:np.abs(end_point-v) for k, v in l.items()}
            label[i] = {k:tuple(v) for k, v in l.items()}
        return image, label

class RandomResizeTransform(object):
    def __call__(self, image, label):
        p = random.random()/2+0.5
        new_size = [np.round(s*p).astype(np.int32) for s in list(image.size)[::-1]]
        image = transforms.functional.resize(image, new_size, Image.ANTIALIAS)
        for i, l in enumerate(label):
            l = {k:np.array(v) for k, v in l.items()}
            l = {k:np.round(v*p).astype(np.int32) for k, v in l.items()}
            label[i] = {k:tuple(v) for k, v in l.items()}
        return image, label

class RandomBlurTransform(object):
    def __init__(self):
        self.blur = ImageFilter.GaussianBlur(2)

    def __call__(self, image, label):
        p = random.random()
        if p>0.75:
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
        if p>0.75:
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
            target = np.zeros((len(self.ratios), (5 + self.classes_len), image.size[1]//stride, image.size[0]//stride), dtype=np.float32)
            for l in labels:
                id = self.classes.index(l['category_id']) + 5
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

class PartAffinityFieldTransform(object):
    def __init__(self, skeleton, distance, heatmap_distance, parts):
        self.skeleton = skeleton
        self.distance = distance
        self.heatmap_distance = heatmap_distance
        self.parts = parts

    def point_segment_distance(self, point, line_point1, line_point2):
        point_distances = [np.linalg.norm(line_point2-point), np.linalg.norm(line_point1-point)]
        max_distance = max(point_distances)
        line_length = np.linalg.norm(line_point1-line_point2)
        if max_distance >= line_length:
            return min(point_distances)
        line_distance = np.abs(np.cross(line_point2-line_point1, line_point1-point)) / line_length
        return line_distance

    def __call__(self, image, labels):
        afinity_fields_shape = [2*len(self.skeleton),*image.size[:2]]
        affinity_fields = torch.zeros(afinity_fields_shape, dtype=torch.float32)
        for label in labels:
            for i, part in enumerate(self.skeleton):
                spart = part[0]
                dpart = part[1]
                if spart in label and dpart in label:
                    spoint = np.array(label[spart][::-1])
                    dpoint = np.array(label[dpart][::-1])
                    direction = dpoint - spoint
                    line_length = np.linalg.norm(direction, 2)
                    direction = direction/line_length
                    points_to_process = [spoint.tolist()]
                    max_distance = 5

                    point_offsets = [np.array([0, 1]), np.array([0, -1]), np.array([1, 0]), np.array([-1, 0])]
                    j = 0
                    while j < len(points_to_process):
                        point = np.array(points_to_process[j])
                        j+=1
                        if point[0]<0 or point[0]>=image.size[0] or point[1]<0 or point[1]>=image.size[1]:
                            continue
                        point_distance = self.point_segment_distance(point, spoint, dpoint)
                        if point_distance > max_distance:
                            continue
                        affinity_fields[2*i:2*i+2, point[0], point[1]] += direction
                        next_points = [(point + offset).tolist() for offset in point_offsets if (point + offset).tolist() not in points_to_process]
                        points_to_process.extend(next_points)

        for i in range(len(self.skeleton)):
            affinity_fields[2*i:2*i+2] /= np.linalg.norm(affinity_fields[2*i:2*i+2], 2, 0) + 1e-8
            # field = affinity_fields[2*i:2*i+2].permute(1, 2, 0).numpy()
            # if (field!=0).any():
            #     image[..., :2] = np.where(field==0, image[..., :2], np.uint8(field*255))

        # img = Image.fromarray(image, 'RGB')
        # plt.imshow(img)
        # plt.show()   

        part_shape = [len(self.parts),*image.size[:2]]
        part_heatmaps = torch.zeros(part_shape, dtype=torch.float32)

        for label in labels: 
            for part in label.items():
                i = self.parts.index(part[0])
                part_heatmap = part_heatmaps[i]
                point = part[1][::-1]
                center_point = np.array(point)
                points_to_process = [list(point)]

                point_offsets = [np.array([0, 1]), np.array([0, -1]), np.array([1, 0]), np.array([-1, 0])]
                j = 0
                while j < len(points_to_process):
                    point = np.array(points_to_process[j])
                    j+=1
                    if point[0]<0 or point[0]>=image.size[0] or point[1]<0 or point[1]>=image.size[1]:
                        continue
                    point_distance = np.linalg.norm(center_point-point)
                    if point_distance > self.heatmap_distance:
                        continue
                    part_heatmap[point[0], point[1]] = max(part_heatmap[point[0], point[1]], 1 - point_distance/self.heatmap_distance)
                    next_points = [(point + offset).tolist() for offset in point_offsets if (point + offset).tolist() not in points_to_process]
                    points_to_process.extend(next_points)

        return image, (labels, affinity_fields, part_heatmaps)
