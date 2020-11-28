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

    def __call__(self, img, label, mask):
        data = [img, label, mask]
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

    def __call__(self, image, label, mask):
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
        mask = mask.crop((*start_point, *end_point))
        start_point = np.array(start_point)
        end_point = np.array(end_point)
        for i, l in enumerate(label):
            l = {k:np.array(v) for k, v in l.items()}
            l = {k:v-start_point for k, v in l.items() if  (start_point <= v).all() and (end_point > v).all()}
            label[i] = {k:tuple(v) for k, v in l.items()}
        label = [l for l in label if bool(l)]
        return image, label, mask

class RandomHorizontalFlipTransform(object):
    def __call__(self, image, label, mask):
        p = random.random()
        if p>=0.5:
            image = transforms.functional.hflip(image)
            mask = transforms.functional.hflip(mask)
            end_point = np.array([image.size[0], 0])
            aligned_labels = []
            for i, l in enumerate(label):
                l = {k.replace('left', 'right') if 'left' in k else k.replace('right', 'left'):np.array(v) for k, v in l.items()}
                label[i] = {k:tuple(v) for k, v in l.items()}
            for i, l in enumerate(label):
                l = {k:np.array(v) for k, v in l.items()}
                l = {k:np.abs(end_point-v) for k, v in l.items()}
                label[i] = {k:tuple(v) for k, v in l.items()}
        return image, label, mask

class RandomResizeTransform(object):
    def __call__(self, image, label, mask):
        p = random.random()/2+0.5
        new_size = [np.round(s*p).astype(np.int32) for s in list(image.size)[::-1]]
        image = transforms.functional.resize(image, new_size, Image.ANTIALIAS)
        mask = transforms.functional.resize(mask, new_size, Image.ANTIALIAS)
        for i, l in enumerate(label):
            l = {k:np.array(v) for k, v in l.items()}
            l = {k:np.round(v*p).astype(np.int32) for k, v in l.items()}
            label[i] = {k:tuple(v) for k, v in l.items()}
        return image, label, mask

class RandomBlurTransform(object):
    def __init__(self):
        self.blur = ImageFilter.GaussianBlur(2)

    def __call__(self, image, label, mask):
        p = random.random()
        if p>0.9:
            image = image.filter(self.blur)
        return image, label, mask

class RandomColorJitterTransform(object):
    def __init__(self):
        self.jitt = transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
        )

    def __call__(self, image, label, mask):
        p = random.random()
        if p>0.75:
            image = self.jitt(image)
        return image, label, mask

class RandomNoiseTransform(object):
    def __init__(self):
        pass

    def __call__(self, image, label, mask):
        p = random.random()
        if p>0.9:
            image = np.array(image)
            image = util.random_noise(image, mode='gaussian', seed=None, clip=True)*255
            image = Image.fromarray(image.astype(np.uint8))
        return image, label, mask

class PaddTransform(object):
    def __init__(self, pad_size = 32):
        self.pad_size = pad_size

    def __call__(self, image, label, mask):
        padding_x = self.pad_size-image.size[0]%self.pad_size
        padding_x = (padding_x!=self.pad_size) * padding_x
        padding_y = self.pad_size-image.size[1]%self.pad_size
        padding_y = (padding_y!=self.pad_size) * padding_y
        image_padded = transforms.functional.pad(image, (0, 0, padding_x, padding_y))
        mask_padded = transforms.functional.pad(mask, (0, 0, padding_x, padding_y))
        return image_padded, label, mask_padded

class OutputTransform(object):
    def __call__(self, image, label, mask):
        image = transforms.functional.to_tensor(image)
        mask = transforms.functional.to_tensor(mask)
        if image.shape[0]==1:
            image = torch.cat([image]*3)
        return image, label[0], label[1], mask

# added to reflect inference state
class RandomJPEGcompression(object):
    def __init__(self, quality):
        self.quality = quality

    def __call__(self, image, label, mask):
        outputIoStream = io.BytesIO()
        image.save(outputIoStream, "JPEG", quality=self.quality, optimice=True)
        outputIoStream.seek(0)
        image = Image.open(outputIoStream)
        return image, label, mask

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

    def gen_part_heatmaps(self, dim, point, cov):
        x = np.arange(dim[1])
        y = np.arange(dim[0])
        xx, yy = np.meshgrid(x,y)
        xxyy = np.c_[xx.ravel(), yy.ravel()]

        # c = np.eye(2)*cov
        # point_prob = multivariate_normal(mean=point, cov=c)
        zz = np.linalg.norm(xxyy-point, 2, axis = 1)
        zz = np.exp(-0.5 * zz / cov**2)
        
        heat_img = zz.reshape(dim)
        


        return heat_img

    def __call__(self, image, labels, mask):
        image_size = np.array(image).shape[:2]
        afinity_fields_shape = [2*len(self.skeleton),*image_size]
        affinity_fields = torch.zeros(afinity_fields_shape, dtype=torch.float32)
        for label in labels:
            for i, part in enumerate(self.skeleton):
                spart = part[0]
                dpart = part[1]
                if spart in label and dpart in label:
                    spoint = np.array(label[spart][::-1])
                    dpoint = np.array(label[dpart][::-1])
                    direction = dpoint - spoint
                    line_length = np.linalg.norm(direction, 2) + 1e-8
                    direction = direction/line_length
                    points_to_process = [spoint.tolist()]
                    max_distance = max(line_length / self.distance, 4)
                    extended_line = [spoint-direction*max_distance, dpoint+direction*max_distance]
                    pdir = np.array([-1,1])*max_distance
                    rec_cord = [[p+direction[::-1]*pdir, p-direction[::-1]*pdir] for p in extended_line]
                    rec_cords = rec_cord[0][::-1] + rec_cord[1]
                    rr, cc = polygon(np.array([x[0] for x in rec_cords]), np.array([x[1] for x in rec_cords]))
                    list_filter = [rr[i]>=0 and cc[i]>=0 and rr[i]<image_size[0] and cc[i]<image_size[1] for i in range(len(rr))]
                    rr = np.array(list(compress(rr, list_filter)))
                    cc = np.array(list(compress(cc, list_filter)))
                    if rr.size:
                        affinity_fields[2*i:2*i+2, rr, cc] += np.vstack([direction]*rr.size).T.astype(np.float32)
                    # point_offsets = [np.array([0, 1]), np.array([0, -1]), np.array([1, 0]), np.array([-1, 0])]
                    # j = 0
                    # while j < len(points_to_process):
                    #     point = np.array(points_to_process[j])
                    #     j+=1
                    #     if point[0]<0 or point[0]>=image.size[0] or point[1]<0 or point[1]>=image.size[1]:
                    #         continue
                    #     point_distance = self.point_segment_distance(point, spoint, dpoint)
                    #     if point_distance > max_distance:
                    #         continue
                    #     affinity_fields[2*i:2*i+2, point[0], point[1]] += direction
                    #     next_points = [(point + offset).tolist() for offset in point_offsets if (point + offset).tolist() not in points_to_process]
                    #     points_to_process.extend(next_points)

        for i in range(len(self.skeleton)):
            affinity_fields[2*i:2*i+2] /= np.linalg.norm(affinity_fields[2*i:2*i+2], 2, 0) + 1e-8
        #     field = affinity_fields[2*i:2*i+2].permute(1, 2, 0).numpy()
        #     image = np.array(image)
        #     if (field!=0).any():
        #         image[..., :2] = np.where(field==0, image[..., :2], np.uint8(field*255))

        # image = Image.fromarray(image, 'RGB')
        # plt.imshow(image)
        # plt.show()   

        part_shape = [len(self.parts)+1,*image_size]
        part_heatmaps = np.zeros(part_shape, dtype=np.float32)

        for label in labels: 
            for part in label.items():
                i = self.parts.index(part[0])
                part_heatmap = part_heatmaps[i]
                point = part[1]
                center_point = np.array(point)
                point_map = self.gen_part_heatmaps(image_size, center_point, self.heatmap_distance)
                part_heatmap[point_map > part_heatmap] = point_map[point_map > part_heatmap]
                # points_to_process = [list(point)]

                # point_offsets = [np.array([0, 1]), np.array([0, -1]), np.array([1, 0]), np.array([-1, 0])]
                # j = 0
                # while j < len(points_to_process):
                #     point = np.array(points_to_process[j])
                #     j+=1
                #     if point[0]<0 or point[0]>=image.size[0] or point[1]<0 or point[1]>=image.size[1]:
                #         continue
                #     point_distance = np.linalg.norm(center_point-point)
                #     if point_distance > self.heatmap_distance:
                #         continue
                #     part_heatmap[point[0], point[1]] = max(part_heatmap[point[0], point[1]], 1 - point_distance/self.heatmap_distance)
                #     next_points = [(point + offset).tolist() for offset in point_offsets if (point + offset).tolist() not in points_to_process]
                #     points_to_process.extend(next_points)
        part_heatmaps[-1] = 1 - np.max(part_heatmaps, axis=0)
        # plt.imshow(part_heatmaps[-1]); 
        # plt.imshow(image, alpha=0.2)
        # plt.show()
        part_heatmaps = torch.from_numpy(part_heatmaps)
        return image, (affinity_fields, part_heatmaps), mask
