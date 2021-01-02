from PIL import ImageDraw
import numpy as np
import scipy
import torch
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image
from skimage.draw import disk
from itertools import compress
from scipy.ndimage.filters import gaussian_filter
from utils import plt_to_np
from scipy.optimize import linear_sum_assignment

class PartAffinityFieldTransform(object):
    def __init__(self, skeleton, heatmap_distance):
        self.skeleton = skeleton
        self.heatmap_distance = heatmap_distance
        self.parts = list(set(sum(self.skeleton, [])))

    def conn_cost(self, fpart, dpart, affinity_field):
        fpart = np.array(fpart)
        dpart = np.array(dpart)
        direction = dpart - fpart
        line_length = np.linalg.norm(direction, 2)
        direction = direction/(line_length + 1e-8)
        line = dpart - fpart
        p_count = 10
        evaluation_points = np.linspace(fpart, dpart, num = p_count, dtype = np.int32)
        costs = [np.dot(affinity_field[:, x[0], x[1]], direction) for x in evaluation_points]
        value = sum(costs)/p_count

        return value

    def __call__(self, affinity_field, part_heatmap, threshold=0.5):
        # return [], []
        body_parts = []
        for i, heatmap in enumerate(part_heatmap[:-1]):
            heatmap = gaussian_filter(heatmap, sigma=self.heatmap_distance)
            map_left = np.zeros(heatmap.shape)
            map_right = np.zeros(heatmap.shape)
            map_top = np.zeros(heatmap.shape)
            map_bottom = np.zeros(heatmap.shape)

            map_left[1:, :] = heatmap[:-1, :]
            map_right[:-1, :] = heatmap[1:, :]
            map_top[:, 1:] = heatmap[:, :-1]
            map_bottom[:, :-1] = heatmap[:, 1:]

            peaks_binary = np.logical_and.reduce((
                heatmap > threshold,
                heatmap > map_left,
                heatmap > map_right,
                heatmap > map_top,
                heatmap > map_bottom,
            ))

            heatmap_cordinates = np.where(peaks_binary)
            heatmap_cordinates = np.asarray(heatmap_cordinates).T
            heatmap_cordinates = [(x[0], x[1], heatmap[x[0], x[1]]) for x in heatmap_cordinates]
            s_heatmap_cordinates = sorted(heatmap_cordinates, key=lambda e: -e[2]) 
            for candidate in s_heatmap_cordinates:
                if len(body_parts)>100:
                    break
                if not peaks_binary[candidate[0], candidate[1]]:
                    continue
                rr, cc = disk(candidate[:2], self.heatmap_distance*2)
                # img[rr, cc] = 1
                list_filter = [rr[i]>=0 and cc[i]>=0 and rr[i]<heatmap.shape[0] and cc[i]<heatmap.shape[1] for i in range(len(rr))]
                rr = np.array(list(compress(rr, list_filter)))
                cc = np.array(list(compress(cc, list_filter)))
                peaks_binary[rr, cc] = False
                body_parts.append((self.parts[i], candidate[:2]))
                # points_to_process = [list(candidate[:2])]

                # point_offsets = [np.array([0, 1]), np.array([0, -1]), np.array([1, 0]), np.array([-1, 0])]
                # j = 0
                # while j < len(points_to_process):
                #     point = np.array(points_to_process[j])
                #     j+=1
                #     if point[0]<0 or point[0]>=heatmap.shape[0] or point[1]<0 or point[1]>=heatmap.shape[1]:
                #         continue
                #     point_distance = np.linalg.norm(candidate[:2]-point)
                #     if point_distance > self.heatmap_distance:
                #         continue
                #     heatmap_candidates[point[0], point[1]] = False
                #     next_points = [(point + offset).tolist() for offset in point_offsets if (point + offset).tolist() not in points_to_process]
                #     points_to_process.extend(next_points)

        joints = []
        # body_parts = body_parts[:100]
        # print(len(body_parts))
        for ind, part_conn in enumerate(self.skeleton):
            fparts = [x for x in body_parts if x[0]==part_conn[0]]
            dparts = [x for x in body_parts if x[0]==part_conn[1]]
            dist = np.zeros([len(fparts), len(dparts)])
            for i, fpart in enumerate(fparts):
                for j, dpart in enumerate(dparts):
                    dist[i, j] = self.conn_cost(fpart[1], dpart[1], affinity_field[2*ind:2*ind+2])

            # if len(dist)>=2:
            #     print(dist)
            assignments = linear_sum_assignment(dist, maximize=True)
            assignments = np.asarray(assignments).T
            new_conns = [(fparts[p[0]], dparts[p[1]]) for p in assignments if dist[p[0], p[1]]>0.00]
            joints.extend(new_conns)

        return body_parts, joints

def getRGBfromI(RGBint):
    blue =  RGBint & 255
    green = (RGBint >> 8) & 255
    red =   (RGBint >> 16) & 255
    return red/255, green/255, blue/255

def get_mapped_image(images, pafs, maps, postprocessing, skeleton, parts):
    outputs = postprocessing(pafs[0].detach().cpu().numpy(), maps[0].detach().cpu().numpy(), 0.5)
    image = (images[0].permute(1,2,0).numpy() * 255).round().astype(np.uint8)
    plt.imshow(image); plt.axis('off')
    for o in outputs[1]:
        c_off = skeleton.index([o[0][0],o[1][0]])+1
        c = getRGBfromI(16777216//c_off)
        line = np.asarray([x[1][::-1] for x in o]).T
        plt.plot(line[0], line[1], '-', linewidth=1, color=c)
    for o in outputs[0]:
        c = getRGBfromI(16777216//(parts.index(o[0])+1))
        plt.plot(o[1][1], o[1][0],'o',markersize=6, markerfacecolor=c, markeredgecolor='w',markeredgewidth=1)

    image = plt_to_np(plt)
    return image