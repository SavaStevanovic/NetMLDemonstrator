from PIL import ImageDraw
import numpy as np
import scipy
import torch

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
        direction = direction/line_length
        line = dpart - fpart
        evaluation_points = np.linspace(fpart, dpart, num = 100, dtype = np.int32)
        costs = [np.dot(affinity_field[:, x[0], x[1]], direction) for x in evaluation_points]
        value = sum(costs)/100

        return value

    def __call__(self, affinity_field, part_heatmap):
        body_parts = []
        for i, heatmap in enumerate(part_heatmap):
            heatmap_candidates = heatmap > 0.5
            heatmap_cordinates = np.where(heatmap_candidates)
            heatmap_cordinates = np.asarray(heatmap_cordinates).T
            heatmap_cordinates = [(x[0], x[1], heatmap[x[0], x[1]]) for x in heatmap_cordinates]
            s_heatmap_cordinates = sorted(heatmap_cordinates, key=lambda e: -e[2]) 
            for candidate in s_heatmap_cordinates:
                if not heatmap_candidates[candidate[0], candidate[1]]:
                    continue
                body_parts.append((self.parts[i], candidate[:2]))
                points_to_process = [list(candidate[:2])]

                point_offsets = [np.array([0, 1]), np.array([0, -1]), np.array([1, 0]), np.array([-1, 0])]
                j = 0
                while j < len(points_to_process):
                    point = np.array(points_to_process[j])
                    j+=1
                    if point[0]<0 or point[0]>=heatmap.shape[0] or point[1]<0 or point[1]>=heatmap.shape[1]:
                        continue
                    point_distance = np.linalg.norm(candidate[:2]-point)
                    if point_distance > self.heatmap_distance:
                        continue
                    heatmap_candidates[point[0], point[1]] = False
                    next_points = [(point + offset).tolist() for offset in point_offsets if (point + offset).tolist() not in points_to_process]
                    points_to_process.extend(next_points)

        joints = []
        for ind, part_conn in enumerate(self.skeleton):
            fparts = [x for x in body_parts if x[0]==part_conn[0]]
            dparts = [x for x in body_parts if x[0]==part_conn[1]]
            dist = np.zeros([len(fparts), len(dparts)])
            for i, fpart in enumerate(fparts):
                for j, dpart in enumerate(dparts):
                    dist[i, j] = self.conn_cost(fpart[1], dpart[1], affinity_field[2*ind:2*ind+2])

            if part_conn == ['left_shoulder', 'right_shoulder'] and len(dist)>=2:
                print(dist)
            assignments = scipy.optimize.linear_sum_assignment(dist, maximize=True)
            assignments = np.asarray(assignments).T
            new_conns = [(fparts[p[0]], dparts[p[1]]) for p in assignments if dist[p[0], p[1]]>0.20]
            joints.extend(new_conns)

        return body_parts, joints
