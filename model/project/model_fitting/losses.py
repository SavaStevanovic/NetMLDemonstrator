import torch.nn.functional as F
import torch
from sklearn.metrics import f1_score
import numpy as np

class YoloLoss(torch.nn.Module):

    def __init__(self, ranges):
        super(YoloLoss, self).__init__()
        self.l1_loss = torch.nn.L1Loss(reduction='none')
        self.l2_loss = torch.nn.MSELoss(reduction='none')
        self.class_loss = torch.nn.NLLLoss(reduction='none')
        self.size_scale = 0.5 
        self.offset_scale = 1.0 
        self.class_scale = 1.0
        self.ranges = ranges

    def forward(self, outputs, labels):
        total_objectness_loss = 0.0
        total_size_loss = 0.0
        total_offset_loss = 0.0
        total_class_loss = 0.0
        loss = 0.0
        for batch in range(labels.size()[0]):
            output, label = outputs[batch], labels[batch]

            obj_objectness = torch.flatten(output[self.ranges.objectness])
            lab_objectness = torch.flatten(label[self.ranges.objectness])
            objectness_loss = self.focal_loss(obj_objectness, lab_objectness)
            loss += objectness_loss
            total_objectness_loss += objectness_loss.item()

            obj_box_size = torch.flatten(output[self.ranges.size])
            lab_box_size = torch.flatten(label[self.ranges.size])
            lab_size_objectness = torch.cat([lab_objectness for _ in self.ranges.size], 0)
            size_loss = self.size_scale * lab_size_objectness.dot(self.l1_loss(obj_box_size, lab_box_size))
            loss += size_loss
            total_size_loss += size_loss.item()

            obj_offset = torch.flatten(output[self.ranges.offset])
            lab_offset = torch.flatten(label[self.ranges.offset])
            lab_offset_objectness = torch.cat([lab_objectness for _ in self.ranges.offset], 0)
            offset_loss = self.offset_scale * lab_offset_objectness.dot(self.l1_loss(obj_offset, lab_offset))
            loss += offset_loss
            total_offset_loss += offset_loss.item()

            obj_class = output[self.ranges.classes].unsqueeze(0)
            lab_class = label[self.ranges.classes].unsqueeze(0).argmax(1)
            lab_class_objectness = lab_objectness
            class_loss = self.class_scale * lab_class_objectness.dot(torch.flatten(self.class_loss(obj_class, lab_class)))
            loss += class_loss
            total_class_loss += class_loss.item()

        # loss = total_objectness_loss + total_size_loss + total_offset_loss + total_class_loss
        return loss, total_objectness_loss, total_size_loss, total_offset_loss, total_class_loss

    def focal_loss(self, x, y):
        alpha = 1
        gamma = 1
        y = y.unsqueeze(1)
        x = x.unsqueeze(1)

        y = torch.cat([y, 1-y], 1)
        x = torch.cat([x, 1-x], 1).clamp(1e-8, 1. - 1e-8)

        F_loss = -y * (1 - x) * torch.log(x)

        return F_loss.sum()