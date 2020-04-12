import torch.nn.functional as F
import torch
from sklearn.metrics import f1_score
import numpy as np

class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.eps = 1e-9

    def forward(self, output1, output2, label):
        distances = (output2 - output1).pow(2).sum(1)  # squared distances
        losses = label * distances + (1 + -1 * label) * F.relu(self.margin - (distances + self.eps).sqrt()).pow(2)
        return losses

class YoloLoss(torch.nn.Module):

    def __init__(self, classes_len, ratios):
        super(YoloLoss, self).__init__()
        self.classes_len = classes_len
        self.ratios = ratios
        self.l1_loss = torch.nn.L1Loss(reduction='none')

    def forward(self, outputs, labels):
        loss = 0
        total_objectness_loss = 0
        total_size_loss = 0
        objectnes_f1_scores = []
        for batch in range(labels.size()[0]):
            output, label = outputs[batch], labels[batch]
            object_range = 5*len(self.ratios)+self.classes_len
            obj_objectness = torch.sigmoid(output[::object_range])
            lab_objectness = label[::object_range]
            obj_objectness = torch.flatten(obj_objectness)
            lab_objectness = torch.flatten(lab_objectness)
            total_objectness_loss+= self.focal_loss(obj_objectness, lab_objectness)

            size_range = [1,2]
            box_size_range = [i for i in range(label.shape[0]) if i%object_range in size_range]
            obj_box_size = torch.flatten(output[box_size_range])
            lab_box_size = torch.flatten(label[box_size_range])
            lab_size_objectness = torch.cat([lab_objectness for _ in size_range], 0)
            total_size_loss += torch.sum(lab_size_objectness * self.l1_loss(obj_box_size, lab_box_size))
            objectnes_f1_scores.append(f1_score(lab_objectness.cpu(), obj_objectness.cpu()>0.5))
        objectnes_f1_score = np.average(objectnes_f1_scores)
        loss = total_objectness_loss + total_size_loss
        return loss, objectnes_f1_score, total_objectness_loss, total_size_loss

    def focal_loss(self, x, y):
        alpha = 1
        gamma = 1
        y = y.unsqueeze(1)
        x = x.unsqueeze(1)

        y = torch.cat([y, 1-y], 1)
        x = torch.cat([x, 1-x], 1).clamp(1e-8, 1. - 1e-8)

        F_loss = -y * (1 - x) * torch.log(x)

        return F_loss.sum()