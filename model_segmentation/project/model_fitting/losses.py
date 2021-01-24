import torch.nn.functional as F
import torch
from sklearn.metrics import f1_score
import numpy as np

class SegmentationLoss(torch.nn.Module):

    def __init__(self):
        super(SegmentationLoss, self).__init__()
        self.focal_loss_scale = 20.0
        self.dice_loss_scale = 1.0

    def forward(self, output, label):
        total_focal_loss = 0.0
        total_dice_loss = 0.0
        loss = 0.0

        output = output.sigmoid()

        fc_loss = self.focal_loss_scale * self.focal_loss(output, label)
        total_focal_loss += fc_loss.item()
        loss += fc_loss

        dice_loss = self.dice_loss_scale * self.dice_loss(output, label)
        total_dice_loss += dice_loss.item()
        loss += dice_loss

        return loss, total_focal_loss, total_dice_loss

    def focal_loss(self, probs, target):
        gamma = 2
        alpha = .80
        eps = 1e-8

        F_loss = -alpha * torch.pow((1. - probs), gamma) * target * torch.log(probs + eps) \
               - (1. - alpha) * torch.pow(probs, gamma) * (1. - target) * torch.log(1. - probs + eps)

        return F_loss.mean()

    def dice_loss(self, x, y):
        x = torch.cat([1-x, x], 1)
        y = torch.cat([1-y, y], 1)
        eps = 1e-8
        dims = (1, 2, 3)
        intersection = torch.sum(x * y, dims)
        cardinality = torch.sum(x + y, dims)

        dice_score = 2. * intersection / (cardinality + eps)

        return dice_score.mean()