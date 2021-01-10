import torch.nn.functional as F
import torch
from sklearn.metrics import f1_score
import numpy as np

class SegmentationLoss(torch.nn.Module):

    def __init__(self):
        super(SegmentationLoss, self).__init__()
        self.focal_loss_scale = 5.0
        self.dice_loss_scale = 1.0

    def forward(self, output, label):
        total_focal_loss = 0.0
        total_dice_loss = 0.0
        loss = 0.0

        output = output.softmax(1)
        label = label.unsqueeze(1)
        label = torch.cat([1-label, label], 1)

        fc_loss = self.focal_loss_scale * self.focal_loss(output, label)
        total_focal_loss += fc_loss.item()
        loss += fc_loss

        dice_loss = self.dice_loss_scale * self.dice_loss(output, label)
        total_dice_loss += dice_loss.item()
        loss += dice_loss

        return loss, total_focal_loss, total_dice_loss

    def focal_loss(self, x, y):
        gamma = 2

        x = x.clamp(1e-8, 1. - 1e-8)

        F_loss = -y * (1 - x)**gamma * torch.log(x)

        return F_loss.mean()

    def dice_loss(self, x, y):
        smooth = 1.

        intersection = (x * y).sum()
        
        return 1 - (2. * intersection + smooth) / (x.sum() + y.sum() + smooth)