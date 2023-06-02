import torch
import torch.nn.functional as F
import torch.nn as nn


class DiceLoss(nn.Module):
    def forward(self, inputs, targets, smooth=1):
        # comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2.0 * intersection + smooth) / \
            (inputs.sum() + targets.sum() + smooth)

        return 1 - dice


ALPHA = 0.8
GAMMA = 2


class FocalLoss(nn.Module):
    def forward(self, inputs, targets, alpha=ALPHA, gamma=GAMMA, smooth=1):
        # comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        # first compute binary cross-entropy
        BCE = F.binary_cross_entropy(inputs, targets.float(), reduction="mean")
        BCE_EXP = torch.exp(-BCE)
        focal_loss = alpha * (1 - BCE_EXP) ** gamma * BCE

        return focal_loss


class SegmentationLoss(torch.nn.Module):
    def forward(self, output, label):
        fc_loss = FocalLoss()(output, label)
        dice_loss = 0  # DiceLoss()(output, label)
        return fc_loss + dice_loss, fc_loss.item(), 0

    def focal_loss(self, probs, target):
        gamma = 2
        alpha = 0.60
        eps = 1e-8

        F_loss = -alpha * torch.pow((1.0 - probs), gamma) * target * torch.log(
            probs + eps
        ) - (1.0 - alpha) * torch.pow(probs, gamma) * (1.0 - target) * torch.log(
            1.0 - probs + eps
        )

        return F_loss.mean()
