import torch
from torchmetrics import Dice


class SegmentationLoss(torch.nn.Module):
    def __init__(self):
        super(SegmentationLoss, self).__init__()
        self.focal_loss_scale = 20.0
        self.dice_loss_scale = 1.0
        self._dice_loss = Dice().cuda()

    def forward(self, output, label):
        output = output.sigmoid()
        fc_loss = self.focal_loss_scale * self.focal_loss(output, label)
        dice_loss = self.dice_loss_scale * self._dice_loss(output, label.int())
        total_loss = dice_loss + fc_loss

        return total_loss, dice_loss, fc_loss

    def focal_loss(self, probs, target):
        gamma = 2
        alpha = 0.25
        eps = 1e-8

        F_loss = -alpha * torch.pow((1.0 - probs), gamma) * target * torch.log(
            probs + eps
        ) - (1.0 - alpha) * torch.pow(probs, gamma) * (1.0 - target) * torch.log(
            1.0 - probs + eps
        )

        return F_loss.mean()
