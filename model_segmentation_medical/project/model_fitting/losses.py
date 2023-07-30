import torch
import segmentation_models_pytorch as smp


class SegmentationLoss(torch.nn.Module):
    def forward(self, output, label):
        fc_loss = smp.losses.LovaszLoss("multiclass")(output, label.argmax(1))
        dice_loss = 0  # DiceLoss()(output, label)
        return fc_loss + dice_loss, fc_loss.item(), 0
