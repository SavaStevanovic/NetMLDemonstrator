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
        self.size_scale = 1.25 
        self.offset_scale = 2.25 
        self.class_scale = 1.0
        self.ranges = ranges

    def forward(self, output, label):
        total_objectness_loss = 0.0
        total_size_loss = 0.0
        total_offset_loss = 0.0
        total_class_loss = 0.0
        loss = 0.0

        obj_objectness = output[:, :, self.ranges.objectness]
        lab_objectness = label[:, :, self.ranges.objectness]
        objectness_loss = 1.5 * self.focal_loss(obj_objectness, lab_objectness).sum()
        loss += objectness_loss
        total_objectness_loss += objectness_loss.item()

        obj_box_size = output[:, :, self.ranges.size]
        lab_box_size = label[:, :, self.ranges.size]
        size_loss = self.size_scale * lab_objectness*self.l1_loss(obj_box_size, lab_box_size)
        size_loss = size_loss.sum()
        loss += size_loss
        total_size_loss += size_loss.item()

        obj_offset = output[:, :, self.ranges.offset]
        lab_offset = label[:, :, self.ranges.offset]
        offset_loss = self.offset_scale * lab_objectness*self.l1_loss(obj_offset, lab_offset)
        offset_loss = offset_loss.sum()
        loss += offset_loss
        total_offset_loss += offset_loss.item()

        obj_class = output[:, :, 5:].transpose(1,2)
        lab_class = label[:, :, 5:].transpose(1,2).argmax(1)
        class_loss = self.class_scale * lab_objectness.squeeze(2) * self.class_loss(obj_class, lab_class)
        class_loss = class_loss.sum()
        loss += class_loss
        total_class_loss += class_loss.item()

        batch_size = label.shape[0]
        return loss/batch_size, total_objectness_loss/batch_size, total_size_loss/batch_size, total_offset_loss/batch_size, total_class_loss/batch_size

    def focal_loss(self, x, y):
        gamma = 2
        alpha = 0.25

        y = y.unsqueeze(-1)
        x = x.unsqueeze(-1)

        y = torch.cat([(1-alpha) * y, alpha * (1-y)], -1)
        x = torch.cat([x, 1-x], -1).clamp(1e-8, 1. - 1e-8)

        F_loss = -y * (1 - x)**gamma * torch.log(x)

        return F_loss.sum()

def sigmoid_focal_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2,
    reduction: str = "none",
) -> torch.Tensor:
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.

    Args:
        inputs (Tensor): A float tensor of arbitrary shape.
                The predictions for each example.
        targets (Tensor): A float tensor with the same shape as inputs. Stores the binary
                classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha (float): Weighting factor in range (0,1) to balance
                positive vs negative examples or -1 for ignore. Default: ``0.25``.
        gamma (float): Exponent of the modulating factor (1 - p_t) to
                balance easy vs hard examples. Default: ``2``.
        reduction (string): ``'none'`` | ``'mean'`` | ``'sum'``
                ``'none'``: No reduction will be applied to the output.
                ``'mean'``: The output will be averaged.
                ``'sum'``: The output will be summed. Default: ``'none'``.
    Returns:
        Loss tensor with the reduction option applied.
    """
    # Original implementation from https://github.com/facebookresearch/fvcore/blob/master/fvcore/nn/focal_loss.py
    p = inputs
    ce_loss = F.binary_cross_entropy(inputs, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()

    return loss
