import torch.nn.functional as F
import torch


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
        alpha = .60
        eps = 1e-8

        F_loss = -alpha * torch.pow((1. - probs), gamma) * target * torch.log(probs + eps) \
            - (1. - alpha) * torch.pow(probs, gamma) * \
            (1. - target) * torch.log(1. - probs + eps)

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


class ContentLoss(torch.nn.Module):
    def __init__(self, target_feature: torch.Tensor):
        super(ContentLoss, self).__init__()
        # we 'detach' the target content from the tree used
        # to dynamically compute the gradient: this is a stated value,
        # not a variable. Otherwise the forward method of the criterion
        # will throw an error.
        self._target_feature = target_feature.detach()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(input, self._target_feature)


class StyleLoss(torch.nn.Module):
    def __init__(self, target_feature: torch.Tensor):
        super(StyleLoss, self).__init__()
        self._target_feature = StyleLoss._gram_matrix(target_feature).detach()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        G = StyleLoss._gram_matrix(input)
        return F.mse_loss(G, torch.cat([self._target_feature]*len(G)))

    @staticmethod
    def _gram_matrix(input: torch.Tensor) -> torch.Tensor:
        B, C, H, W = input.shape
        x = input.view(B, C, H*W)
        x_t = x.transpose(1, 2)
        return torch.bmm(x, x_t) / (C*H*W)
