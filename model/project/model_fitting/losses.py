import torch.nn.functional as F
import torch
from sklearn.metrics import f1_score
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

    def forward(self, outputs, labels):
        loss = 0
        for batch in range(labels.size()[0]):
            output, label = outputs[batch], labels[batch]
            obj_objectness = torch.sigmoid(output[::(5*len(self.ratios)+self.classes_len)])
            lab_objectness = label[::(5*len(self.ratios)+self.classes_len)]
            obj_objectness = torch.flatten(obj_objectness)
            lab_objectness = torch.flatten(lab_objectness)
            # loss += torch.sum(lab_objectness * (lab_objectness - obj_objectness) ** 2)
            # loss += torch.sum(0.5 * (1 - lab_objectness) * (lab_objectness - obj_objectness) ** 2)
            # loss += (1-obj_objectness).dot(F.binary_cross_entropy(obj_objectness, lab_objectness.type(torch.cuda.FloatTensor), reduction='none'))/len(obj_objectness)
            # loss += F.binary_cross_entropy(obj_objectness, lab_objectness.type(torch.cuda.FloatTensor))
            loss+= self.focal_loss(obj_objectness, lab_objectness)
        return loss, f1_score(lab_objectness.cpu(), obj_objectness.cpu()>0.5)

    def focal_loss(self, x, y):
        '''Focal loss.
        Args:
          x: (tensor) sized [N,D].
          y: (tensor) sized [N,].
        Return:
          (tensor) focal loss.
        '''
        alpha = 1
        gamma = 1
        y = y.unsqueeze(1)
        x = x.unsqueeze(1)

        y = torch.cat([y, 1-y], 1)
        # t = t.cuda() 
        x = torch.cat([x, 1-x], 1).clamp(1e-8, 1. - 1e-8)

        F_loss = -y * (1 - x) * torch.log(x)

        return F_loss.sum()