import torch.nn as nn
from model import utils, blocks
import torch
import torch.nn.functional as F
import segmentation_models_pytorch as smp
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import torchvision
import ttach

# from torchvision.models.detection import FasterRCNN
# from torchvision.models.detection.rpn import AnchorGenerator


class FasterRCNN(nn.Module, utils.Identifier):
    def __init__(self, block, inplanes, in_dim, labels, depth=4, norm_layer=None):
        super().__init__()
        model = torchvision.models.detection.maskrcnn_resnet50_fpn_v2(weights="DEFAULT")

        # replace the classifier with a new one, that has
        # num_classes which is user-defined
        num_classes = len(labels)  # 1 class (person) + background
        # get number of input features for the classifier
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        # replace the pre-trained head with a new one
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = 256
        # and replace the mask predictor with a new one
        model.roi_heads.mask_predictor = MaskRCNNPredictor(
            in_features_mask, hidden_layer, num_classes
        )
        self._model = model

    def forward(self, *args):
        return self._model(*args)


class Unet(nn.Module, utils.Identifier):
    def __init__(self, block, inplanes, in_dim, labels, depth=4, norm_layer=None):
        super(Unet, self).__init__()
        self._model = smp.Unet(
            encoder_name="resnet34",
            in_channels=in_dim,
            classes=len(labels),
            encoder_depth=depth,
        )
        # for i, x in enumerate(self._model.encoder.parameters()):
        #     if getattr(x, "shape", [0])[0] == len(labels):
        #         print(
        #             f"stoped freasing at {i} out of {len(list(self._model.parameters()))} layers"
        #         )
        #         break
        #     if hasattr(x, "requires_grad"):
        #         x.requires_grad = False
        self.labels = labels
        self.depth = depth
        self.inplanes = inplanes

    def forward(self, x):
        return self._model(x)

    def unlock_layer(self):
        for i, param in reversed(list(enumerate(self._model.parameters()))):
            if hasattr(param, "requires_grad"):
                if not param.requires_grad:
                    param.requires_grad = True
                    print(f"Layer {i} unlocked")
                    return


class ResNetBackbone(nn.Module, utils.Identifier):
    def __init__(self, block, block_counts, inplanes, norm_layer=nn.InstanceNorm2d):
        super(ResNetBackbone, self).__init__()

        self.feature_count = len(block_counts)
        self.block_counts = block_counts
        self.feature_start_layer = 2
        self.depth = self.feature_start_layer + self.feature_count
        self.block = block
        self._norm_layer = norm_layer
        self.inplanes = inplanes
        self.first_layer = nn.Sequential(
            nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False),
            self._norm_layer(self.inplanes),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )

        self.layers = nn.ModuleList(
            [
                self._make_layer(block, int(i > 0) + 1, layer_count, int(i > 0) + 1)
                for i, layer_count in enumerate(block_counts)
            ]
        )

    def _make_layer(self, block, expansion, blocks, stride=1):
        downsample = None
        outplanes = self.inplanes * expansion
        if stride != 1 or expansion != 1:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes, outplanes, kernel_size=1, stride=stride, bias=False
                ),
                nn.InstanceNorm2d(outplanes),
            )
        layers = []
        layers.append(
            block(
                self.inplanes,
                outplanes,
                stride,
                downsample=downsample,
                norm_layer=self._norm_layer,
            )
        )
        self.inplanes *= expansion
        for _ in range(1, blocks):
            layers.append(block(outplanes, outplanes, norm_layer=self._norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.first_layer(x)

        outputs = [x]
        for l in self.layers:
            outputs.append(l(outputs[-1]))

        return tuple(outputs[1:])


class DeepLabV3Plus(nn.Module, utils.Identifier):
    def __init__(self, backbone, labels):
        super(DeepLabV3Plus, self).__init__()

        self.feature_count = 1
        self.backbone = backbone
        self.depth = self.backbone.depth
        self.feature_start_layer = 1
        self.inplanes = self.backbone.inplanes
        self.out_dim = len(labels)
        self.head = DeepLabHead(self.inplanes, 256)
        self.mid_layer = nn.Conv2d(self.inplanes // 4, 256, kernel_size=1, bias=True)
        self.lateral_layers = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, self.out_dim, kernel_size=1, stride=1),
        )

    def forward(self, x):
        features = self.backbone(x)
        bmid_out = features[-3]
        bmid_out = self.mid_layer(bmid_out)
        boutput = features[-1]
        head = self.head(boutput)
        m_out = F.interpolate(
            head,
            size=[x for x in bmid_out.size()[-2:]],
            mode="bilinear",
            align_corners=True,
        )
        m_out = torch.cat((m_out, bmid_out), dim=1)
        m_out = self.lateral_layers(m_out)
        output = F.interpolate(
            m_out,
            size=[x * 4 for x in bmid_out.size()[-2:]],
            mode="bilinear",
            align_corners=True,
        )
        return output


class DeepLabHead(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(DeepLabHead, self).__init__(
            blocks.ASPP(in_channels, [6, 12, 18]),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, out_channels, 1),
        )
