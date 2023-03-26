from abc import abstractmethod
import torch
from model import utils
import torchvision.models as models
import torchvision
from data_loader import augmentation


class FeatureExtractor(torch.nn.Module, utils.Identifier):

    @abstractmethod
    def forward(self, input) -> list:
        pass


class VggExtractor(FeatureExtractor):
    def __init__(self) -> None:
        super().__init__()
        model = models.vgg16(
            pretrained=True).eval()
        for param in model.parameters():
            param.requires_grad = False
        self._features = model.cuda().features
        self._preproces = torchvision.transforms.Compose([
            augmentation.Normalization(
                [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def forward(self, input) -> list:
        input = self._preproces(input)
        output_features = []
        for i, f in enumerate(self._features):
            input = f(input)
            layer_num = i-1
            if layer_num < len(self._features) and "Pool" in self._features[layer_num].__class__.__name__:
                output_features.append(input)

        return output_features
