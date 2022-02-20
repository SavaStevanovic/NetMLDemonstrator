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
        self._features = models.vgg19(
            pretrained=True).eval().features.cuda()
        self._preproces = torchvision.transforms.Compose([
            augmentation.Normalization(
                [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def forward(self, input) -> list:
        input = self._preproces(input)
        output_features = []
        for i, f in enumerate(self._features):
            input = f(input)
            if "Pool" in self._features[i-1].__class__.__name__ and i > 0:
                output_features.append(input)

        return output_features
