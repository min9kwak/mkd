import torch
import torch.nn as nn
from collections import OrderedDict

from models.slice.backbone import BackboneBase
from utils.initialization import initialize_weights


class DemoEncoder(BackboneBase):

    def __init__(self,
                 in_channels: int,
                 hidden: str) -> None:
        super(DemoEncoder, self).__init__(in_channels)

        self.in_channels = in_channels
        self.hidden = list(int(a) for a in hidden.split(','))

        self.out_features = self.hidden[-1]

        self.layers = None
        self.layers = self.make_layers(self)
        initialize_weights(self.layers)

    @staticmethod
    def make_layers(self):
        layers = nn.Sequential()

        input_layer = nn.Sequential(
            OrderedDict(
                [
                    ("flatten", nn.Flatten()),
                    ("input", nn.Linear(self.in_channels, self.hidden[0])),
                    ("bn", nn.BatchNorm1d(num_features=self.hidden[0])),
                    ("relu", nn.ReLU(inplace=True)),
                ]
            )
        )
        layers.add_module("input", input_layer)

        if len(self.hidden) > 1:
            for i in range(len(self.hidden) - 1):
                block = nn.Sequential(
                    OrderedDict(
                        [
                            ("linear", nn.Linear(self.hidden[i], self.hidden[i + 1])),
                            ("bn", nn.BatchNorm1d(self.hidden[i + 1])),
                            ("relu", nn.ReLU(inplace=True)),
                        ]
                    )
                )
                layers.add_module(f"block{i}", block)
        return layers

    def forward(self, x: torch.FloatTensor):
        return self.layers(x)


class LinearDemoClassifier(BackboneBase):
    def __init__(self,
                 image_dims: int,
                 demo_dims: int,
                 num_classes: int):
        super(LinearDemoClassifier, self).__init__(in_channels=None)

        self.image_dims = image_dims
        self.demo_dims = demo_dims
        self.num_classes = num_classes

        self.classifier = self.make_layers(
            image_dims=self.image_dims,
            demo_dims=self.demo_dims,
            num_classes=self.num_classes,
        )
        initialize_weights(self.classifier)

    @staticmethod
    def make_layers(image_dims: int, demo_dims: int, num_classes: int):

        classifier = nn.Sequential(OrderedDict([
            ('linear', nn.Linear(image_dims + demo_dims, num_classes))
        ]))
        return classifier

    def forward(self, image: torch.Tensor, demo: torch.Tensor):
        h = torch.concat([image, demo], dim=1)
        logit = self.classifier(h)
        return logit

    @property
    def num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == '__main__':
    classifier = LinearDemoClassifier(image_dims=10, demo_dims=5, num_classes=10)

    image = torch.randn(size=(16, 10))
    demo = torch.randn(size=(16, 5))

    logit = classifier(image, demo)
