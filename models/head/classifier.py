import collections

import torch
import torch.nn as nn

from models.head.base import HeadBase
from utils.initialization import initialize_weights


class LinearClassifier(HeadBase):
    def __init__(self,
                 in_channels: int,
                 num_classes: int,
                 gap: bool = False):
        """
        Arguments:
            in_channels: int, number of input feature maps.
            num_classes: int, number of output features.
        """
        super(LinearClassifier, self).__init__()

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.gap = gap
        self.layers = self.make_layers()
        initialize_weights(self.layers)

    def make_layers(self):
        layers = [
            ('relu', nn.ReLU(inplace=True)),
            ('linear', nn.Linear(self.in_channels, self.num_classes))
        ]
        if self.gap:
            layers = [('gap', nn.AdaptiveAvgPool3d(1)), ('flatten', nn.Flatten(1))] + layers
        layers = nn.Sequential(collections.OrderedDict(layers))

        return layers

    def forward(self, x: torch.Tensor):
        return self.layers(x)

    @property
    def num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
