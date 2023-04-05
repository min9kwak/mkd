import collections

import torch
import torch.nn as nn

from models.head.base import HeadBase
from utils.initialization import initialize_weights


class PredictorMLP(HeadBase):
    def __init__(self, in_channels: int, num_features: int):
        """
        Arguments:
            in_channels: int, number of input feature maps.
            num_features: int, number of output units.
        """
        super(PredictorMLP, self).__init__()

        self.in_channels = in_channels
        self.num_features = num_features
        self.layers = self.make_layers()
        initialize_weights(self.layers)

    def make_layers(self):

        layers = [
            ('linear1', nn.Linear(self.in_channels, self.in_channels)),
            ('bn1', nn.BatchNorm1d(self.in_channels)),
            ('relu1', nn.ReLU(inplace=True)),
            ('linear2', nn.Linear(self.in_channels, self.num_features))
        ]
        layers = nn.Sequential(collections.OrderedDict(layers))
        return layers

    def forward(self, x: torch.Tensor):
        return self.layers(x)

    @property
    def num_parameters(self):
        return sum(p.numel() for p in self.layers.parameters() if p.requires_grad)
