# -*- coding: utf-8 -*-

import math
import collections

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.initialization import initialize_weights


class HeadBase(nn.Module):
    def __init__(self, output_size: int):
        super(HeadBase, self).__init__()
        assert isinstance(output_size, int)

    def freeze_weights(self):
        for p in self.parameters():
            p.requires_grad = False

    def save_weights(self, path: str):
        """Save weights to a file with weights only."""
        torch.save(self.state_dict(), path)

    def load_weights(self, path: str):
        """Load weights from a file with weights only."""
        self.load_state_dict(torch.load(path))

    def load_weights_from_checkpoint(self, path: str, key: str):
        """
        Load weights from a checkpoint.
        Arguments:
            path: str, path to pretrained `.pt` file.
            key: str, key to retrieve the model from a state dictionary of pretrained modules.
        """
        ckpt = torch.load(path, map_location='cpu')
        self.load_state_dict(ckpt[key])


class GAPHeadBase(HeadBase):
    def __init__(self, in_channels: int, output_size: int):
        super(GAPHeadBase, self).__init__(output_size)
        assert isinstance(in_channels, int), "Number of output feature maps of backbone."


class LinearHeadBase(HeadBase):
    def __init__(self, in_channels: int, output_size: int):
        super(LinearHeadBase, self).__init__(output_size)
        assert isinstance(in_channels, int), "Number of output feature maps of backbone."


class GAPLinearClassifier(GAPHeadBase):
    def __init__(self, name: str, in_channels: int, n_classes: int, dropout: float = 0.0):
        """
        Arguments:
            in_channels: int, number of input feature maps.
            n_classes: int, number of output features.
        """
        super(GAPLinearClassifier, self).__init__(in_channels, n_classes)

        self.name = name
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.dropout = dropout
        self.layers = self.make_layers(
            name=self.name,
            in_channels=self.in_channels,
            n_classes=self.n_classes,
            dropout=self.dropout,
        )
        initialize_weights(self.layers)

    @staticmethod
    def make_layers(name: str, in_channels: int, n_classes: int, dropout: float = 0.0):
        if 'resnet' in name:
            layers = nn.Sequential(
                collections.OrderedDict(
                    [
                        ('gap', nn.AdaptiveAvgPool2d(1)),
                        ('flatten', nn.Flatten(1)),
                        ('dropout', nn.Dropout(p=dropout)),
                        ('linear', nn.Linear(in_channels, n_classes))
                    ]
                )
            )
        elif 'densenet' in name:
            layers = nn.Sequential(
                collections.OrderedDict(
                    [
                        ('relu', nn.ReLU(inplace=False)),
                        ('gap', nn.AdaptiveAvgPool2d(1)),
                        ('flatten', nn.Flatten(1)),
                        ('dropout', nn.Dropout(p=dropout)),
                        ('linear', nn.Linear(in_channels, n_classes))
                    ]
                )
            )
        else:
            raise ValueError

        return layers

    def forward(self, x: torch.Tensor):
        return self.layers(x)

    @property
    def num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class GAPLinearProjector(GAPHeadBase):
    # https://github.com/declare-lab/MISA/blob/master/src/models.py
    def __init__(self, name: str, in_channels: int, out_channels: int):
        super(GAPLinearProjector, self).__init__(in_channels, out_channels)

        self.name = name
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.layers = self.make_layers(self.name, self.in_channels, self.out_channels)
        initialize_weights(self.layers)

    @staticmethod
    def make_layers(name: str, in_channels: int, out_channels: int):
        if 'resnet' in name:
            layers = nn.Sequential(
                collections.OrderedDict(
                    [
                        ('gap', nn.AdaptiveAvgPool2d(1)),
                        ('flatten', nn.Flatten(1)),
                        ('linear', nn.Linear(in_channels, out_channels)),
                        ('relu', nn.ReLU(inplace=False)),
                        ('norm', nn.LayerNorm(out_channels))
                    ]
                )
            )
        elif 'densenet' in name:
            layers = nn.Sequential(
                collections.OrderedDict(
                    [
                        ('relu1', nn.ReLU(inplace=False)),
                        ('gap', nn.AdaptiveAvgPool2d(1)),
                        ('flatten', nn.Flatten(1)),
                        ('linear', nn.Linear(in_channels, out_channels)),
                        ('relu2', nn.ReLU(inplace=False)),
                        ('norm', nn.LayerNorm(out_channels))
                    ]
                )
            )
        else:
            raise ValueError

        return layers

    def forward(self, x: torch.Tensor):
        return self.layers(x)


class LinearEncoder(LinearHeadBase):
    def __init__(self, in_channels: int, out_channels: int):
        super(LinearEncoder, self).__init__(in_channels, out_channels)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.layers = nn.Sequential(
            collections.OrderedDict(
                [
                    ('linear', nn.Linear(self.in_channels, self.out_channels)),
                    ('sigmoid', nn.Sigmoid())
                ]
            )
        )
        initialize_weights(self.layers)

    def forward(self, x: torch.Tensor):
        return self.layers(x)


class LinearDecoder(LinearHeadBase):
    def __init__(self, in_channels: int, out_channels: int):
        super(LinearDecoder, self).__init__(in_channels, out_channels)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.layers = nn.Sequential(
            collections.OrderedDict(
                [
                    ('linear', nn.Linear(self.in_channels, self.out_channels))
                ]
            )
        )
        initialize_weights(self.layers)

    def forward(self, x: torch.Tensor):
        return self.layers(x)


class Classifier(LinearHeadBase):
    def __init__(self, in_channels: int, n_classes: int, mlp: bool = False, dropout: float = 0.0):
        super(Classifier, self).__init__(in_channels, n_classes)
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.mlp = mlp
        self.dropout = dropout

        if self.mlp:
            self.layers = nn.Sequential(
                collections.OrderedDict(
                    [
                        ('linear1', nn.Linear(self.in_channels, self.in_channels // 2)),
                        ('dropout', nn.Dropout(p=self.dropout)),
                        ('relu', nn.ReLU(inplace=False)),
                        ('linear2', nn.Linear(self.in_channels // 2, self.n_classes))
                    ]
                )
            )
        else:
            self.layers = nn.Sequential(
                collections.OrderedDict(
                    [
                        ('dropout', nn.Dropout(p=self.dropout)),
                        ('linear', nn.Linear(self.in_channels, self.n_classes))
                    ]
                )
            )
        initialize_weights(self.layers)

    def forward(self, x: torch.Tensor):
        return self.layers(x)


class TransformerEncoder(HeadBase):
    def __init__(self, in_channels: int):
        super(TransformerEncoder, self).__init__(in_channels)
        self.in_channels = in_channels

        encoder_layer = nn.TransformerEncoderLayer(d_model=self.in_channels, nhead=2)
        self.layers = nn.TransformerEncoder(encoder_layer, num_layers=1)
        initialize_weights(self.layers)

    def forward(self, x: torch.Tensor):
        return self.layers(x)
