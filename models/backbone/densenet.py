import torch
import torch.nn as nn
from collections import OrderedDict

from models.backbone.base import BackboneBase
from utils.semi import SemiBatchNorm3d
from utils.initialization import initialize_weights


class DenseNetBackbone(BackboneBase):
    def __init__(
        self,
        in_channels: int,
        init_features: int = 64,
        growth_rate: int = 32,
        block_config: tuple = (6, 12, 24, 16),
        bn_size: int = 4,
        dropout_rate: float = 0.0
    ) -> None:

        super().__init__(in_channels=in_channels)

        BN3d = nn.BatchNorm3d

        self.features = nn.Sequential(
            OrderedDict(
                [
                    ("conv0", nn.Conv3d(in_channels, init_features, kernel_size=(7, 7, 7),
                                        stride=(2, 2, 2), padding=(3, 3, 3), bias=False)),
                    ("norm0", BN3d(init_features)),
                    ("relu0", nn.ReLU(inplace=True)),
                    ("pool0", nn.MaxPool3d(kernel_size=3, stride=2, padding=1)),
                ]
            )
        )

        in_channels = init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(
                layers=num_layers,
                in_channels=in_channels,
                bn_size=bn_size,
                growth_rate=growth_rate,
                dropout_rate=dropout_rate,
                semi=self.semi
            )
            self.features.add_module(f"denseblock{i + 1}", block)
            in_channels += num_layers * growth_rate
            if i == len(block_config) - 1:
                self.features.add_module(
                    "norm5", BN3d(in_channels)
                )
            else:
                _out_channels = in_channels // 2
                trans = _Transition(in_channels=in_channels, out_channels=_out_channels, semi=self.semi)
                self.features.add_module(f"transition{i + 1}", trans)
                in_channels = _out_channels

        initialize_weights(self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return x

    def _fix_first_conv(self):
        conv0 = self.features.conv0
        self.features.conv0 = nn.Conv3d(conv0.in_channels, conv0.out_channels,
                                        kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False)


class _DenseLayer(nn.Module):
    def __init__(self,
                 in_channels: int,
                 growth_rate: int,
                 bn_size: int,
                 dropout_rate: float,
                 ):
        super().__init__()

        BN3d = nn.BatchNorm3d

        out_channels = bn_size * growth_rate

        self.layers = nn.Sequential()

        self.layers.add_module("norm1", BN3d(in_channels))
        self.layers.add_module("relu1", nn.ReLU(inplace=True))
        self.layers.add_module("conv1", nn.Conv3d(in_channels, out_channels, kernel_size=(1, 1, 1), bias=False))

        self.layers.add_module("norm2", BN3d(out_channels))
        self.layers.add_module("relu2", nn.ReLU(inplace=True))
        self.layers.add_module("conv2", nn.Conv3d(out_channels, growth_rate, kernel_size=(3, 3, 3),
                                                  padding=(1, 1, 1), bias=False))

        if dropout_rate > 0:
            self.layers.add_module("dropout", nn.Dropout3d(dropout_rate))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        new_features = self.layers(x)
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):
    def __init__(
        self,
        layers: int,
        in_channels: int,
        bn_size: int,
        growth_rate: int,
        dropout_rate: float,
    ) -> None:
        super().__init__()
        for i in range(layers):
            layer = _DenseLayer(in_channels, growth_rate, bn_size, dropout_rate)
            in_channels += growth_rate
            self.add_module("denselayer%d" % (i + 1), layer)


class _Transition(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ) -> None:
        super().__init__()

        BN3d = nn.BatchNorm3d

        self.add_module("norm", BN3d(in_channels))
        self.add_module("relu", nn.ReLU(inplace=True))
        self.add_module("conv", nn.Conv3d(in_channels, out_channels, kernel_size=(1, 1, 1), bias=False))
        self.add_module("pool", nn.AvgPool3d(kernel_size=2, stride=2))


if __name__ == '__main__':

    import torch

    network = DenseNetBackbone(in_channels=1,
                               init_features=64,
                               growth_rate=32,
                               block_config=(6, 12, 24, 16),
                               bn_size=4,
                               dropout_rate=0.0)
    x = torch.randn(size=(3, 1, 96, 96, 96))

    with torch.no_grad():
        features = network(x)
        print(features.shape)

    network._fix_first_conv_kernel_size()
    x = torch.randn(size=(3, 1, 32, 32, 32))

    with torch.no_grad():
        features = network(x)
        print(features.shape)
