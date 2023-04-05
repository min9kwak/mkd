import torch
import torch.nn as nn
from collections import OrderedDict

class UnimodalDenseNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        init_features: int = 64,
        growth_rate: int = 32,
        block_config: tuple = (6, 12, 24, 16),
        bn_size: int = 4,
        dropout_rate: float = 0.0,
    ) -> None:

        super().__init__()
        self.features = nn.Sequential(
            OrderedDict(
                [
                    ("conv0", nn.Conv3d(in_channels, init_features, kernel_size=7, stride=2, padding=3, bias=False)),
                    ("norm0", nn.BatchNorm3d(num_features=init_features)),
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
            )
            self.features.add_module(f"denseblock{i + 1}", block)
            in_channels += num_layers * growth_rate
            if i == len(block_config) - 1:
                self.features.add_module(
                    "norm5", nn.BatchNorm3d(num_features=in_channels)
                )
            else:
                _out_channels = in_channels // 2
                trans = _Transition(in_channels=in_channels, out_channels=_out_channels)
                self.features.add_module(f"transition{i + 1}", trans)
                in_channels = _out_channels

        # pooling and classification
        self.class_layers = nn.Sequential(
            OrderedDict(
                [
                    ("relu", nn.ReLU(inplace=True)),
                    ("pool", nn.AdaptiveAvgPool3d(1)),
                    ("flatten", nn.Flatten(1)),
                    ("out", nn.Linear(in_channels, out_channels)),
                ]
            )
        )

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(torch.as_tensor(m.weight))
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                nn.init.constant_(torch.as_tensor(m.weight), 1)
                nn.init.constant_(torch.as_tensor(m.bias), 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(torch.as_tensor(m.bias), 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.class_layers(x)
        return x


class _DenseLayer(nn.Module):
    def __init__(self,
                 in_channels: int,
                 growth_rate: int,
                 bn_size: int,
                 dropout_rate: float):
        super().__init__()

        out_channels = bn_size * growth_rate

        self.layers = nn.Sequential()

        self.layers.add_module("norm1", nn.BatchNorm3d(num_features=in_channels))
        self.layers.add_module("relu1", nn.ReLU(inplace=True))
        self.layers.add_module("conv1", nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False))

        self.layers.add_module("norm2", nn.BatchNorm3d(num_features=out_channels))
        self.layers.add_module("relu2", nn.ReLU(inplace=True))
        self.layers.add_module("conv2", nn.Conv3d(out_channels, growth_rate, kernel_size=3, padding=1, bias=False))

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
        dropout_rate: float
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

        self.add_module("norm", nn.BatchNorm3d(num_features=in_channels))
        self.add_module("relu", nn.ReLU(inplace=True))
        self.add_module("conv", nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False))
        self.add_module("pool", nn.AvgPool3d(kernel_size=2, stride=2))


if __name__ == '__main__':

    import torch

    network = UnimodalDenseNet(in_channels=1, out_channels=2)
    x = torch.randn(size=(3, 1, 96, 96, 96))
    logits = network(x)

    network = UnimodalDenseNet(in_channels=1, out_channels=2, init_features=32, growth_rate=16)
    x = torch.randn(size=(3, 1, 96, 96, 96))
    logits = network(x)

