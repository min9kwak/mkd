from functools import partial
from typing import Any, List, Type, Union

import torch
import torch.nn as nn

from models.backbone.base import BackboneBase
from utils.initialization import initialize_weights


class ResNetBlock(nn.Module):
    expansion = 1
    def __init__(self,
                 in_planes: int,
                 planes: int,
                 stride: int = 1,
                 downsample: Union[nn.Module, partial, None] = None,
                 ) -> None:
        super().__init__()

        BN3d = nn.BatchNorm3d

        self.conv1 = nn.Conv3d(in_planes, planes, kernel_size=(3, 3, 3), stride=(stride, stride, stride),
                               padding=(1, 1, 1), bias=False)
        self.bn1 = BN3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=(3, 3, 3), padding=(1, 1, 1), bias=False)
        self.bn2 = BN3d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out: torch.Tensor = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class ResNetBottleneck(nn.Module):
    expansion = 4
    def __init__(self,
                 in_planes: int,
                 planes: int,
                 stride: int = 1,
                 downsample: Union[nn.Module, partial, None] = None,
                 ) -> None:
        super().__init__()

        BN3d = nn.BatchNorm3d

        self.conv1 = nn.Conv3d(in_planes, planes, kernel_size=(1, 1, 1), bias=False)
        self.bn1 = BN3d(planes)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=(3, 3, 3), stride=(stride, stride, stride),
                               padding=(1, 1, 1), bias=False)
        self.bn2 = BN3d(planes)
        self.conv3 = nn.Conv3d(planes, planes * self.expansion, kernel_size=(1, 1, 1), bias=False)
        self.bn3 = BN3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out: torch.Tensor = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class ResNetBackbone(BackboneBase):
    def __init__(self,
                 block: Type[Union[ResNetBlock, ResNetBottleneck]],
                 layers: List[int],
                 block_inplanes: List[int],
                 in_channels: int = 1,
                 no_max_pool: bool = False,
                 ) -> None:

        super().__init__(in_channels=in_channels)

        BN3d = nn.BatchNorm3d

        self.in_planes = block_inplanes[0]
        self.no_max_pool = no_max_pool

        self.conv1 = nn.Conv3d(in_channels, self.in_planes,
                               kernel_size=(7, 7, 7), stride=(2, 2, 2), padding=(3, 3, 3), bias=False)
        self.bn1 = BN3d(self.in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, block_inplanes[0], layers[0])
        self.layer2 = self._make_layer(block, block_inplanes[1], layers[1], stride=2)
        self.layer3 = self._make_layer(block, block_inplanes[2], layers[2], stride=2)
        self.layer4 = self._make_layer(block, block_inplanes[3], layers[3], stride=2)

        initialize_weights(self)

    def _make_layer(self,
                    block: Type[Union[ResNetBlock, ResNetBottleneck]],
                    planes: int,
                    blocks: int,
                    stride: int = 1,
                    ) -> nn.Sequential:

        BN3d = nn.BatchNorm3d

        downsample: Union[nn.Module, partial, None] = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(self.in_planes, planes * block.expansion, kernel_size=(1, 1, 1),
                          stride=(stride, stride, stride)),
                BN3d(planes * block.expansion),
            )

        layers = [block(in_planes=self.in_planes, planes=planes, stride=stride, downsample=downsample)]

        self.in_planes = planes * block.expansion
        for _i in range(1, blocks):
            layers.append(block(self.in_planes, planes))

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if not self.no_max_pool:
            x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x

    def _fix_first_conv(self):
        conv1 = self.conv1
        self.conv1 = nn.Conv3d(conv1.in_channels, conv1.out_channels,
                               kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False)


def _resnet(
    block: Type[Union[ResNetBlock, ResNetBottleneck]],
    layers: List[int],
    block_inplanes: List[int],
    **kwargs: Any,
) -> ResNetBackbone:
    model: ResNetBackbone = ResNetBackbone(block, layers, block_inplanes, **kwargs)
    return model


def build_resnet_backbone(arch: int = 18, **kwargs):
    if arch == 10:
        return _resnet(ResNetBlock, [1, 1, 1, 1], [64, 128, 256, 512], **kwargs)
    elif arch == 18:
        return _resnet(ResNetBlock, [2, 2, 2, 2], [64, 128, 256, 512], **kwargs)
    elif arch == 34:
        return _resnet(ResNetBlock, [3, 4, 6, 3], [64, 128, 256, 512], **kwargs)
    elif arch == 50:
        return _resnet(ResNetBottleneck, [3, 4, 6, 3], [64, 128, 256, 512], **kwargs)
    elif arch == 101:
        return _resnet(ResNetBottleneck, [3, 4, 23, 3], [64, 128, 256, 512], **kwargs)
    elif arch == 152:
        return _resnet(ResNetBottleneck, [3, 8, 36, 3], [64, 128, 256, 512], **kwargs)
    elif arch == 200:
        return _resnet(ResNetBottleneck, [3, 24, 36, 3], [64, 128, 256, 512], **kwargs)
    else:
        raise NotImplementedError


if __name__ == '__main__':

    import torch
    import torch.nn as nn

    images = torch.randn(size=(3, 1, 36, 36, 36))

    backbone = build_resnet_backbone(arch=50, no_max_pool=False, in_channels=1)

    feature_map = backbone(images)
    print(backbone.conv1)
    print(feature_map.shape)

    print('---------------')

    backbone = build_resnet_backbone(arch=50, no_max_pool=False, in_channels=1)
    feature_map = backbone(images)
    print(backbone.conv1)
    print(feature_map.shape)
