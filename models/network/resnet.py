from functools import partial
from typing import Any, Callable, List, Optional, Type, Union

import torch
import torch.nn as nn

# TODO: chage to semi resnet using previous implementation or use Batchnorm -> SemiBN
class ResNetBlock(nn.Module):
    expansion = 1
    def __init__(self,
                 in_planes: int,
                 planes: int,
                 stride: int = 1,
                 downsample: Union[nn.Module, partial, None] = None,
                 ) -> None:
        super().__init__()

        self.conv1 = nn.Conv3d(in_planes, planes, kernel_size=3, padding=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
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

        self.conv1 = nn.Conv3d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
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


class ResNet(nn.Module):
    def __init__(self,
                 block: Type[Union[ResNetBlock, ResNetBottleneck]],
                 layers: List[int],
                 block_inplanes: List[int],
                 in_channels: int = 3,
                 no_max_pool: bool = False,
                 num_classes: int = 400) -> None:

        super().__init__()

        self.in_planes = block_inplanes[0]
        self.no_max_pool = no_max_pool

        self.conv1 = nn.Conv3d(in_channels, self.in_planes,
                               kernel_size=(7, 7, 7), stride=(2, 2, 2), padding=(3, 3, 3), bias=False)
        self.bn1 = nn.BatchNorm3d(self.in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, block_inplanes[0], layers[0])
        self.layer2 = self._make_layer(block, block_inplanes[1], layers[1], stride=2)
        self.layer3 = self._make_layer(block, block_inplanes[2], layers[2], stride=2)
        self.layer4 = self._make_layer(block, block_inplanes[3], layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(block_inplanes[3] * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(torch.as_tensor(m.weight), mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(torch.as_tensor(m.weight), 1)
                nn.init.constant_(torch.as_tensor(m.bias), 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(torch.as_tensor(m.bias), 0)

    def _make_layer(self,
                    block: Type[Union[ResNetBlock, ResNetBottleneck]],
                    planes: int,
                    blocks: int,
                    stride: int = 1
                    ) -> nn.Sequential:

        downsample: Union[nn.Module, partial, None] = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(self.in_planes, planes * block.expansion, kernel_size=1, stride=stride),
                nn.BatchNorm3d(planes * block.expansion),
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

        x = self.avgpool(x)

        x = x.view(x.size(0), -1)
        if self.fc is not None:
            x = self.fc(x)

        return x


def _resnet(
    block: Type[Union[ResNetBlock, ResNetBottleneck]],
    layers: List[int],
    block_inplanes: List[int],
    **kwargs: Any,
) -> ResNet:
    model: ResNet = ResNet(block, layers, block_inplanes, **kwargs)
    return model


def build_unimodal_resnet(arch: int = 18, **kwargs):
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
