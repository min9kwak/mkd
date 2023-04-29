# -*- coding: utf-8 -*-

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.slice.resnet import resnet18, resnet50, resnet101, wide_resnet50_2, wide_resnet101_2
from torchvision.models.densenet import densenet121, densenet161
from utils.initialization import initialize_weights


BACKBONE_FUNCTIONS = {
    'resnet18': resnet18,
    'resnet50': resnet50,
    'resnet101': resnet101,
    'wide50_2': wide_resnet50_2,
    'wide101_2': wide_resnet101_2,
    'densenet121': densenet121,
    'densenet161': densenet161,
}


class BackboneBase(nn.Module):
    def __init__(self, in_channels: int):
        super(BackboneBase, self).__init__()
        self.in_channels = in_channels

    def forward(self, x: torch.FloatTensor):
        raise NotImplementedError

    def freeze_weights(self):
        for p in self.parameters():
            p.requires_grad = False

    def save_weights_to_checkpoint(self, path: str):
        torch.save(self.state_dict(), path)

    def load_weights_from_checkpoint(self, path: str, key: str):
        ckpt = torch.load(path, map_location='cpu')
        self.load_state_dict(ckpt[key])

    @property
    def num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class DenseNetBackbone(BackboneBase):
    def __init__(self, name: str = 'densenet121', in_channels: int = 1):
        super(DenseNetBackbone, self).__init__(in_channels)

        self.name = name
        self.layers = BACKBONE_FUNCTIONS[self.name]().features

        if self.in_channels != 3:
            conv0 = self.layers.conv0
            self.layers.conv0 = nn.Conv2d(in_channels=self.in_channels,
                                          out_channels=conv0.out_channels,
                                          kernel_size=(7, 7),
                                          stride=(2, 2),
                                          padding=(3, 3),
                                          bias=False)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        return self.layers(x)

    @property
    def out_channels(self):
        if self.name == 'densenet121':
            return 1024
        elif self.name == 'densenet161':
            return 2208
        else:
            raise ValueError

    def _fix_first_conv(self):
        conv0 = self.layers.conv0
        self.layers.conv0 = nn.Conv2d(conv0.in_channels, conv0.out_channels,
                                      kernel_size=3, stride=1, padding=1, bias=False)


class ResNetBackbone(BackboneBase):
    def __init__(self, name: str = 'resnet50', in_channels: int = 1):
        super(ResNetBackbone, self).__init__(in_channels)

        self.name = name  # resnet18, resnet50, resnet101
        self.layers = BACKBONE_FUNCTIONS[self.name]()

        self.layers = self._remove_gap_and_fc(self.layers)
        if self.in_channels != 3:
            self.layers = self._fix_first_conv_in_channels(self.layers, in_channels=self.in_channels)
        initialize_weights(self)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        return self.layers(x)

    @staticmethod
    def _fix_first_conv_in_channels(resnet: nn.Module, in_channels: int) -> nn.Module:
        """
        Change the number of incoming channels for the first layer.
        """
        model = nn.Sequential()
        for name, child in resnet.named_children():
            if name == 'conv1':
                conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
                model.add_module(name, conv1)
            else:
                model.add_module(name, child)

        return model

    @staticmethod
    def _remove_gap_and_fc(resnet: nn.Module) -> nn.Module:
        """
        Remove global average pooling & fully-connected layer
        For torchvision ResNet models only."""
        model = nn.Sequential()
        for name, child in resnet.named_children():
            if name not in ['avgpool', 'fc']:
                model.add_module(name, child)  # preserve original names

        return model

    def _fix_first_conv(self):
        conv1 = self.layers.conv1
        self.layers.conv1 = nn.Conv2d(conv1.in_channels, conv1.out_channels,
                                      kernel_size=3, stride=1, padding=1, bias=False)

    @staticmethod
    def _remove_maxpool(resnet: nn.Module):
        """Remove first max pooling layer of ResNet."""
        model = nn.Sequential()
        for name, child in resnet.named_children():
            if name == 'maxpool':
                continue
            else:
                model.add_module(name, child)

        return model

    @property
    def out_channels(self):
        if self.name == 'resnet18':
            return 512
        else:
            return 2048


def relu_inplace(module, flag=False):
    for attr_str in dir(module):
        target_attr = getattr(module, attr_str)
        if isinstance(target_attr, nn.ReLU):
            setattr(module, attr_str, nn.ReLU(inplace=flag))
    for n, ch in module.named_children():
        relu_inplace(ch, flag)


if __name__ == '__main__':

    from models.slice.head import GAPLinearClassifier

    x = torch.randn(8, 1, 64, 64)

    backbone = ResNetBackbone(name='resnet50')
    classifier = GAPLinearClassifier(name='resnet50', in_channels=backbone.out_channels, n_classes=2)
    out = backbone(x)
    print(out.shape)
    out = classifier(out)
    print(out.shape)

    backbone = DenseNetBackbone(name='densenet121')
    classifier = GAPLinearClassifier(name='densenet121', in_channels=backbone.out_channels, n_classes=2)
    out = backbone(x)
    print(out.shape)
    out = classifier(out)
    print(out.shape)
