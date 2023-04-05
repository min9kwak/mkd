
# -*- coding: utf-8 -*-

import torch.nn as nn


def initialize_weights(module: nn.Module, activation: str = 'relu'):
    """Initialize trainable weights."""

    for name, module_ in module.named_children():
        if isinstance(module_, (nn.Conv3d, nn.ConvTranspose3d)):
            nn.init.kaiming_normal_(module_.weight, mode='fan_out', nonlinearity=activation)
            if module_.bias is not None:
                nn.init.constant_(module_.bias, 0)

        elif isinstance(module_, nn.BatchNorm3d):
            nn.init.constant_(module_.weight, 1)
            try:
                nn.init.constant_(module_.bias, 0)
            except AttributeError:
                pass

        elif isinstance(module_, nn.Linear):
            nn.init.normal_(module_.weight, 0, 0.02)
            try:
                nn.init.constant_(module_.bias, 0)
            except AttributeError:
                pass

    for n, ch in module.named_children():
        initialize_weights(ch, activation)
