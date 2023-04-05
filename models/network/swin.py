import collections
from typing import Optional, Sequence, Tuple, Type, Union

from monai.networks.nets import SwinUNETR
import torch.nn as nn


class SwinMRI(SwinUNETR):

    pretrained_config = {
        'base': {'feature_size': 48,
                 'bottleneck_channel': 768},
        'small': {'feature_size': 24,
                  'bottleneck_channel': 384},
        'tiny': {'feature_size': 12,
                 'bottleneck_channel': 192}
    }

    def __init__(self,
                 network_type: str,
                 img_size: Union[Sequence[int], int],
                 in_channels: int,
                 out_channels: int,
                 depths: Sequence[int] = (2, 2, 2, 2),
                 num_heads: Sequence[int] = (3, 6, 12, 24),
                 norm_name: Union[Tuple, str] = "instance",
                 drop_rate: float = 0.0,
                 attn_drop_rate: float = 0.0,
                 dropout_path_rate: float = 0.0,
                 normalize: bool = True,
                 use_checkpoint: bool = False,
                 spatial_dims: int = 3,
                 downsample="merging",
                 ):
        feature_size = self.pretrained_config[network_type]['feature_size']
        super(SwinMRI, self).__init__(img_size, in_channels, out_channels, depths, num_heads, feature_size,
                                      norm_name, drop_rate, attn_drop_rate, dropout_path_rate, normalize,
                                      use_checkpoint, spatial_dims, downsample)
        self.network_type = network_type

    def forward(self, x):
        if self.bottleneck == 'bottleneck':
            logits = self.forward_bottleneck(x)
        elif self.bottleneck == 'last':
            logits = self.forward_last(x)
        return logits

    def _revise_layers(self, bottleneck, num_classes):

        assert bottleneck in ['bottleneck', 'last']
        self.bottleneck = bottleneck
        del self.out

        if self.bottleneck == 'bottleneck':
            # TODO: replace linear classifier with MLP classifier
            self.classifier = nn.Sequential(
                collections.OrderedDict(
                    [
                        ('gap', nn.AdaptiveAvgPool3d(1)),
                        ('flatten', nn.Flatten(1)),
                        ('linear', nn.Linear(self.pretrained_config[self.network_type]['bottleneck_channel'], num_classes))
                    ]
                )
            )
            del self.encoder2, self.encoder3, self.encoder4

        elif self.bottleneck == 'last':
            self.classifier = nn.Sequential(
                collections.OrderedDict(
                    [
                        ('gap', nn.AdaptiveAvgPool3d(1)),
                        ('flatten', nn.Flatten(1)),
                        ('linear', nn.Linear(self.pretrained_config[self.network_type]['feature_size'], num_classes))
                    ]
                )
            )

    def forward_bottleneck(self, x):
        hidden_states_out = self.swinViT(x, self.normalize)
        dec4 = self.encoder10(hidden_states_out[4])
        logits = self.classifier(dec4)
        return logits

    def forward_last(self, x):
        hidden_states_out = self.swinViT(x, self.normalize)
        enc0 = self.encoder1(x)
        enc1 = self.encoder2(hidden_states_out[0])
        enc2 = self.encoder3(hidden_states_out[1])
        enc3 = self.encoder4(hidden_states_out[2])
        dec4 = self.encoder10(hidden_states_out[4])
        dec3 = self.decoder5(dec4, hidden_states_out[3])
        dec2 = self.decoder4(dec3, enc3)
        dec1 = self.decoder3(dec2, enc2)
        dec0 = self.decoder2(dec1, enc1)
        out = self.decoder1(dec0, enc0)
        logits = self.classifier(out)
        return logits



if __name__ == '__main__':

    import torch

    model_size = 'tiny'
    pretrained_path = f'./pretrained/{model_size}_ct.pt'
    model_dict = torch.load(pretrained_path)["state_dict"]

    x = torch.randn(size=(2, 1, 96, 96, 96)).cuda()
    model = SwinMRI(network_type=model_size,
                    img_size=(96, 96, 96),
                    in_channels=1,
                    out_channels=14)
    model.load_state_dict(model_dict)

    model._revise_layers(bottleneck='last', num_classes=2)
    model.cuda()
