from typing import Sequence, Tuple, Union

from monai.networks.nets import SwinUNETR

import torch
import torch.nn as nn


class SwinEncoder(SwinUNETR):

    pretrained_config = {
        'base': {'feature_size': 48,
                 'bottleneck_channel': 768},
        'small': {'feature_size': 24,
                  'bottleneck_channel': 384},
        'tiny': {'feature_size': 12,
                 'bottleneck_channel': 192}
    }

    def __init__(self,
                 network_type: str = 'small',
                 pretrained_path: str = None,
                 img_size: Union[Sequence[int], int] = (96, 96, 96),
                 in_channels: int = 1,
                 out_channels: int = 14,
                 depths: Sequence[int] = (2, 2, 2, 2),
                 num_heads: Sequence[int] = (3, 6, 12, 24),
                 norm_name: Union[Tuple, str] = "instance",
                 drop_rate: float = 0.0,
                 attn_drop_rate: float = 0.0,
                 dropout_path_rate: float = 0.0,
                 normalize: bool = True,
                 use_checkpoint: bool = False,
                 spatial_dims: int = 3,
                 downsample: str = "merging",
                 feature_layer: str = "last",
                 ):

        assert network_type in self.pretrained_config.keys()
        self.network_type = network_type
        self.feature_size = self.pretrained_config[self.network_type]['feature_size']
        self.bottleneck_channel = self.pretrained_config[self.network_type]['bottleneck_channel']

        assert feature_layer in ['bottleneck', 'last']
        self.feature_layer = feature_layer

        super(SwinEncoder, self).__init__(img_size, in_channels, out_channels, depths, num_heads, self.feature_size,
                                          norm_name, drop_rate, attn_drop_rate, dropout_path_rate, normalize,
                                          use_checkpoint, spatial_dims, downsample)
        if pretrained_path is not None:
            pretrained_weight = torch.load(pretrained_path, map_location='cpu')["state_dict"]
            self.load_state_dict(pretrained_weight)
            print("Load pretrained weights of Swin network")

        del self.out

    def forward(self, x):
        if self.feature_layer == 'bottleneck':
            out = self.forward_bottleneck(x)
        elif self.feature_layer == 'last':
            out = self.forward_last(x)
        else:
            raise ValueError
        return out

    def forward_bottleneck(self, x):
        hidden_states_out = self.swinViT(x, self.normalize)
        dec4 = self.encoder10(hidden_states_out[4])
        return dec4

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
        return out


if __name__ == '__main__':

    network_type = 'tiny'
    pretrained_path = f'D:/data/ADNI/swin/{network_type}_ct.pt'

    x = torch.randn(size=(2, 1, 96, 96, 96)).cuda()
    model = SwinEncoder(network_type=network_type,
                        pretrained_path=pretrained_path,
                        img_size=(96, 96, 96),
                        in_channels=1,
                        out_channels=14,
                        feature_layer='last')
    model.cuda()

    h = model(x)

    print(h.shape)
    # last: [2, 12, 96, 96, 96] / bottleneck: [2, 192, 3, 3, 3]

    print(nn.Flatten(1)(nn.AdaptiveAvgPool3d(1)(h)).shape)
    # last: [2, 12] / bottleneck: [2, 192]
