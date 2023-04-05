# Copyright 2020 - 2021 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import math
import numpy as np
import torch
import torch.nn as nn
from einops.layers.torch import Rearrange


class PatchEmbeddingBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 img_size: tuple,
                 patch_size: tuple,
                 hidden_size: int,
                 num_heads: int,
                 pos_embed: str,
                 dropout_rate: float = 0.0):
        super().__init__()

        self.pos_embed = pos_embed
        assert 0 <= dropout_rate <= 1
        assert hidden_size % num_heads == 0
        assert pos_embed in ['conv', 'perceptron']
        for i, p in zip(img_size, patch_size):
            assert i % p == 0

        self.n_patches = np.prod([im_d // p_d for im_d, p_d in zip(img_size, patch_size)])
        self.patch_dim = in_channels * np.prod(patch_size)

        self.patch_embeddings: nn.Module
        if self.pos_embed == 'conv':
            self.patch_embeddings = nn.Conv3d(in_channels=in_channels, out_channels=hidden_size,
                                              kernel_size=patch_size, stride=patch_size)
        elif self.pos_embed == 'perceptron':
            chars = (("h", "p1"), ("w", "p2"), ("d", "p3"))
            from_chars = "b c " + " ".join(f"({k} {v})" for k, v in chars)
            to_chars = f"b ({' '.join([c[0] for c in chars])}) ({' '.join([c[1] for c in chars])} c)"
            axes_len = {f"p{i+1}": p for i, p in enumerate(patch_size)}
            self.patch_embeddings = nn.Sequential(
                Rearrange(f"{from_chars} -> {to_chars}", **axes_len), nn.Linear(self.patch_dim, hidden_size)
            )
        self.position_embeddings = nn.Parameter(torch.zeros(1, self.n_patches, hidden_size))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.dropout = nn.Dropout(dropout_rate)
        self.trunc_normal_(self.position_embeddings, mean=0.0, std=0.02, a=-2.0, b=2.0)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            self.trunc_normal_(m.weight, mean=0.0, std=0.02, a=-2.0, b=2.0)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def trunc_normal_(self, tensor, mean, std, a, b):
        # From PyTorch official master until it's in a few official releases - RW
        # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
        def norm_cdf(x):
            return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

        with torch.no_grad():
            l = norm_cdf((a - mean) / std)
            u = norm_cdf((b - mean) / std)
            tensor.uniform_(2 * l - 1, 2 * u - 1)
            tensor.erfinv_()
            tensor.mul_(std * math.sqrt(2.0))
            tensor.add_(mean)
            tensor.clamp_(min=a, max=b)
            return tensor

    def forward(self, x):
        x = self.patch_embeddings(x)
        if self.pos_embed == "conv":
            x = x.flatten(2).transpose(-1, -2)
        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings


if __name__ == '__main__':

    patch_conv = PatchEmbeddingBlock(in_channels=1, img_size=(192, 192, 192), patch_size=(16, 16, 16),
                                     hidden_size=768, num_heads=12, pos_embed='conv', dropout_rate=0.0)
    patch_perceptron = PatchEmbeddingBlock(in_channels=1, img_size=(192, 192, 192), patch_size=(16, 16, 16),
                                           hidden_size=768, num_heads=12, pos_embed='perceptron', dropout_rate=0.0)

    images = torch.randn(size=(8, 1, 192, 192, 192)).cuda()
    patch_conv.cuda()
    patch_perceptron.cuda()

    conv_output = patch_conv(images)
    print('conv - ', conv_output.shape) # 8, 1728, 768

    perceptron_output = patch_perceptron(images)
    print('perceptron - ', perceptron_output.shape) # 8, 1728, 768
