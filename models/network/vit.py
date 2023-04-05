# https://github.com/Project-MONAI/MONAI/blob/dev/monai/networks/nets/vit.py

import torch
import torch.nn as nn

from layers.patch import PatchEmbeddingBlock
from layers.transformer import TransformerBlock


# Uni-modal ViT
class UnimodalViT(nn.Module):
    def __init__(self,
                 in_channels: int,
                 img_size: tuple,
                 patch_size: tuple,
                 hidden_size: int = 768,
                 mlp_dim: int = 3072,
                 num_layers: int = 12,
                 num_heads: int = 12,
                 pos_embed: str = "conv",
                 num_classes: int = 2,
                 dropout_rate: float = 0.0):
        super().__init__()

        assert 0 <= dropout_rate <= 1
        assert hidden_size % num_heads == 0

        self.patch_embedding = PatchEmbeddingBlock(
            in_channels=in_channels,
            img_size=img_size,
            patch_size=patch_size,
            hidden_size=hidden_size,
            num_heads=num_heads,
            pos_embed=pos_embed,
            dropout_rate=dropout_rate
        )
        self.blocks = nn.ModuleList(
            [TransformerBlock(hidden_size, mlp_dim, num_heads, dropout_rate) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(hidden_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
        # self.classification_head = nn.Sequential(nn.Linear(hidden_size, num_classes), nn.Tanh())
        self.classification_head = nn.Linear(hidden_size, num_classes)
        self.apply(self.initialize)

    def initialize(self, module):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, x):
        x = self.patch_embedding(x)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)

        # to save memory
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        x = self.classification_head(x[:, 0])
        return x


if __name__ == '__main__':

    images = torch.randn(size=(4, 1, 192, 192, 192)).cuda()

    vit = UnimodalViT(in_channels=1, img_size=(192, 192, 192), patch_size=(24, 24, 24),
                      hidden_size=512, mlp_dim=2048, num_layers=8, num_heads=8,
                      pos_embed='conv', num_classes=2, dropout_rate=0.0)
    vit.cuda()
    out, hidden_out = vit(images)

    import time
    s = time.time()
    for _ in range(50):
        out, _ = vit(images)
    e = time.time()
    print(e - s)

    # out - (B, num_classe)
    # hidden_out = list of num_layers elements (B, 1 + patch_dim, hidden_size)
    # patch_dim = (img_size / patch_size) ** 3
