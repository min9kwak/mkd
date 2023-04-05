import torch
import torch.nn as nn
import einops


class TransformerBlock(nn.Module):
    def __init__(self,
                 hidden_size: int,
                 mlp_dim: int,
                 num_heads: int,
                 dropout_rate: float = 0.0):
        super().__init__()
        assert 0 <= dropout_rate <= 1
        assert hidden_size % num_heads == 0

        self.mlp = MLPBlock(hidden_size, mlp_dim, dropout_rate)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.attn = SelfAttentionBlock(hidden_size, num_heads, dropout_rate)
        self.norm2 = nn.LayerNorm(hidden_size)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class MLPBlock(nn.Module):
    def __init__(self, hidden_size: int, mlp_dim: int, dropout_rate: float = 0.0):
        super().__init__()
        assert 0 <= dropout_rate <= 1

        self.linear1 = nn.Linear(hidden_size, mlp_dim)
        self.linear2 = nn.Linear(mlp_dim, hidden_size)
        self.gelu = nn.GELU()
        self.drop1 = nn.Dropout(dropout_rate)
        self.drop2 = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.gelu(self.linear1(x))
        x = self.drop1(x)
        x = self.linear2(x)
        x = self.drop2(x)
        return x


class SelfAttentionBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, dropout_rate: float = 0.0):
        super().__init__()
        assert 0 <= dropout_rate <= 1
        assert hidden_size % num_heads == 0

        self.num_heads = num_heads
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.qkv = nn.Linear(hidden_size, hidden_size * 3, bias=False)
        self.drop_output = nn.Dropout(dropout_rate)
        self.drop_weights = nn.Dropout(dropout_rate)
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim ** -0.5

    def forward(self, x):
        q, k, v = einops.rearrange(self.qkv(x), "b h (qkv l d) -> qkv b l h d", qkv=3, l=self.num_heads)
        att_mat = (torch.einsum("blxd, blyd -> blxy", q, k) * self.scale).softmax(dim=-1)
        att_mat = self.drop_weights(att_mat)
        x = torch.einsum("bhxy, bhyd -> bhxd", att_mat, v)
        x = einops.rearrange(x, "b h l d -> b l (h d)")
        x = self.out_proj(x)
        x = self.drop_output(x)
        return x


if __name__ == '__main__':

    from layers.patch import PatchEmbeddingBlock

    images = torch.randn(size=(8, 1, 192, 192, 192)).cuda()

    # patch - conv
    patch_conv = PatchEmbeddingBlock(in_channels=1, img_size=(192, 192, 192), patch_size=(16, 16, 16),
                                     hidden_size=768, num_heads=12, pos_embed='conv', dropout_rate=0.0)
    patch_conv.cuda()
    conv_output = patch_conv(images)
    print('conv - ', conv_output.shape)  # 8, 1728, 768

    transformer_block = TransformerBlock(hidden_size=768, mlp_dim=3072, num_heads=12, dropout_rate=0.0)
    transformer_block.cuda()

    transformer_out = transformer_block(conv_output)
    print('conv-transformer - ', transformer_out.shape)

    # patch - perceptron
    patch_perceptron = PatchEmbeddingBlock(in_channels=1, img_size=(192, 192, 192), patch_size=(16, 16, 16),
                                           hidden_size=768, num_heads=12, pos_embed='perceptron', dropout_rate=0.0)
    patch_perceptron.cuda()
    perceptron_output = patch_perceptron(images)
    print('perceptron - ', perceptron_output.shape) # 8, 1728, 768

    transformer_out = transformer_block(perceptron_output)
    print('perceptron-transformer - ', transformer_out.shape) # B, mlp_dim, hidden_size
