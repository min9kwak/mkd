import torch
import torch.nn as nn


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


def calculate_out_features(backbone, in_channels, image_size):
    arr = torch.randn(size=(1, in_channels, image_size, image_size, image_size))
    out = backbone(arr)
    return out.shape[1]
