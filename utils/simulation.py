import argparse
import collections
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from easydict import EasyDict as edict
from sklearn.model_selection import train_test_split

from utils.initialization import initialize_weights


# Data Generation
def generate_data(n_train, n_test, x1_dim, x2_dim, x1_common_dim, x2_common_dim, y_dummy, missing_rate, random_state):

    # create y_dummxy
    y_dummy = np.ones(y_dummy)
    y_dummy = y_dummy[0: x1_common_dim + x2_common_dim]

    # x1 and x2 share xs2_dim decisive features. assign binary labels
    x_common = np.random.randn(n_train + n_test, x1_common_dim + x2_common_dim)
    y = (np.dot(x_common, y_dummy) > 0).ravel()
    y = y.astype(int)

    # x1, among all x1_dim channels, xs1_dim+xs2_dim are decisive;
    # among xs1_dim+xs2_dim decisive channels, xs2_dim are shared
    x1 = np.random.randn(n_train + n_test, x1_dim)
    x1[:, 0:x1_common_dim + x2_common_dim] = x_common

    # x2, among all x2_dim channels, xs2_dim are decisive
    x2 = np.random.randn(n_train + n_test, x2_dim)
    x2[:, 0:x2_common_dim] = x_common[:, 0:x2_common_dim]

    # train-test split
    train_index, test_index = train_test_split(range(len(y)), test_size=n_test, random_state=random_state, stratify=y)
    x1_train, x2_train, y_train = x1[train_index, :], x2[train_index, :], y[train_index]
    x1_test, x2_test, y_test = x1[test_index, :], x2[test_index, :], y[test_index]

    n_train = len(x1_train)

    if missing_rate is not None:
        assert 0 < missing_rate < 1
        n_missing = int(n_train * missing_rate)
        missing_index = np.random.choice(np.arange(n_train), n_missing, replace=False)

        x2_train_complete = np.delete(x2_train, missing_index, axis=0)
        x1_train_complete = np.delete(x1_train, missing_index, axis=0)
        x1_train_incomplete = x1_train[missing_index, :]

        y_train_complete = np.delete(y_train, missing_index)
        y_train_incomplete = y_train[missing_index]
    else:
        x1_train_complete = x1_train
        x2_train_complete = x2_train
        y_train_complete = y_train
        x1_train_incomplete, y_train_incomplete = None, None

    data = dict(
        x1_train_complete=x1_train_complete, x2_train_complete=x2_train_complete, y_train_complete=y_train_complete,
        x1_train_incomplete=x1_train_incomplete, y_train_incomplete=y_train_incomplete,
        x1_test=x1_test, x2_test=x2_test, y_test=y_test
    )

    return data


# Dataset
class MultiModalDataset(Dataset):

    def __init__(self, x1, x2, y):
        self.x1 = x1
        self.x2 = x2
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x1 = self.x1[idx]
        if self.x2 is not None:
            x2 = self.x2[idx]
        else:
            x2 = None
        y = self.y[idx]

        return dict(x1=x1, x2=x2, y=y)


# extractor-projector
class Extractor(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(Extractor, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.layers = nn.Sequential(collections.OrderedDict(
            [
                ('linear', nn.Linear(in_channels, out_channels)),
                ('relu', nn.ReLU()),
                # ('norm', nn.LayerNorm(out_channels))
            ]
        ))
        initialize_weights(self.layers)

    def forward(self, x: torch.Tensor):
        out = self.layers(x)
        return out


# encoder (common and specific)
class Encoder(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(Encoder, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.layers = nn.Sequential(collections.OrderedDict(
            [
                ('linear1', nn.Linear(self.in_channels, self.out_channels)),
                # ('norm', nn.LayerNorm(self.out_channels)),
                ('relu', nn.ReLU()),
                ('linear2', nn.Linear(self.out_channels, self.out_channels))
            ]
        ))
        initialize_weights(self.layers)

    def forward(self, x: torch.Tensor):
        out = self.layers(x)
        out = F.normalize(out, p=2, dim=1)
        return out


# decoder
class Decoder(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(Decoder, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.layers = nn.Sequential(collections.OrderedDict(
            [
                ('linear1', nn.Linear(self.in_channels, self.in_channels)),
                ('relu', nn.ReLU()),
                # ('norm', nn.LayerNorm(self.in_channels)),
                ('linear2', nn.Linear(self.in_channels, self.out_channels))
            ]
        ))
        initialize_weights(self.layers)

    def forward(self, x: torch.Tensor):
        out = self.layers(x)
        return out


# classifier
class Classifier(nn.Module):
    def __init__(self, in_channels: int, n_classes: int):
        super(Classifier, self).__init__()
        self.in_channels = in_channels
        self.n_classes = n_classes

        self.layers = nn.Sequential(collections.OrderedDict(
            [
                ('linear', nn.Linear(self.in_channels, self.n_classes))
            ]
        ))
        initialize_weights(self.layers)

    def forward(self, x: torch.Tensor):
        out = self.layers(x)
        return out


def build_general_teacher(config: argparse.Namespace or edict, **kwargs):
    extractor_1 = Extractor(in_channels=config.x1_dim, out_channels=config.hidden)
    extractor_2 = Extractor(in_channels=config.x2_dim, out_channels=config.hidden)

    encoder_1 = Encoder(in_channels=config.hidden, out_channels=config.hidden // 2)
    encoder_2 = Encoder(in_channels=config.hidden, out_channels=config.hidden // 2)
    encoder_general = Encoder(in_channels=config.hidden, out_channels=config.hidden // 2)

    decoder_1 = Decoder(in_channels=config.hidden // 2, out_channels=config.hidden)
    decoder_2 = Decoder(in_channels=config.hidden // 2, out_channels=config.hidden)

    classifier = Classifier(in_channels=config.hidden // 2, n_classes=2)

    networks = dict(extractor_1=extractor_1, extractor_2=extractor_2,
                    encoder_1=encoder_1, encoder_2=encoder_2, encoder_general=encoder_general,
                    decoder_1=decoder_1, decoder_2=decoder_2,
                    classifier=classifier)

    return networks
