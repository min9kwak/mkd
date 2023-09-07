import argparse
import collections
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from easydict import EasyDict as edict

from utils.initialization import initialize_weights


# Data Generation
def set_random_state(random_state=2021):
    random.seed(random_state)
    np.random.seed(random_state)
    torch.manual_seed(random_state)
    torch.cuda.manual_seed(random_state)
    torch.cuda.manual_seed_all(random_state)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_dataset(n_train=200, n_test=1000,
                   x1_dim=25, x2_dim=50,
                   xs1_dim=10, xs2_dim=10, overlap_dim=10,
                   hyperplane_dim=500,
                   missing_rate=0.5,
                   mm_mode='increase_gamma',
                   missing_mode='add',
                   random_state=2021):

    # https://github.com/zihuixue/MFH/blob/main/gauss/main.py - exp1
    # when gamma is large, multimodal data share many label-relevant information
    set_random_state(random_state=random_state)
    hyperplane = np.random.randn(hyperplane_dim)

    # generate data - train
    x1_train_complete, x2_train_complete, y_train_complete, x1_train_incomplete, y_train_incomplete = \
        generate_mm_data(n_samples=n_train,
                         x1_dim=x1_dim, x2_dim=x2_dim,
                         xs1_dim=xs1_dim, xs2_dim=xs2_dim, overlap_dim=overlap_dim, hyperplane=hyperplane,
                         missing_rate=missing_rate, mm_mode=mm_mode, missing_mode=missing_mode)

    # generate data - test
    set_random_state(random_state=random_state + 365)
    x1_test, x2_test, y_test, _, _ = \
        generate_mm_data(n_samples=n_test,
                         x1_dim=x1_dim, x2_dim=x2_dim,
                         xs1_dim=xs1_dim, xs2_dim=xs2_dim, overlap_dim=overlap_dim, hyperplane=hyperplane,
                         missing_rate=None, mm_mode=mm_mode, missing_mode=missing_mode)

    data = dict(
        x1_train_complete=x1_train_complete, x2_train_complete=x2_train_complete, y_train_complete=y_train_complete,
        x1_train_incomplete=x1_train_incomplete, y_train_incomplete=y_train_incomplete,
        x1_test=x1_test, x2_test=x2_test, y_test=y_test
    )

    return data


def generate_mm_data(n_samples, x1_dim, x2_dim, xs1_dim, xs2_dim, overlap_dim, hyperplane, missing_rate,
                     mm_mode, missing_mode):

    def generate_mm_data_(n_samples, x1_dim, x2_dim, xs1_dim, xs2_dim, overlap_dim, hyperplane, mm_mode):

        if mm_mode == 'increase_gamma':
            # decisive features
            xs = np.random.randn(n_samples, xs1_dim + xs2_dim)

            # separating hyperplane
            hyperplane = hyperplane[0:xs1_dim + xs2_dim]

            # decisive featuers xs -> label y
            y = (np.dot(xs, hyperplane) > 0).ravel()

            # x2, 0:xs2_dim are decisive features, others gaussian noise
            x2 = np.random.randn(n_samples, x2_dim)
            x2[:, 0:xs2_dim] = xs[:, 0:xs2_dim]

            # x1, among all x1_dim channels, xs1_dim channels are decisive, others gaussian noise
            # among all xs1_dim decisive channels, overlap_dim are shared between x1 and x2
            x1 = np.random.randn(n_samples, x1_dim)
            x1[:, xs2_dim - overlap_dim:xs2_dim - overlap_dim + xs1_dim] = \
                xs[:, xs2_dim - overlap_dim:xs2_dim - overlap_dim + xs1_dim]

        elif mm_mode == 'increase_alpha':
            xs = np.random.randn(n_samples, x1_dim)
            hyperplane = hyperplane[0:x1_dim]
            y = (np.dot(xs, hyperplane) > 0).ravel()

            x2 = np.random.randn(n_samples, x2_dim)
            x2[:, 0:xs2_dim] = xs[:, 0:xs2_dim]

            # x1: 0:xs1_dim+xs_2dim-decisive features, other dim-gaussian noise
            x1 = np.random.randn(n_samples, x1_dim)
            x1[:, 0:xs1_dim + xs2_dim] = xs[:, 0:xs1_dim + xs2_dim]

        else:
            raise NotImplementedError

        return x1, x2, y

    # create incomplete dataset. x1 is large dataset
    x1, x2, y = generate_mm_data_(n_samples, x1_dim, x2_dim, xs1_dim, xs2_dim, overlap_dim, hyperplane, mm_mode)

    if missing_rate is not None:
        assert 0 < missing_rate < 1
        if missing_mode == 'remove':

            n_missing = int(n_samples * missing_rate)
            missing_index = np.random.choice(np.arange(n_samples), n_missing, replace=False)

            # x1 is a large dataset
            x2_complete = np.delete(x2, missing_index, axis=0)
            x1_complete = np.delete(x1, missing_index, axis=0)
            x1_incomplete = x1[missing_index, :]

            y_complete = np.delete(y, missing_index)
            y_incomplete = y[missing_index]

        elif missing_mode == 'add':
            n_add = int(n_samples * (1 - missing_rate) / missing_rate)
            x1_incomplete, _, y_incomplete = generate_mm_data_(n_add, x1_dim, x2_dim, xs1_dim, xs2_dim,
                                                               overlap_dim, hyperplane, mm_mode)

            x1_complete = x1
            x2_complete = x2
            y_complete = y

        else:
            raise NotImplementedError

    else:
        x1_complete = x1
        x2_complete = x2
        y_complete = y
        x1_incomplete, y_incomplete = None, None

    return x1_complete, x2_complete, y_complete, x1_incomplete, y_incomplete


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
        y = self.y[idx]

        if self.x2 is not None:
            x2 = self.x2[idx]
            return dict(x1=x1, x2=x2, y=y)
        else:
            return dict(x1=x1, y=y)


# Networks
# extractor-projector
class Extractor(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(Extractor, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.layers = nn.Sequential(collections.OrderedDict(
            [
                ('linear', nn.Linear(in_channels, out_channels)),
                ('relu', nn.LeakyReLU(negative_slope=0.1, inplace=False)),
                ('norm', nn.LayerNorm(out_channels))
            ]
        ))
        initialize_weights(self.layers)

    def forward(self, x: torch.Tensor):
        out = self.layers(x)
        return out


# encoder (common and specific)
class Encoder(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, act: str = 'relu'):
        super(Encoder, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.act = act
        if self.act == 'relu':
            act_layer = ('relu', nn.ReLU(inplace=False))
        elif self.act == 'lrelu':
            act_layer = ('lrelu', nn.LeakyReLU(negative_slope=0.1, inplace=False))
        elif self.act == 'sigmoid':
            act_layer = ('sigmoid', nn.Sigmoid())
        else:
            raise ValueError

        self.layers = nn.Sequential(collections.OrderedDict(
            [
                ('linear1', nn.Linear(self.in_channels, self.out_channels)),
                ('norm', nn.LayerNorm(self.out_channels)),
                act_layer,
                ('linear2', nn.Linear(self.out_channels, self.out_channels))
            ]
        ))
        initialize_weights(self.layers)

    def forward(self, x: torch.Tensor):
        out = self.layers(x)
        out = F.normalize(out, p=2, dim=1)
        return out


class SimpleEncoder(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, act: str):
        super(SimpleEncoder, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.act = act
        if self.act is None:
            pass
        elif self.act == 'relu':
            act_layer = ('relu', nn.ReLU(inplace=False))
        elif self.act == 'lrelu':
            act_layer = ('lrelu', nn.LeakyReLU(negative_slope=0.1, inplace=False))
        elif self.act == 'sigmoid':
            act_layer = ('sigmoid', nn.Sigmoid())
        else:
            raise ValueError

        if self.act is None:
            self.layers = nn.Sequential(collections.OrderedDict(
                [
                    ('linear1', nn.Linear(self.in_channels, self.out_channels)),
                    ('norm', nn.LayerNorm(self.out_channels)),
                ]
            ))
        else:
            self.layers = nn.Sequential(
                collections.OrderedDict(
                    [
                        ('linear', nn.Linear(self.in_channels, self.out_channels)),
                        act_layer,
                        ('norm', nn.LayerNorm(self.out_channels)),
                    ]
                )
            )
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
                ('norm', nn.LayerNorm(self.in_channels)),
                ('linear2', nn.Linear(self.in_channels, self.out_channels))
            ]
        ))
        initialize_weights(self.layers)

    def forward(self, x: torch.Tensor):
        out = self.layers(x)
        return out


class SimpleDecoder(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(SimpleDecoder, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.layers = nn.Sequential(collections.OrderedDict(
            [
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


def build_simple_networks(config: argparse.Namespace or edict, **kwargs):

    extractor_1 = Extractor(in_channels=config.x1_dim, out_channels=config.hidden)
    extractor_2 = Extractor(in_channels=config.x2_dim, out_channels=config.hidden)

    encoder_1 = SimpleEncoder(in_channels=config.hidden, out_channels=config.hidden // 2, act=config.encoder_act)
    encoder_2 = SimpleEncoder(in_channels=config.hidden, out_channels=config.hidden // 2, act=config.encoder_act)
    encoder_general = SimpleEncoder(in_channels=config.hidden, out_channels=config.hidden // 2, act=config.encoder_act)

    decoder_1 = SimpleDecoder(in_channels=config.hidden // 2, out_channels=config.hidden)
    decoder_2 = SimpleDecoder(in_channels=config.hidden // 2, out_channels=config.hidden)

    classifier = Classifier(in_channels=config.hidden // 2, n_classes=2)

    networks = dict(extractor_1=extractor_1, extractor_2=extractor_2,
                    encoder_1=encoder_1, encoder_2=encoder_2, encoder_general=encoder_general,
                    decoder_1=decoder_1, decoder_2=decoder_2,
                    classifier=classifier)

    return networks


def build_networks(config: argparse.Namespace or edict, **kwargs):
    # TODO: change x_dim to x1_dim and x2_dim
    extractor_1 = Extractor(in_channels=config.x1_dim, out_channels=config.hidden)
    extractor_2 = Extractor(in_channels=config.x2_dim, out_channels=config.hidden)

    encoder_1 = Encoder(in_channels=config.hidden, out_channels=config.hidden // 2, act=config.encoder_act)
    encoder_2 = Encoder(in_channels=config.hidden, out_channels=config.hidden // 2, act=config.encoder_act)
    encoder_general = Encoder(in_channels=config.hidden, out_channels=config.hidden // 2, act=config.encoder_act)

    decoder_1 = Decoder(in_channels=config.hidden // 2, out_channels=config.hidden)
    decoder_2 = Decoder(in_channels=config.hidden // 2, out_channels=config.hidden)

    classifier = Classifier(in_channels=config.hidden // 2, n_classes=2)

    networks = dict(extractor_1=extractor_1, extractor_2=extractor_2,
                    encoder_1=encoder_1, encoder_2=encoder_2, encoder_general=encoder_general,
                    decoder_1=decoder_1, decoder_2=decoder_2,
                    classifier=classifier)

    return networks


def build_short_networks(config: argparse.Namespace or edict, **kwargs):

    extractor_1 = Extractor(in_channels=config.x1_dim, out_channels=config.hidden)
    extractor_2 = Extractor(in_channels=config.x2_dim, out_channels=config.hidden)
    classifier = Classifier(in_channels=config.hidden, n_classes=2)

    networks = dict(extractor_1=extractor_1, extractor_2=extractor_2, classifier=classifier)

    return networks


if __name__ == 'main':
    dataset = create_dataset(n_train=1000, n_test=1000, x1_dim=50, x2_dim=50, xs1_dim=20, xs2_dim=20,
                             overlap_dim=15, hyperplane_dim=500, missing_rate=0.3, random_state=2021)



