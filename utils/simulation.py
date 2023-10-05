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
from sklearn.model_selection import train_test_split


# Data Generation
def set_random_state(random_state=2021):
    random.seed(random_state)
    np.random.seed(random_state)
    torch.manual_seed(random_state)
    torch.cuda.manual_seed(random_state)
    torch.cuda.manual_seed_all(random_state)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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


def build_networks(config: argparse.Namespace or edict, **kwargs):

    extractor_1 = Extractor(in_channels=config.xs_dim + config.x1_dim, out_channels=config.hidden)
    extractor_2 = Extractor(in_channels=config.xs_dim + config.x2_dim, out_channels=config.hidden)

    if config.simple:
        encoder_1 = SimpleEncoder(in_channels=config.hidden, out_channels=config.hidden, act=config.encoder_act)
        encoder_2 = SimpleEncoder(in_channels=config.hidden, out_channels=config.hidden, act=config.encoder_act)
        encoder_general = SimpleEncoder(in_channels=config.hidden, out_channels=config.hidden, act=config.encoder_act)

        decoder_1 = SimpleDecoder(in_channels=config.hidden, out_channels=config.hidden)
        decoder_2 = SimpleDecoder(in_channels=config.hidden, out_channels=config.hidden)
    else:
        encoder_1 = Encoder(in_channels=config.hidden, out_channels=config.hidden, act=config.encoder_act)
        encoder_2 = Encoder(in_channels=config.hidden, out_channels=config.hidden, act=config.encoder_act)
        encoder_general = Encoder(in_channels=config.hidden, out_channels=config.hidden, act=config.encoder_act)

        decoder_1 = Decoder(in_channels=config.hidden, out_channels=config.hidden)
        decoder_2 = Decoder(in_channels=config.hidden, out_channels=config.hidden)

    classifier = Classifier(in_channels=config.hidden, n_classes=2)

    networks = dict(extractor_1=extractor_1, extractor_2=extractor_2,
                    encoder_1=encoder_1, encoder_2=encoder_2, encoder_general=encoder_general,
                    decoder_1=decoder_1, decoder_2=decoder_2,
                    classifier=classifier)

    return networks


def build_networks_disc(config: argparse.Namespace or edict, **kwargs):
    extractor_1 = Extractor(in_channels=config.xs_dim + config.x1_dim, out_channels=config.hidden)
    extractor_2 = Extractor(in_channels=config.xs_dim + config.x2_dim, out_channels=config.hidden)

    if config.simple:
        encoder_1 = SimpleEncoder(in_channels=config.hidden, out_channels=config.hidden, act=config.encoder_act)
        encoder_2 = SimpleEncoder(in_channels=config.hidden, out_channels=config.hidden, act=config.encoder_act)
        encoder_general = SimpleEncoder(in_channels=config.hidden, out_channels=config.hidden, act=config.encoder_act)

        decoder_1 = SimpleDecoder(in_channels=config.hidden, out_channels=config.hidden)
        decoder_2 = SimpleDecoder(in_channels=config.hidden, out_channels=config.hidden)
    else:
        encoder_1 = Encoder(in_channels=config.hidden, out_channels=config.hidden, act=config.encoder_act)
        encoder_2 = Encoder(in_channels=config.hidden, out_channels=config.hidden, act=config.encoder_act)
        encoder_general = Encoder(in_channels=config.hidden, out_channels=config.hidden, act=config.encoder_act)

        decoder_1 = Decoder(in_channels=config.hidden, out_channels=config.hidden)
        decoder_2 = Decoder(in_channels=config.hidden, out_channels=config.hidden)

    discriminator = Classifier(in_channels=config.hidden, n_classes=3)

    classifier = Classifier(in_channels=config.hidden, n_classes=2)

    networks = dict(extractor_1=extractor_1, extractor_2=extractor_2,
                    encoder_1=encoder_1, encoder_2=encoder_2, encoder_general=encoder_general,
                    decoder_1=decoder_1, decoder_2=decoder_2,
                    discriminator=discriminator,
                    classifier=classifier)

    return networks


class DataGenerator(object):

    def __init__(self, zs_dim=16, z1_dim=16, z2_dim=16, rho=0.5, sigma=1.0, random_state=2021):

        # save arguments
        self.zs_dim = zs_dim
        self.z1_dim = z1_dim
        self.z2_dim = z2_dim
        self.rho = rho
        self.sigma = sigma
        self.random_state = random_state

        # initialize covariance matrix
        set_random_state(random_state=self.random_state)
        self._create_covariance()

    def sample_z(self, mu, n_samples):
        set_random_state(random_state=self.random_state)
        total_dim = self.zs_dim + self.z1_dim + self.z2_dim
        mean = np.full(shape=total_dim, fill_value=mu)

        samples = np.random.multivariate_normal(mean=mean, cov=self.cov, size=n_samples)

        zs = samples[:, :self.zs_dim]
        z1 = samples[:, self.zs_dim:self.zs_dim + self.z1_dim]
        z2 = samples[:, self.zs_dim + self.z1_dim:]

        zs = torch.tensor(zs, dtype=torch.float)
        z1 = torch.tensor(z1, dtype=torch.float)
        z2 = torch.tensor(z2, dtype=torch.float)

        return zs, z1, z2

    def _create_covariance(self):

        total_dim = self.zs_dim + self.z1_dim + self.z2_dim

        zs_cov = self.single_covariance(self.zs_dim)
        z1_cov = self.single_covariance(self.z1_dim)
        z2_cov = self.single_covariance(self.z2_dim)

        cov = np.zeros((total_dim, total_dim))
        cov[:self.zs_dim, :self.zs_dim] = zs_cov
        cov[self.zs_dim:self.zs_dim + self.z1_dim, self.zs_dim:self.zs_dim + self.z1_dim] = z1_cov
        cov[self.zs_dim + self.z1_dim:, self.zs_dim + self.z1_dim:] = z2_cov

        self.cov = cov

    def single_covariance(self, z_dim):
        cov = np.empty((z_dim, z_dim))
        cov[0, :] = np.arange(z_dim)
        for i in range(1, z_dim):
            cov[i, :] = cov[i - 1, :] - 1
        cov = np.abs(cov)
        cov = np.power(self.rho, cov) * self.sigma ** 2
        return cov

    def _create_layer(self, xs_dim, x1_dim, x2_dim, slope=0.1):
        set_random_state(random_state=self.random_state)
        self.layer_s = nn.Sequential(collections.OrderedDict(
            [('linear', nn.Linear(self.zs_dim, xs_dim)),
             ('relu', nn.LeakyReLU(negative_slope=slope))]))
        self.layer_1 = nn.Sequential(collections.OrderedDict(
            [('linear', nn.Linear(self.z1_dim, x1_dim)),
             ('relu', nn.LeakyReLU(negative_slope=slope))]))
        self.layer_2 = nn.Sequential(collections.OrderedDict(
            [('linear', nn.Linear(self.z2_dim, x2_dim)),
             ('relu', nn.LeakyReLU(negative_slope=slope))]))

    def sample_x(self, zs, z1, z2):
        xs = self.layer_s(zs).detach()
        x1 = self.layer_1(z1).detach()
        x2 = self.layer_2(z2).detach()
        return xs, x1, x2

    def generate_data(self, mu_0, mu_1, xs_dim, x1_dim, x2_dim, slope, n_complete, n_incomplete, n_validation, n_test):
        self._create_layer(xs_dim=xs_dim, x1_dim=x1_dim, x2_dim=x2_dim, slope=slope)

        # complete data
        # class 0 & class1
        zs_0, z1_0, z2_0 = self.sample_z(mu=mu_0, n_samples=(n_complete+n_test)//2)
        self.random_state = self.random_state + 100
        zs_1, z1_1, z2_1 = self.sample_z(mu=mu_1, n_samples=(n_complete+n_test)//2)

        # z -> x (first notation: modality / second notation: class)
        xs_0, x1_0, x2_0 = self.layer_s(zs_0), self.layer_1(z1_0), self.layer_2(z2_0)
        xs_1, x1_1, x2_1 = self.layer_s(zs_1), self.layer_1(z1_1), self.layer_2(z2_1)

        X1_0 = torch.concat([xs_0, x1_0], dim=1)
        X1_1 = torch.concat([xs_1, x1_1], dim=1)
        x1_complete = torch.concat([X1_0, X1_1], dim=0)

        X2_0 = torch.concat([xs_0, x2_0], dim=1)
        X2_1 = torch.concat([xs_1, x2_1], dim=1)
        x2_complete = torch.concat([X2_0, X2_1], dim=0)

        y_complete = [0] * len(X1_0) + [1] * len(X1_1)
        y_complete = np.array(y_complete)

        x1_train_complete, x1_test, x2_train_complete, x2_test, y_train_complete, y_test = \
            train_test_split(x1_complete, x2_complete, y_complete,
                             test_size=n_test, random_state=self.random_state,
                             stratify=y_complete)

        x1_train_complete = x1_train_complete.detach()
        x1_test = x1_test.detach()
        x2_train_complete = x2_train_complete.detach()
        x2_test = x2_test.detach()

        # incomplete data: modality 1
        if n_incomplete > 0:
            self.random_state = self.random_state + 100
            zs_0, z1_0, _ = self.sample_z(mu=mu_0, n_samples=n_incomplete//2)
            self.random_state = self.random_state + 100
            zs_1, z1_1, _ = self.sample_z(mu=mu_1, n_samples=n_incomplete//2)

            xs_0, x1_0 = self.layer_s(zs_0), self.layer_1(z1_0)
            xs_1, x1_1 = self.layer_s(zs_1), self.layer_1(z1_1)
            X1_0 = torch.concat([xs_0, x1_0], dim=1)
            X1_1 = torch.concat([xs_1, x1_1], dim=1)
            x1_train_incomplete = torch.concat([X1_0, X1_1], dim=0)
            x1_train_incomplete = x1_train_incomplete.detach()

            y_train_incomplete = [0] * len(X1_0) + [1] * len(X1_1)
            y_train_incomplete = np.array(y_train_incomplete)
        else:
            x1_train_incomplete = None
            y_train_incomplete = None

        # validation set
        # TODO: combine with the previous train/test generation step
        self.random_state = self.random_state + 100
        zs_0, z1_0, z2_0 = self.sample_z(mu=mu_0, n_samples=n_validation // 2)
        self.random_state = self.random_state + 100
        zs_1, z1_1, z2_1 = self.sample_z(mu=mu_1, n_samples=n_validation // 2)

        xs_0, x1_0, x2_0 = self.layer_s(zs_0), self.layer_1(z1_0), self.layer_2(z2_0)
        xs_1, x1_1, x2_1 = self.layer_s(zs_1), self.layer_1(z1_1), self.layer_2(z2_1)

        X1_0 = torch.concat([xs_0, x1_0], dim=1)
        X1_1 = torch.concat([xs_1, x1_1], dim=1)
        x1_validation = torch.concat([X1_0, X1_1], dim=0).detach()

        X2_0 = torch.concat([xs_0, x2_0], dim=1)
        X2_1 = torch.concat([xs_1, x2_1], dim=1)
        x2_validation = torch.concat([X2_0, X2_1], dim=0).detach()

        y_complete = [0] * len(X1_0) + [1] * len(X1_1)
        y_validation = np.array(y_complete)

        dataset = dict(x1_train_complete=x1_train_complete,
                       x2_train_complete=x2_train_complete,
                       y_train_complete=y_train_complete,
                       x1_train_incomplete=x1_train_incomplete,
                       y_train_incomplete=y_train_incomplete,
                       x1_validation=x1_validation,
                       x2_validation=x2_validation,
                       y_validation=y_validation,
                       x1_test=x1_test,
                       x2_test=x2_test,
                       y_test=y_test)

        return dataset


if __name__ == '__main__':

    # data generation
    data_generator = DataGenerator(zs_dim=16, z1_dim=16, z2_dim=16,
                                   rho=0.5, sigma=1.0, random_state=2021)
    data_generator._create_layer(xs_dim=32, x1_dim=32, x2_dim=32)
    z0, z1, z2 = data_generator.sample_z(mu=0, n_samples=100)
    x0, x1, x2 = data_generator.sample_x(z0, z1, z2)

    # experiment 1. only complete data, but different gamma (common representation ratio)
    # z_dim = total dimension
    # z_dim = 32
    # for z0_dim in [0, 16, 32]:
    #     z1_dim = int(z_dim - z0_dim)
    #     z2_dim = int(z_dim - z0_dim)
    data_generator = DataGenerator(zs_dim=16, z1_dim=16, z2_dim=16,
                                   rho=0.5, sigma=1.0, random_state=2021)
    datasets = data_generator.generate_data(mu_0=0.0, mu_1=1.0,
                                            xs_dim=20, x1_dim=20, x2_dim=20,
                                            slope=0.1,
                                            n_complete=500, n_incomplete=200, n_validation=500, n_test=500)

    from easydict import EasyDict as edict
    config = edict
    config.xs_dim = 20
    config.x1_dim = 20
    config.x2_dim = 20
    config.hidden = 25
    config.encoder_act = 'relu'
    networks = build_networks(config=config)

    for name, network in networks.items():
        print(network)

    datasets['x1_train_complete']