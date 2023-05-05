# -*- coding: utf-8 -*-

from collections import Counter

import numpy as np

import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data.sampler import Sampler

from sklearn.model_selection import StratifiedShuffleSplit


class ImbalancedDatasetSampler(Sampler):
    def __init__(self, dataset, indices=None, num_samples=None):
        super(ImbalancedDatasetSampler, self).__init__(dataset)

        if indices is None:
            self.indices = list(range(len(dataset)))
        else:
            self.indices = indices

        if num_samples is None:
            self.num_samples = len(self.indices)
        else:
            self.num_samples = num_samples

        target_counts = self.get_target_counts(dataset)

        weights = []
        for idx in self.indices:
            target = self.get_target(dataset, idx)
            weights += [1.0 / target_counts[target]]

        self.weights = torch.Tensor(weights).float()

    def __iter__(self):
        return (
            self.indices[i] for i in torch.multinomial(self.weights, self.num_samples, replacement=True)
        )

    def __len__(self):
        return self.num_samples

    @staticmethod
    def get_target_counts(dataset: Dataset):
        targets = [l for l in dataset.y]
        return Counter(targets)

    @staticmethod
    def get_target(dataset: Dataset, idx: int):
        return dataset.y[idx]


class StratifiedSampler(Sampler):
    """Stratified Sampling
    Provides equal representation of target classes in each batch
    """

    def __init__(self, class_vector, batch_size):
        """
        Arguments
        ---------
        class_vector : torch tensor
            a vector of class labels
        batch_size : integer
            batch_size
        """
        self.n_splits = int(len(class_vector) / batch_size)
        self.class_vector = class_vector

    def gen_sample_array(self):

        s = StratifiedShuffleSplit(n_splits=self.n_splits, test_size=0.5)
        X = torch.randn(len(self.class_vector), 2).numpy()
        y = np.array(self.class_vector)
        s.get_n_splits(X, y)

        train_index, test_index = next(s.split(X, y))
        return np.hstack([train_index, test_index])

    def __iter__(self):
        return iter(self.gen_sample_array())

    def __len__(self):
        return len(self.class_vector)


if __name__ == '__main__':

    class Dummy(Dataset):
        def __init__(self):
            self.x = range(1000)
            self.y = [0] * 100 + [1] * 900
        def __len__(self):
            return len(self.y)
        def __getitem__(self, idx):
            return dict(x=self.x[idx], y=self.y[idx], idx=idx)

    from torch.utils.data import DataLoader

    dset = Dummy()
    loader = DataLoader(dataset=dset, batch_size=20, sampler=ImbalancedDatasetSampler(dataset=dset))

    idx_list = []
    import numpy as np
    for batch in loader:
        print(np.bincount(batch['y'].numpy()))
        break

    loader = DataLoader(dataset=dset, batch_size=20,
                        sampler=StratifiedSampler(class_vector=dset.y, batch_size=10))
    for batch in loader:
        print(np.bincount(batch['y'].numpy()))