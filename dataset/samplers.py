# -*- coding: utf-8 -*-

from collections import Counter

import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data.sampler import Sampler


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


if __name__ == '__main__':

    class Dummy(Dataset):
        def __init__(self):
            self.x = range(100)
            self.y = [0] * 10 + [1] * 90
        def __len__(self):
            return 100
        def __getitem__(self, idx):
            return dict(x=self.x[idx], y=self.y[idx], idx=idx)

    from torch.utils.data import DataLoader

    dset = Dummy()
    loader = DataLoader(dataset=dset, batch_size=10, sampler=ImbalancedDatasetSampler(dataset=dset))

    idx_list = []
    import numpy as np
    for batch in loader:
        print(np.bincount(batch['y'].numpy()))
