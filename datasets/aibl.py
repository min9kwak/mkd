import os
import pandas as pd
import numpy as np
import pickle

import random
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.utils import class_weight
from sklearn.preprocessing import MinMaxScaler


class AIBLProcessor(object):
    def __init__(self,
                 root: str = 'D:/data/AIBL',
                 data_info: str = 'data_info_mri.csv',
                 time_window: int = 36,
                 random_state: int = 2021):

        # Processor for PiB scans
        self.root = root
        self.time_window = time_window
        self.random_state = random_state

        # PiB data (paired with MRI)
        data_info = pd.read_csv(os.path.join(root, data_info), converters={'RID': str})
        data_info = data_info.loc[data_info.is_file]

        data_info = data_info.replace({f'Conv_{time_window}': {'NC': 0, 'C': 1}})
        data_info[f'Conv_{time_window}'] = data_info[f'Conv_{time_window}'].fillna(value=-1)
        data_info[f'Conv_{time_window}'] = data_info[f'Conv_{time_window}'].astype(int)

        # MCI only
        data_info = data_info.loc[data_info['DX'].isin(['MCI', 'AD'])]

        self.u_data_info = data_info.loc[data_info['Conv_36'].isin([-1])].reset_index(drop=True)
        self.data_info = data_info.loc[data_info['Conv_36'].isin([0, 1])].reset_index(drop=True)

    def process(self, n_splits=10, n_cv=0, test_only=False):

        # def process
        rid = self.data_info.RID.tolist()
        conv = self.data_info['Conv_36'].tolist()
        assert 0 <= n_cv < n_splits

        cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=2021)

        train_idx_list, test_idx_list = [], []
        for train_idx, test_idx in cv.split(X=rid, y=conv, groups=rid):
            train_idx_list.append(train_idx)
            test_idx_list.append(test_idx)
        train_idx, test_idx = train_idx_list[n_cv], test_idx_list[n_cv]

        train_info = self.data_info.iloc[train_idx].reset_index(drop=True)
        test_info = self.data_info.iloc[test_idx].reset_index(drop=True)

        # filter rids in unlabeled data
        test_rid = list(set(test_info.RID))
        self.u_data_info = self.u_data_info[~self.u_data_info.RID.isin(test_rid)]
        u_train_info = self.u_data_info.reset_index(drop=True)

        # parse to make paths
        train_data = self.parse_info(train_info)
        test_data = self.parse_info(test_info)
        u_train_data = self.parse_info(u_train_info)

        # set class weight
        self.class_weight = class_weight.compute_class_weight(class_weight='balanced',
                                                              classes=np.unique(train_data['y']),
                                                              y=train_data['y'])
        if test_only:
            test_data.update(train_data)
            train_data = None

        datasets = {'train': train_data, 'test': test_data, 'u_train': u_train_data}

        return datasets

    def parse_info(self, data_info):
        image_files = [os.path.join(self.root, f'template/MRI/{p}') if type(p) == str else p
                       for p in data_info['image_file'].tolist()]
        y = data_info['Conv_36'].values
        return dict(image_files=image_files, y=y)

#
class AIBLDataset(Dataset):

    def __init__(self,
                 dataset: dict,
                 transform,
                 **kwargs):
        self.image_files = dataset['image_files']
        self.y = dataset['y']
        self.transform = transform

    def __getitem__(self, idx):
        img = self.load_image(path=self.image_files[idx])
        if self.transform is not None:
            img = self.transform(img)
        y = self.y[idx]
        return dict(x=img, y=y, idx=idx)

    @staticmethod
    def load_image(path):
        with open(path, 'rb') as fb:
            image = pickle.load(fb)
        return image

    def __len__(self):
        return len(self.y)


if __name__ == '__main__':

    processor = AIBLProcessor(root='D:/data/AIBL',
                              data_info='data_info_mri.csv',
                              time_window=36,
                              random_state=2021)
    datasets = processor.process(5, 0, test_only=False)
    np.bincount(datasets['train']['y'])
    np.bincount(datasets['test']['y'])

    from datasets.slice.transforms import make_mri_transforms
    train_transform, test_transform = make_mri_transforms(image_size_mri=72,
                                                          intensity_mri='scale',
                                                          crop_size_mri=64,
                                                          rotate_mri=True,
                                                          flip_mri=True,
                                                          affine_mri=False,
                                                          blur_std_mri=None,
                                                          train_slices='fixed',
                                                          prob=0.5)
    train_set = AIBLDataset(dataset=datasets['test'], transform=train_transform)

    from torch.utils.data import DataLoader
    train_loader = DataLoader(dataset=train_set, batch_size=4,
                              sampler=None, drop_last=True)
    for batch in train_loader:
        slice = batch['x'][0][0, 0, :, :]
        import matplotlib.pyplot as plt
        import seaborn as sns
        sns.heatmap(slice, cmap='binary')
        plt.show()
        assert False

    # concatenate datasets
