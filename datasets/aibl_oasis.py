import os
import pandas as pd
import numpy as np
import pickle

from torch.utils.data import Dataset
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.utils import class_weight
from sklearn.preprocessing import MinMaxScaler


# AIBL & OASIS
class AOProcessor(object):
    def __init__(self,
                 root: str = 'D:/data',
                 data_info: str = 'aibl_oasis_data_info.csv',
                 data_type: str = 'mri',
                 use_cdr: bool = True,
                 scale_demo: bool = True,
                 random_state: int = 2021):

        # Processor for PiB scans
        self.root = root
        self.use_cdr = use_cdr
        self.scale_demo = scale_demo
        self.random_state = random_state
        self.demo_columns = ['Age', 'PTGENDER', 'CDGLOBAL', 'MMSCORE', 'APOE']
        if not self.use_cdr:
            self.demo_columns.remove('CDGLOBAL')

        # TODO: pib -> pet with argument (switching pib and av45)
        assert data_type in ['mri', 'mri+pib']
        self.data_type = data_type

        # read data_info
        data_info = os.path.join(root, data_info)
        data_info = pd.read_csv(data_info, converters={'RID': str})

        # remove failed image
        data_info['PIB Path'] = data_info['PIB Path'].replace('OAS30896_PIB_d1601.pkl', np.nan)

        # convert Conv and filter
        data_info = data_info.replace({'Conv': {'NC': 0, 'C': 1}})
        data_info['Conv'] = data_info['Conv'].fillna(value=-1)
        data_info = data_info.loc[data_info['Conv'].isin([0, 1])].reset_index(drop=True)

        # preprocess demo
        cols = ['MMSCORE', 'CDGLOBAL']
        records = data_info.groupby('Conv').mean()[cols].to_dict()
        for col in cols:
            nan_index = data_info.index[data_info[col].isna()]
            for i in nan_index:
                value = records[col][data_info.loc[i, 'Conv']]
                data_info.loc[i, col] = value

        # data_type filtering
        if data_type == 'mri+pib':
            data_info = data_info.loc[~data_info['PIB Path'].isna()].reset_index(drop=True)

        self.data_info = data_info

    def process(self, n_splits=10, n_cv=0, test_only=False):

        # split
        rid = self.data_info['RID'].tolist()
        conv = self.data_info['Conv'].tolist()
        assert 0 <= n_cv < n_splits

        cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=2021)

        train_idx_list, test_idx_list = [], []
        for train_idx, test_idx in cv.split(X=rid, y=conv, groups=rid):
            train_idx_list.append(train_idx)
            test_idx_list.append(test_idx)
        train_idx, test_idx = train_idx_list[n_cv], test_idx_list[n_cv]

        train_info = self.data_info.iloc[train_idx].reset_index(drop=True)
        test_info = self.data_info.iloc[test_idx].reset_index(drop=True)

        # demographic scaling
        if self.scale_demo:
            scaler = MinMaxScaler()
            train_info[self.demo_columns] = scaler.fit_transform(train_info[self.demo_columns])
            test_info[self.demo_columns] = scaler.transform(test_info[self.demo_columns])

        # parsing
        # parse to make paths
        train_data = self.parse_info(train_info)
        test_data = self.parse_info(test_info)

        # set class weight
        self.class_weight = class_weight.compute_class_weight(class_weight='balanced',
                                                              classes=np.unique(train_data['y']),
                                                              y=train_data['y'])
        if test_only:
            test_data.update(train_data)
            train_data = None

        datasets = {'train': train_data, 'test': test_data}

        return datasets

    def parse_info(self, data_info):
        mri_files, pib_files = [], []
        for i, row in data_info.iterrows():
            mri_path, pib_path, source = row[['MR Path', 'PIB Path', 'Source']]
            if type(mri_path) == str:
                mri_path = os.path.join(self.root, source, 'template', 'MRI', mri_path)
            if type(pib_path) == str:
                pib_path = os.path.join(self.root, source, 'template', 'PIB', pib_path)
            mri_files.append(mri_path), pib_files.append(pib_path)
        y = data_info['Conv'].values
        demo = data_info[self.demo_columns].values
        return dict(mri_files=mri_files, pib_files=pib_files, demo=demo, y=y)

# TODO: make base, single, multi
class AODataset(Dataset):

    def __init__(self,
                 dataset: dict,
                 data_type: str,
                 transform,
                 **kwargs):
        assert data_type in ['mri', 'mri+pib']
        self.data_type = data_type
        self.mri_files = dataset['mri_files'] if 'mri' in data_type else None
        self.pib_files = dataset['pib_files'] if 'pib' in data_type else None

        if 'pib' in data_type:
            assert all(isinstance(item, str) for item in self.pib_files)

        self.demo = dataset['demo']
        self.y = dataset['y']
        self.transform = transform

    def __getitem__(self, idx):

        mri = self.load_image(path=self.mri_files[idx])
        if self.transform is not None:
            mri = self.transform(mri)

        if 'pib' in self.data_type:
            pib = self.load_image(path=self.pib_files[idx])
            if self.transform is not None:
                pib = self.transform(pib)

        demo = self.demo[idx]
        y = self.y[idx]

        if 'pib' in self.data_type:
            return dict(mri=mri, pet=pib, demo=demo, y=y, idx=idx)
        else:
            return dict(mri=mri, demo=demo, y=y, idx=idx)

    @staticmethod
    def load_image(path):
        with open(path, 'rb') as fb:
            image = pickle.load(fb)
        return image

    def __len__(self):
        return len(self.y)


if __name__ == '__main__':

    data_type = 'mri'
    processor = AOProcessor(data_type=data_type)
    datasets = processor.process(5, 0, test_only=True)

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
    test_set = AODataset(dataset=datasets['test'], data_type=data_type, transform=train_transform)

    from torch.utils.data import DataLoader
    import tqdm

    test_loader = DataLoader(dataset=test_set, batch_size=4,
                             sampler=None, drop_last=True)
    for batch in tqdm.tqdm(test_loader):
        ''
