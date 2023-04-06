import os

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle

import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight


class BrainProcessor(object):
    def __init__(self,
                 root: str,
                 data_file: str = 'labels/data_info_multi.csv',
                 pet_type: str = 'FBP',
                 mci_only: bool = False,
                 random_state: int = 2023):

        self.root = root
        self.data_file = data_file
        assert pet_type in ['FDG', 'FBP']
        self.pet_type = pet_type
        self.mci_only = mci_only
        self.random_state = random_state

        self.demo_columns = ['PTGENDER (1=male, 2=female)', 'Age', 'PTEDUCAT',
                             'APOE Status', 'MMSCORE']
                             # , 'CDGLOBAL', 'SUM BOXES']

        self.missing_rate = None

    def process(self, validation_size: float = 0.1, test_size: float = 0.1, missing_rate: float = None):

        data = pd.read_csv(os.path.join(self.root, self.data_file), converters={'RID': str, 'Conv': int})
        data['PET'] = data[self.pet_type]

        data = data.loc[data['Conv'].isin([0, 1])].reset_index(drop=True)
        data['PET_available'] = ~data['PET'].isna()

        # 0. preprocess data
        data = self.preprocess_mc_hippo(data)
        data = self.preprocess_demo(data)

        if self.mci_only:
            data = data.loc[data.MCI == 1]

        # 1. Get stratified test split of MRI + PET data
        mri_pet_data = data.loc[data['PET_available'], :]
        mri_pet_label = mri_pet_data.groupby(['RID'])['Conv'].max()
        mri_pet_rid, mri_pet_label = mri_pet_label.index.values, mri_pet_label.values

        train_size = 1 - (validation_size + test_size)

        rid_train, rid_test, label_train, label_test = train_test_split(mri_pet_rid,
                                                                        mri_pet_label,
                                                                        test_size=test_size,
                                                                        shuffle=True,
                                                                        random_state=self.random_state,
                                                                        stratify=mri_pet_label)
        rid_train, rid_validation = train_test_split(rid_train,
                                                     test_size=validation_size / (train_size + validation_size),
                                                     shuffle=True,
                                                     random_state=self.random_state,
                                                     stratify=label_train)

        assert np.intersect1d(rid_train, rid_validation).__len__() == 0
        assert np.intersect1d(rid_train, rid_test).__len__() == 0
        assert np.intersect1d(rid_validation, rid_test).__len__() == 0

        data_train = data.loc[np.isin(data['RID'], rid_train), :]
        data_validation = data.loc[np.isin(data['RID'], rid_validation), :]
        data_test = data.loc[np.isin(data['RID'], rid_test), :]

        # 2. Missing rate
        # 2-1. Current missing rate
        current_missing_rate = np.sum(~data['PET_available']) / len(data_train)

        # 2-2. Adjust missing rate
        if missing_rate is not None:
            assert missing_rate > current_missing_rate

            pet_available_index = data_train.loc[data_train['PET_available']].index.values

            # target missing obs - current missing obs
            num_adjust = int(len(data_train) * missing_rate) - np.sum(~data_train['PET_available'])
            np.random.seed(self.random_state)
            adjust_index = np.random.choice(pet_available_index, num_adjust, replace=False)

            data_train.loc[adjust_index, 'PET'] = np.nan
            data_train.loc[adjust_index, 'PET_available'] = False

        else:
            missing_rate = current_missing_rate

        self.missing_rate = missing_rate

        # 3. Split data into complete and MRI-only
        # 3-1. Incomplete MRI-only training
        incomplete_train = data_train.loc[~data_train['PET_available'], :]

        # 3-2. Complete MRI & PET training
        complete_train = data_train.loc[data_train['PET_available'], :]

        # 3-3. Complete MRI & PET validation & testing
        complete_validation = data_validation.loc[data_validation['PET_available'], :]
        complete_test = data_test.loc[data_test['PET_available'], :]

        # 3-4. Complete + Incomplete MRI-only training
        total_mri_train = data_train.copy()

        # 4. Parse
        mri_pet_complete_train = self.parse_data(complete_train)
        mri_incomplete_train = self.parse_data(incomplete_train)
        mri_total_train = self.parse_data(total_mri_train)
        mri_pet_complete_validation = self.parse_data(complete_validation)
        mri_pet_complete_test = self.parse_data(complete_test)

        # 5. Class Weight for Balancing
        self.class_weight = class_weight.compute_class_weight(class_weight='balanced',
                                                              classes=np.unique(mri_pet_complete_train['y']),
                                                              y=mri_pet_complete_train['y'])

        # 6. Return
        datasets = {'mri_pet_complete_train': mri_pet_complete_train,
                    'mri_incomplete_train': mri_incomplete_train,
                    'mri_total_train': mri_total_train,
                    'mri_pet_complete_validation': mri_pet_complete_validation,
                    'mri_pet_complete_test': mri_pet_complete_test}

        return datasets

    def preprocess_mc_hippo(self, data):

        mc_table = pd.read_excel(os.path.join(self.root, 'labels/AV45_FBP_SUVR.xlsx'), sheet_name='list_id_SUVR_RSF')
        mri_table = pd.read_csv(os.path.join(self.root, 'labels/MRI_BAI_features.csv'))

        # MC
        mc_table = mc_table.rename(columns={'ID': 'PET'})
        data = pd.merge(left=data, right=mc_table[['PET', 'MC']], how='left', on='PET')
        data['MC'] = 53.6 * data['MC'] - 43.2

        # Hippocampus Volume
        mri_table['MRI'] = mri_table['Filename'].str.split('/', expand=True).iloc[:, 1]
        mri_table['Volume'] = mri_table['Left-Hippocampus'] + mri_table['Right-Hippocampus']
        data = pd.merge(left=data, right=mri_table[['MRI', 'Volume']], how='left', on='MRI')

        return data

    def preprocess_demo(self, data):

        data_demo = data[['RID', 'Conv'] + self.demo_columns]

        # 1. Gender
        if 'PTGENDER (1=male, 2=female)' in self.demo_columns:
            data_demo_ = data_demo.copy()
            data_demo_['PTGENDER (1=male, 2=female)'] = data_demo_['PTGENDER (1=male, 2=female)'] - 1
            data_demo = data_demo_.copy()

        # 2. APOE Status
        if 'APOE Status' in self.demo_columns:
            data_demo_ = data_demo.copy()
            data_demo_.loc[:, 'APOE Status'] = data_demo['APOE Status'].fillna('NC')
            data_demo_['APOE Status'] = [0 if a == 'NC' else 1 for a in data_demo_['APOE Status'].values]
            data_demo = data_demo_.copy()

        # 3. Others
        cols = ['MMSCORE', 'CDGLOBAL', 'SUM BOXES']
        cols = [c for c in cols if c in self.demo_columns]
        records = data_demo.groupby('Conv').mean()[cols].to_dict()

        data_demo_ = data_demo.copy()
        for col in cols:
            nan_index = data_demo_.index[data_demo[col].isna()]
            for i in nan_index:
                value = records[col][data_demo_.loc[i, 'Conv']]
                data_demo_.loc[i, col] = value
        data_demo = data_demo_.copy()
        data[self.demo_columns] = data_demo[self.demo_columns]
        self.num_demo_columns = len(self.demo_columns)

        return data

    def parse_data(self, data):
        mri_files = [self.str2mri(p) if type(p) == str else p for p in data['MRI']]
        pet_files = [self.str2pet(p) if type(p) == str else p for p in data['PET']]
        demo = data[self.demo_columns].values
        mc = data['MC'].values
        volume = data['Volume'].values
        y = data.Conv.values
        return dict(mri=mri_files, pet=pet_files, demo=demo, mc=mc, volume=volume, y=y)

    def str2mri(self, i):
        return os.path.join(self.root, 'template/FS7', f'{i}.pkl')

    def str2pet(self, i):
        if self.pet_type == 'FDG':
            return os.path.join(self.root, 'template/FDG', f'{i}.pkl')
        elif self.pet_type == 'FBP':
            return os.path.join(self.root, 'template/PUP_FBP', f'{i}.pkl')


class BrainBase(Dataset):

    def __init__(self, dataset: dict, **kwargs):
        self.mri = dataset['mri']
        self.pet = dataset['pet']
        self.demo = dataset['demo']
        self.mc = dataset['mc']
        self.volume = dataset['volume']
        self.y = dataset['y']

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        raise NotImplementedError

    @staticmethod
    def load_image(path):
        # if math.isnan(path):
        #     raise ValueError("Path is NaN")
        with open(path, 'rb') as fb:
            image = pickle.load(fb)
        return image


class BrainMRI(BrainBase):

    def __init__(self, dataset, mri_transform):
        super().__init__(dataset)
        self.mri_transform = mri_transform

    def __getitem__(self, idx):
        mri = self.load_image(path=self.mri[idx])
        if self.mri_transform is not None:
            mri = self.mri_transform(mri)
        demo = self.demo[idx]
        mc = self.mc[idx]
        volume = self.volume[idx]
        y = self.y[idx]
        return dict(mri=mri, demo=demo, mc=mc, volume=volume, y=y, idx=idx)


class BrainPET(BrainBase):

    def __init__(self, dataset, pet_transform):
        super().__init__(dataset)
        self.pet_transform = pet_transform

    def __getitem__(self, idx):
        pet = self.load_image(path=self.pet[idx])
        if self.pet_transform is not None:
            pet = self.pet_transform(pet)
        demo = self.demo[idx]
        mc = self.mc[idx]
        volume = self.volume[idx]
        y = self.y[idx]
        return dict(pet=pet, demo=demo, mc=mc, volume=volume, y=y, idx=idx)


class BrainMulti(BrainBase):

    def __init__(self, dataset, mri_transform, pet_transform):
        super().__init__(dataset)
        self.mri_transform = mri_transform
        self.pet_transform = pet_transform

    def __getitem__(self, idx):
        mri = self.load_image(path=self.mri[idx])
        if self.mri_transform is not None:
            mri = self.mri_transform(mri)
        pet = self.load_image(path=self.pet[idx])
        if self.pet_transform is not None:
            pet = self.pet_transform(pet)
        demo = self.demo[idx]
        mc = self.mc[idx]
        volume = self.volume[idx]
        y = self.y[idx]
        return dict(mri=mri, pet=pet, demo=demo, mc=mc, volume=volume, y=y, idx=idx)


if __name__ == '__main__':

    from torch.utils.data import DataLoader
    from datasets.slice.transforms import make_pet_transforms
    import matplotlib.pyplot as plt

    processor = BrainProcessor(root='D:/data/ADNI',
                               data_file='labels/data_info_multi.csv',
                               pet_type='FBP',
                               random_state=2023)
    datasets = processor.process(validation_size=0.1, test_size=0.1, missing_rate=None)

    for k, v in datasets.items():
        print(k, f": {len(v['y'])} observations")

    mri_pet_complete_train = datasets['mri_pet_complete_train']

    train_transform, test_transform = make_pet_transforms(image_size=72, affine=True)

    pet_set = BrainPET(mri_pet_complete_train, train_transform)
    pet_loader = DataLoader(pet_set, batch_size=5)
    M = -100
    for i, batch in enumerate(pet_loader):
        for x in batch['pet']:
            plt.imshow(x[0][0], cmap='binary')
            plt.show()
        break
