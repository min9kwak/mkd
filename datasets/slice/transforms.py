import random
import numpy as np

import torch
from monai.transforms import (
    Compose, AddChannel, RandRotate, RandRotate90, Resize, ScaleIntensity, ToTensor, RandFlip, RandZoom, RandAffine,
    RandSpatialCrop, NormalizeIntensity, RandGaussianNoise, Transform, CenterSpatialCrop,
    AddChannel
)
from torchvision.transforms import ConvertImageDtype, Normalize
from monai.utils.enums import TransformBackends
from monai.config import DtypeLike
from monai.utils import convert_data_type


class Divide(Transform):
    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __call__(self, img, *args, **kwargs):
        return img / 255.0


class RandomSlices(Transform):

    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __init__(self, num_slices: int = 5, image_size: int = 72, slice_range: float = 0.15):
        self.num_slices = num_slices
        self.image_size = image_size

        m = int(image_size * slice_range)
        center = image_size // 2
        self.slice_range = (center - m, center + m + 1)

    def __call__(self, img, *args, **kwargs):

        ret = []
        for i in range(self.num_slices):
            dim = random.choice(range(3))
            point = random.choice(range(*self.slice_range))
            if dim == 0:
                slice = img[:, point, :, :]
            elif dim == 1:
                slice = img[:, :, point, :]
            elif dim == 2:
                slice = img[:, :, :, point]
            else:
                raise ValueError
            ret.append(slice)
        return ret


class FixedSlices(Transform):
    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __init__(self, image_size: int = 72, space: int = 3, n_points: int = 5):
        '''
        if n_points=0, slice only at the center point
        total number of slices = 3 * (2 * n_points + 1)
        '''

        center = image_size // 2
        self.slicing_points = center + np.linspace(start=-n_points * space,
                                                   stop=n_points * space,
                                                   num=2 * n_points + 1)
        self.slicing_points = self.slicing_points.astype(int)

    def __call__(self, img, *args, **kwargs):

        ret = [img[:, i, :, :] for i in self.slicing_points] + \
            [img[:, :, i, :] for i in self.slicing_points] + \
            [img[:, :, :, i] for i in self.slicing_points]

        return ret


class SingleSlices(Transform):
    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __init__(self, image_size: int = 72, slice_view: str = 'sagittal'):
        self.image_size = image_size
        self.center = image_size // 2
        assert slice_view in ['sagittal', 'coronal', 'axial']
        self.slice_view = slice_view

    def __call__(self, img, *args, **kwargs):
        ret = []
        if self.slice_view == 'sagittal':
            ret.append(img[:, self.center, :, :])
        elif self.slice_view == 'coronal':
            ret.append(img[:, :, self.center, :])
        elif self.slice_view == 'axial':
            ret.append(img[:, :, :, self.center])
        return ret


def make_mri_transforms(image_size_mri: int = 72,
                        intensity_mri: str = 'simple',
                        crop_size_mri: int = None,
                        rotate_mri: bool = True,
                        flip_mri: bool = True,
                        affine_mri: bool = True,
                        blur_std_mri: float = 0.1,
                        train_slices: str = 'random',
                        num_slices: int = 5,
                        slice_range: float = 0.15,
                        space: int = 3,
                        n_points: int = 5,
                        prob: float = 0.5,
                        **kwargs):

    base_transform = [ToTensor(),
                      AddChannel(),
                      Resize((image_size_mri, image_size_mri, image_size_mri))]
    if intensity_mri is None:
        pass
    elif intensity_mri == 'scale':
        base_transform.insert(1, ScaleIntensity())
    elif intensity_mri == 'normalize':
        base_transform.insert(1, NormalizeIntensity(nonzero=True))
    elif intensity_mri == 'simple':
        base_transform.insert(1, Divide())
    else:
        raise NotImplementedError

    train_transform, test_transform = base_transform.copy(), base_transform.copy()

    if crop_size_mri:
        # train_transform.append(RandSpatialCrop(roi_size=(cropsize, cropsize, cropsize), random_size=False))
        train_transform.append(CenterSpatialCrop(roi_size=(crop_size_mri, crop_size_mri, crop_size_mri)))
        test_transform.append(CenterSpatialCrop(roi_size=(crop_size_mri, crop_size_mri, crop_size_mri)))

    if rotate_mri:
        train_transform.append(RandRotate90(prob=prob))
    if flip_mri:
        train_transform.append(RandFlip(prob=prob))
    if blur_std_mri:
        train_transform.append(RandGaussianNoise(prob=prob, std=blur_std_mri))

    if crop_size_mri is not None:
        slice_size = crop_size_mri
    else:
        slice_size = image_size_mri

    # slice - training
    if train_slices == 'random':
        train_transform.append(RandomSlices(num_slices=num_slices,
                                            image_size=slice_size,
                                            slice_range=slice_range))
    elif train_slices == 'fixed':
        train_transform.append(FixedSlices(image_size=slice_size, space=space, n_points=n_points))
    elif train_slices in ['sagittal', 'coronal', 'axial']:
        train_transform.append(SingleSlices(image_size=slice_size, slice_view=train_slices))
    else:
        raise ValueError

    if affine_mri:
        import warnings
        warnings.filterwarnings("ignore")
        train_transform.append(RandAffine(rotate_range=(-5.0, 5.0),
                                          translate_range=(-3.0, 3.0),
                                          # scale_range=(0.95, 1.05),
                                          prob=prob))

    # slice - testing
    if train_slices in ['random', 'fixed']:
        test_transform.append(FixedSlices(image_size=slice_size, space=space, n_points=n_points))
    elif train_slices in ['sagittal', 'coronal', 'axial']:
        test_transform.append(SingleSlices(image_size=slice_size, slice_view=train_slices))
    else:
        raise ValueError

    train_transform.append(ConvertImageDtype(torch.float32))
    test_transform.append(ConvertImageDtype(torch.float32))

    return Compose(train_transform), Compose(test_transform)


def make_pet_transforms(image_size_pet: int = 72,
                        intensity_pet: str = 'scale',
                        crop_size_pet: int = None,
                        rotate_pet: bool = True,
                        flip_pet: bool = True,
                        affine_pet: bool = True,
                        blur_std_pet: float = 0.1,
                        train_slices: str = 'random',
                        num_slices: int = 5,
                        slice_range: float = 0.15,
                        space: int = 3,
                        n_points: int = 5,
                        prob: float = 0.5,
                        **kwargs):

    base_transform = [ToTensor(),
                      AddChannel(),
                      Resize((image_size_pet, image_size_pet, image_size_pet))]
    if intensity_pet is None:
        pass
    elif intensity_pet == 'scale':
        base_transform.insert(1, ScaleIntensity())
    elif intensity_pet == 'normalize':
        base_transform.insert(1, NormalizeIntensity(nonzero=True))
    else:
        raise NotImplementedError

    train_transform, test_transform = base_transform.copy(), base_transform.copy()

    if crop_size_pet:
        # train_transform.append(RandSpatialCrop(roi_size=(cropsize, cropsize, cropsize), random_size=False))
        train_transform.append(CenterSpatialCrop(roi_size=(crop_size_pet, crop_size_pet, crop_size_pet)))
        test_transform.append(CenterSpatialCrop(roi_size=(crop_size_pet, crop_size_pet, crop_size_pet)))

    if rotate_pet:
        train_transform.append(RandRotate90(prob=prob))
    if flip_pet:
        train_transform.append(RandFlip(prob=prob))
    if blur_std_pet:
        train_transform.append(RandGaussianNoise(prob=prob, std=blur_std_pet))

    if crop_size_pet is not None:
        slice_size = crop_size_pet
    else:
        slice_size = image_size_pet

    # slice - training
    if train_slices == 'random':
        train_transform.append(RandomSlices(num_slices=num_slices,
                                            image_size=slice_size,
                                            slice_range=slice_range))
    elif train_slices == 'fixed':
        train_transform.append(FixedSlices(image_size=slice_size, space=space, n_points=n_points))
    elif train_slices in ['sagittal', 'coronal', 'axial']:
        train_transform.append(SingleSlices(image_size=slice_size, slice_view=train_slices))
    else:
        raise ValueError

    if affine_pet:
        import warnings
        warnings.filterwarnings("ignore")
        train_transform.append(RandAffine(rotate_range=(-5.0, 5.0),
                                          translate_range=(-3.0, 3.0),
                                          # scale_range=(0.95, 1.05),
                                          prob=prob))

    # slice - testing
    if train_slices in ['random', 'fixed']:
        test_transform.append(FixedSlices(image_size=slice_size, space=space, n_points=n_points))
    elif train_slices in ['sagittal', 'coronal', 'axial']:
        test_transform.append(SingleSlices(image_size=slice_size, slice_view=train_slices))
    else:
        raise ValueError

    train_transform.append(ConvertImageDtype(torch.float32))
    test_transform.append(ConvertImageDtype(torch.float32))

    return Compose(train_transform), Compose(test_transform)


if __name__ == '__main__':
    config = {
        'image_size_mri': 96,
        'intensity_mri': 'simple',
        'crop_size_mri': 64,
        'rotate_mri': True,
        'flip_mri': True,
        'affine_mri': False,
        'blur_std_mri': False,
        'train_slices': 'random',
        'num_slices': 5,
        'slice_range': 0.10,
        'prob': 0.5,
        'others': 'others'
    }
    train_trans, test_trans = make_mri_transforms(**config)
    print(train_trans.transforms)
