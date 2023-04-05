import torch
from torch import Tensor
from torchvision.utils import _log_api_usage_once
from monai import transforms


# TODO: add RandAffine(shear and translate)
def make_mri_transforms(image_size: int = 96, crop_size: int = None):

    # TODO: replace AddChannel() with EnsureChannelFirst()
    test_transform = [
        transforms.ToTensor(),
        transforms.AddChannel(),
        Clamp(min=0.0),
        transforms.NormalizeIntensity(nonzero=True),
        # transforms.ScaleIntensityRange(a_min=0.0, a_max=1.0, b_min=0.0, b_max=1.0, clip=True),
        transforms.Resize(spatial_size=(image_size, image_size, image_size)),
    ]

    if crop_size:
        # TODO: select randspatial / center
        test_transform.append(transforms.RandSpatialCrop(roi_size=(crop_size, crop_size, crop_size)))

    train_transform = test_transform.copy()
    train_transform.extend(
        [
            transforms.RandFlip(prob=0.5, spatial_axis=0),
            transforms.RandFlip(prob=0.5, spatial_axis=1),
            transforms.RandFlip(prob=0.5, spatial_axis=2),
            transforms.RandAffine(prob=0.5, rotate_range=(5.0, 15.0),
                                  shear_range=(0.3, 0.8),
                                  translate_range=(5.0, 10.0)),
            transforms.RandScaleIntensity(factors=0.1, prob=1.0),
            transforms.RandShiftIntensity(offsets=0.1, prob=1.0)
        ]
    )

    train_transform, test_transform = transforms.Compose(train_transform), transforms.Compose(test_transform)

    return train_transform, test_transform


def make_pet_transforms(image_size: int = 96, crop_size: int = None):

    test_transform = [
        transforms.ToTensor(),
        transforms.AddChannel(),
        Clamp(min=0.0),
        transforms.NormalizeIntensity(nonzero=True),
        # transforms.ScaleIntensityRange(a_min=0.0, a_max=1.0, b_min=0.0, b_max=1.0, clip=True),
        transforms.Resize(spatial_size=(image_size, image_size, image_size))
    ]

    if crop_size:
        test_transform.append(transforms.RandSpatialCrop(roi_size=(crop_size, crop_size, crop_size)))

    train_transform = test_transform.copy()
    train_transform.extend(
        [
            transforms.RandFlip(prob=0.5, spatial_axis=0),
            transforms.RandFlip(prob=0.5, spatial_axis=1),
            transforms.RandFlip(prob=0.5, spatial_axis=2),
            transforms.RandAffine(prob=0.5, rotate_range=(5.0, 15.0),
                                  shear_range=(0.3, 0.8),
                                  translate_range=(5.0, 10.0)),
            transforms.RandScaleIntensity(factors=0.1, prob=1.0),
            transforms.RandShiftIntensity(offsets=0.1, prob=1.0)
        ]
    )

    train_transform, test_transform = transforms.Compose(train_transform), transforms.Compose(test_transform)

    return train_transform, test_transform


class Clamp(torch.nn.Module):
    """Clamp a tensor image"""

    def __init__(self, min=None, max=None):
        super().__init__()
        _log_api_usage_once(self)
        self.min = min
        self.max = max

    def forward(self, tensor: Tensor) -> Tensor:
        out = torch.clamp(input=tensor, min=self.min, max=self.max)
        return out

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(min={self.min}, max={self.max})"


if __name__ == '__main__':

    train_transform_mri, test_transform_mri = make_mri_transforms(image_size=96)
    train_transform_pet, test_transform_pet = make_pet_transforms(image_size=96)

    from torch.utils.data import DataLoader
    from dataset.brain import BrainProcessor, Brain

    processor = BrainProcessor(root='D:/data/ADNI',
                               data_file='labels/data_info_multi.csv',
                               pet_type='FDG',
                               random_state=2021)
    datasets = processor.process(test_size=0.2, missing_rate=None)

    mri_pet_complete_train = datasets['mri_pet_complete_train']

    pet_set = Brain(mri_pet_complete_train, 'pet', None, test_transform_pet)
    pet_loader = DataLoader(pet_set, batch_size=1)
    rotate_range=(5.0, 15.0)
    translate_range = (5.0, 10.0)
    pet = next(pet_loader.__iter__())['pet'].squeeze().numpy()
    trans = transforms.RandAffine(prob=1.0, translate_range=translate_range)
    pet_t = trans(pet).numpy()

    import seaborn as sns
    import matplotlib.pyplot as plt

    sns.heatmap(pet[48, :, :], cmap='binary')
    plt.show()
    sns.heatmap(pet_t[48, :, :], cmap='binary')
    plt.show()