import argparse
from configs.base import ConfigBase


class IMFiConfig(ConfigBase):

    def __init__(self, args=None, **kwargs):
        super(IMFiConfig, self).__init__(args, **kwargs)

    @staticmethod
    def data_parser() -> argparse.ArgumentParser:
        """Returns an `argparse.ArgumentParser` instance containing data-related arguments."""

        parser = argparse.ArgumentParser("Data", add_help=False)
        parser.add_argument('--root', type=str, default='D:/data/ADNI')
        parser.add_argument('--data_file', type=str, default='labels/data_info_multi.csv')
        parser.add_argument('--pet_type', type=str, choices=('FDG', 'FBP'), default='FBP')

        parser.add_argument('--mri_image_size', type=int, default=96)
        parser.add_argument('--mri_crop_size', type=int)

        parser.add_argument('--pet_image_size', type=int, default=96)
        parser.add_argument('--pet_crop_size', type=int)

        parser.add_argument('--random_state', type=int, default=2022)
        parser.add_argument('--test_size', type=float, default=0.2)
        parser.add_argument('--missing_rate', type=float)

        return parser

    @staticmethod
    def model_parser() -> argparse.ArgumentParser:
        """Returns an `argparse.ArgumentParser` instance containing model-related arguments."""
        parser = argparse.ArgumentParser("Model Configuration", add_help=False)
        parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint file to resume training from.')

        # Network Type
        # MRI
        parser.add_argument('--mri_network_type', type=str, choices=('resnet', 'densenet', 'swin'), default='swin')
        parser.add_argument('--mri_architecture', type=str, choices=('18', '50', 'base', 'small', 'tiny'),
                            default='small')
        parser.add_argument('--mri_pretrained_path', type=str, help='swin/base_ct.pt or swin/base_mri.pt') # Swin
        parser.add_argument('--mri_feature_layer', type=str, choices=('bottleneck', 'last')) # Swin
        parser.add_argument('--mri_small_kernel', action='store_true') # ResNet
        parser.add_argument('--mri_projector_type', type=str, choices=('linear', 'mlp'), default='mlp')

        # PET
        parser.add_argument('--pet_network_type', type=str, choices=('resnet', 'densenet', 'swin'), default='swin')
        parser.add_argument('--pet_architecture', type=str, choices=('18', '50', 'base', 'small', 'tiny'),
                            default='base')
        parser.add_argument('--pet_feature_layer', type=str, choices=('bottleneck', 'last'))  # Swin
        parser.add_argument('--pet_pretrained_path', type=str, help='swin/base_ct.pt or swin/base_mri.pt') # Swin
        parser.add_argument('--pet_small_kernel', action='store_true')  # ResNet
        parser.add_argument('--pet_projector_type', type=str, choices=('linear', 'mlp'), default='mlp')

        # Common and Both
        parser.add_argument('--common_type', type=str, default='concat', choices=('add', 'concat'))
        parser.add_argument('--common_projector_type', type=str, choices=('linear', 'mlp'), default='mlp')
        parser.add_argument('--projector_dim', type=int, default=128)

        # Predictor
        parser.add_argument('--predictor_type', type=str, choices=('linear', 'mlp'), default='mlp')

        # loss
        parser.add_argument('--cosine_ratio', type=float, default=1.0)

        return parser

    @staticmethod
    def task_specific_parser() -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser('IMF', add_help=False)
        return parser
