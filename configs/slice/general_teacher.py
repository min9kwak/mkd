import argparse
from configs.base import ConfigBase, str2bool


class SliceGeneralTeacher(ConfigBase):

    def __init__(self, args=None, **kwargs):
        super(SliceGeneralTeacher, self).__init__(args, **kwargs)

    @staticmethod
    def data_parser() -> argparse.ArgumentParser:
        """Returns an `argparse.ArgumentParser` instance containing data-related arguments."""

        parser = argparse.ArgumentParser("Data", add_help=False)
        parser.add_argument('--data_file', type=str, default='labels/data_info_multi.csv')
        parser.add_argument('--pet_type', type=str, choices=('FDG', 'FBP'), default='FBP')
        parser.add_argument('--mci_only', type=str2bool, default=False)

        parser.add_argument('--random_state', type=int, default=2023)
        parser.add_argument('--validation_size', type=float, default=0.1)
        parser.add_argument('--test_size', type=float, default=0.1)
        parser.add_argument('--missing_rate', type=float)

        # MRI augmentation
        parser.add_argument('--mri_type', type=str, default='template', choices=('individual', 'template'))
        parser.add_argument('--image_size_mri', type=int, default=72)
        parser.add_argument('--intensity_mri', type=str, default='scale', choices=('scale', 'normalize', 'simple'))
        parser.add_argument('--crop_size_mri', type=int)
        parser.add_argument('--rotate_mri', type=str2bool, default=True)
        parser.add_argument('--flip_mri', type=str2bool, default=True)
        parser.add_argument('--affine_mri', type=str2bool, default=False)
        parser.add_argument('--blur_std_mri', type=float)

        # PET augmentation
        parser.add_argument('--image_size_pet', type=int, default=72)
        parser.add_argument('--intensity_pet', type=str, default='scale', choices=('scale', 'normalize'))
        parser.add_argument('--crop_size_pet', type=int)
        parser.add_argument('--rotate_pet', type=str2bool, default=True)
        parser.add_argument('--flip_pet', type=str2bool, default=True)
        parser.add_argument('--affine_pet', type=str2bool, default=False)
        parser.add_argument('--blur_std_pet', type=float)

        parser.add_argument('--prob', type=float, default=0.5)

        # slice
        parser.add_argument('--train_slices', type=str, default='fixed',
                            choices=('random', 'fixed', 'sagittal', 'coronal', 'axial'))
        parser.add_argument('--num_slices', type=int, default=5)
        parser.add_argument('--slice_range', type=float, default=0.15)

        return parser

    @staticmethod
    def model_parser() -> argparse.ArgumentParser:
        """Returns an `argparse.ArgumentParser` instance containing model-related arguments."""
        parser = argparse.ArgumentParser("Model Configuration", add_help=False)
        parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint file to resume training from.')

        # Network Type
        parser.add_argument('--extractor_type', type=str, default='resnet18')
        parser.add_argument('--small_kernel', type=str2bool, default=True)

        # Projector, Encoder, and Decoder
        parser.add_argument('--hidden', type=int, default=128)
        parser.add_argument('--swap', type=str2bool, default=False)

        # Classifier
        parser.add_argument('--mlp', type=str2bool, default=False) # classifier_type - linear, mlp, transformer
        parser.add_argument('--dropout', type=float, default=0.0)

        # Optional
        parser.add_argument('--use_projector', type=str2bool, default=True)
        parser.add_argument('--use_specific', type=str2bool, default=False)
        parser.add_argument('--use_transformer', type=str2bool, default=True)
        parser.add_argument('--ce_only', type=str2bool, default=False)
        parser.add_argument('--add_type', type=str, default='add', choices=('concat', 'add'))

        return parser

    @staticmethod
    def train_parser() -> argparse.ArgumentParser:
        """Returns an `argparse.ArgumentParser` instance containing training-related arguments."""
        parser = argparse.ArgumentParser("Model Training", add_help=False)
        parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs.')
        parser.add_argument('--batch_size', type=int, default=16, help='Mini-batch size.')
        parser.add_argument('--num_workers', type=int, default=4, help='Number of CPU threads.')

        # Optimizer: MRI & PET
        parser.add_argument('--optimizer', type=str, default='adamw', choices=('sgd', 'adamw'),
                            help='Optimization algorithm.')
        parser.add_argument('--learning_rate', type=float, default=0.0001, help='Base learning rate to start from.')
        parser.add_argument('--weight_decay', type=float, default=0.0001, help='Weight decay factor.')

        # Scheduler
        parser.add_argument('--cosine_warmup', type=int, default=0,
                            help='Number of warmups before cosine LR scheduling (-1 to disable.)')
        parser.add_argument('--cosine_cycles', type=int, default=1,
                            help='Number of hard cosine LR cycles with hard restarts.')
        parser.add_argument('--cosine_min_lr', type=float, default=0.0,
                            help='LR lower bound when cosine scheduling is used.')
        parser.add_argument('--mixed_precision', type=str2bool, default=True, help='Use float16 precision.')

        return parser

    @staticmethod
    def task_specific_parser() -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser('Distillation', add_help=False)
        parser.add_argument('--balance', type=str2bool, default=True, help='apply class balance weight')

        # Weight Alpha
        parser.add_argument('--alpha_sim', type=float, default=1.0)
        parser.add_argument('--alpha_diff', type=float, default=0.5)
        parser.add_argument('--alpha_recon', type=float, default=1.0)

        # Loss
        parser.add_argument('--loss_diff', type=str, default='diff', choices=('diff', 'mse'))
        parser.add_argument('--n_moments', type=int, default=5)

        return parser
