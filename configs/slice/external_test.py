# -*- coding: utf-8 -*-

import argparse
from configs.base import ConfigBase, str2bool


class SliceExternalTestConfig(ConfigBase):
    def __init__(self, args=None, **kwargs):
        super(SliceExternalTestConfig, self).__init__(args, **kwargs)

    @staticmethod
    def data_parser() -> argparse.ArgumentParser:
        """Returns an `argparse.ArgumentParser` instance containing data-related arguments."""

        parser = argparse.ArgumentParser("Data", add_help=False)
        parser.add_argument('--data_info', type=str, default='aibl_oasis_data_info_new.csv')
        parser.add_argument('--n_splits', type=int, default=5)
        parser.add_argument('--n_cv', type=int, default=0)

        return parser

    @staticmethod
    def model_parser() -> argparse.ArgumentParser:
        """Returns an `argparse.ArgumentParser` instance containing model-related arguments."""
        parser = argparse.ArgumentParser("Model Configuration", add_help=False)
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
        parser.add_argument('--learning_rate', type=float, default=0.001, help='Base learning rate to start from.')
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
        parser = argparse.ArgumentParser('General Teacher', add_help=False)

        parser.add_argument('--pretrained_dir', type=str)
        parser.add_argument('--pretrained_task', type=str, choices=('multi_demo', 'single_demo', 'student', 'final'))
        parser.add_argument('--external_data_type', type=str, default='mri', choices=('mri', 'mri+pib', 'mri+av45'))
        parser.add_argument('--train_mode', type=str, default='test', choices=('train', 'test'))

        return parser
