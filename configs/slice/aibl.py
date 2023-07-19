# -*- coding: utf-8 -*-

import os
import copy
import json
import argparse
import datetime


class AIBLConfig(object):
    def __init__(self, args: argparse.Namespace = None, **kwargs):

        if isinstance(args, dict):
            attrs = args
        elif isinstance(args, argparse.Namespace):
            attrs = copy.deepcopy(vars(args))
        else:
            attrs = dict()

        if kwargs:
            attrs.update(kwargs)
        for k, v in attrs.items():
            setattr(self, k, v)

        if not hasattr(self, 'hash'):
            self.hash = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        self._task = None

    @classmethod
    def parse_arguments(cls) -> argparse.Namespace:
        """Create a configuration object from command line arguments."""
        parents = [
            cls.ddp_parser(),            # task-agnostic
            cls.train_parser(),          # task-agnostic
            cls.logging_parser(),        # task-agnostic
            cls.finetune_parser()
        ]

        parser = argparse.ArgumentParser(add_help=True, parents=parents, fromfile_prefix_chars='@')
        parser.convert_arg_line_to_args = cls.convert_arg_line_to_args

        config = cls()
        parser.parse_args(namespace=config)  # sets parsed arguments as attributes of namespace

        return config

    @classmethod
    def from_json(cls, json_path: str):
        """Create a configuration object from a .json file."""
        with open(json_path, 'r') as f:
            configs = json.load(f)

        return cls(args=configs)

    def save(self, path: str = None):
        """Save configurations to a .json file."""
        if path is None:
            path = os.path.join(self.checkpoint_dir, 'configs.json')
        os.makedirs(os.path.dirname(path), exist_ok=True)

        attrs = copy.deepcopy(vars(self))
        attrs['task'] = self.task
        attrs['checkpoint_dir'] = self.checkpoint_dir

        with open(path, 'w') as f:
            json.dump(attrs, f, indent=2)

    @property
    def task(self):
        return self._task

    @task.setter
    def task(self, value):
        self._task = value

    @property
    def checkpoint_dir(self) -> str:
        ckpt = os.path.join(
            self.checkpoint_root,
            self.task,          # 'mri', 'pet',
            self.hash           # ...
            )
        os.makedirs(ckpt, exist_ok=True)
        return ckpt

    @staticmethod
    def convert_arg_line_to_args(arg_line):
        for arg in arg_line.split():
            if not arg.strip():
                continue
            yield arg

    @staticmethod
    def ddp_parser() -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser("Data Distributed Training", add_help=False)
        parser.add_argument('--gpus', type=int, nargs='+', default=None, help='')
        parser.add_argument('--server', type=str, choices=('main', 'workstation2', 'workstation3'))
        parser.add_argument('--num_nodes', type=int, default=1, help='')
        parser.add_argument('--node_rank', type=int, default=0, help='')
        parser.add_argument('--dist_url', type=str, default='tcp://127.0.0.1:3500', help='')
        parser.add_argument('--dist_backend', type=str, default='nccl', help='')
        return parser

    @staticmethod
    def train_parser() -> argparse.ArgumentParser:
        """Returns an `argparse.ArgumentParser` instance containing training-related arguments."""
        parser = argparse.ArgumentParser("Model Training", add_help=False)
        parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs.')
        parser.add_argument('--batch_size', type=int, default=4, help='Mini-batch size.')
        parser.add_argument('--num_workers', type=int, default=4, help='Number of CPU threads.')
        parser.add_argument('--optimizer', type=str, default='sgd', choices=('sgd', 'adamw'), help='Optimization algorithm.')
        parser.add_argument('--learning_rate', type=float, default=0.01, help='Base learning rate to start from.')
        parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay factor.')
        parser.add_argument('--cosine_warmup', type=int, default=0, help='Number of warmups before cosine LR scheduling (-1 to disable.)')
        parser.add_argument('--cosine_cycles', type=int, default=1, help='Number of hard cosine LR cycles with hard restarts.')
        parser.add_argument('--cosine_min_lr', type=float, default=0.0, help='LR lower bound when cosine scheduling is used.')
        parser.add_argument('--mixed_precision', action='store_true', help='Use float16 precision.')
        return parser

    @staticmethod
    def logging_parser() -> argparse.ArgumentParser:
        """Returns an `argparse.ArgumentParser` instance containing logging-related arguments."""
        parser = argparse.ArgumentParser("Logging", add_help=False)
        parser.add_argument('--checkpoint_root', type=str, default='./checkpoints/', help='Top-level directory of checkpoints.')
        parser.add_argument('--save_every', type=int, default=100, help='Save model checkpoint every `save_every` epochs.')
        parser.add_argument('--enable_wandb', action='store_true', help='Use Weights & Biases plugin.')
        return parser

    @staticmethod
    def finetune_parser() -> argparse.ArgumentParser:
        """Returns an `argparse.ArgumentParser` instance containing logging-related arguments."""
        parser = argparse.ArgumentParser("Finetune", add_help=False)
        parser.add_argument('--student_dir', type=str)
        parser.add_argument('--student_position', type=str, default='last')

        parser.add_argument('--balance', action='store_true', help='apply class balance weight')

        # data
        parser.add_argument('--data_info', type=str, default='data_info_mri.csv')
        parser.add_argument('--time_window', type=int, default=36, choices=(18, 36))
        parser.add_argument('--random_state', type=int, default=2021)
        parser.add_argument('--n_splits', type=int, default=5)
        parser.add_argument('--n_cv', type=int, default=0)

        parser.add_argument('--train_mode', type=str, default='train', choices=('train', 'test'))
        return parser


class DemoAIBLConfig(object):
    def __init__(self, args: argparse.Namespace = None, **kwargs):

        if isinstance(args, dict):
            attrs = args
        elif isinstance(args, argparse.Namespace):
            attrs = copy.deepcopy(vars(args))
        else:
            attrs = dict()

        if kwargs:
            attrs.update(kwargs)
        for k, v in attrs.items():
            setattr(self, k, v)

        if not hasattr(self, 'hash'):
            self.hash = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        self._task = None

    @classmethod
    def parse_arguments(cls) -> argparse.Namespace:
        """Create a configuration object from command line arguments."""
        parents = [
            cls.ddp_parser(),  # task-agnostic
            cls.train_parser(),  # task-agnostic
            cls.logging_parser(),  # task-agnostic
            cls.finetune_parser()
        ]

        parser = argparse.ArgumentParser(add_help=True, parents=parents, fromfile_prefix_chars='@')
        parser.convert_arg_line_to_args = cls.convert_arg_line_to_args

        config = cls()
        parser.parse_args(namespace=config)  # sets parsed arguments as attributes of namespace

        return config

    @classmethod
    def from_json(cls, json_path: str):
        """Create a configuration object from a .json file."""
        with open(json_path, 'r') as f:
            configs = json.load(f)

        return cls(args=configs)

    def save(self, path: str = None):
        """Save configurations to a .json file."""
        if path is None:
            path = os.path.join(self.checkpoint_dir, 'configs.json')
        os.makedirs(os.path.dirname(path), exist_ok=True)

        attrs = copy.deepcopy(vars(self))
        attrs['task'] = self.task
        attrs['checkpoint_dir'] = self.checkpoint_dir

        with open(path, 'w') as f:
            json.dump(attrs, f, indent=2)

    @property
    def task(self):
        return self._task

    @task.setter
    def task(self, value):
        self._task = value

    @property
    def model_name(self) -> str:
        return self.backbone_type

    @property
    def checkpoint_dir(self) -> str:
        ckpt = os.path.join(
            self.checkpoint_root,
            self.task,  # 'mri', 'pet',
            self.model_name,  # 'densenet', 'resnet'
            self.hash  # ...
        )
        os.makedirs(ckpt, exist_ok=True)
        return ckpt

    @staticmethod
    def convert_arg_line_to_args(arg_line):
        for arg in arg_line.split():
            if not arg.strip():
                continue
            yield arg

    @staticmethod
    def ddp_parser() -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser("Data Distributed Training", add_help=False)
        parser.add_argument('--gpus', type=int, nargs='+', default=None, help='')
        parser.add_argument('--server', type=str, choices=('main', 'workstation2', 'workstation3'))
        parser.add_argument('--num_nodes', type=int, default=1, help='')
        parser.add_argument('--node_rank', type=int, default=0, help='')
        parser.add_argument('--dist_url', type=str, default='tcp://127.0.0.1:3500', help='')
        parser.add_argument('--dist_backend', type=str, default='nccl', help='')
        return parser

    @staticmethod
    def train_parser() -> argparse.ArgumentParser:
        """Returns an `argparse.ArgumentParser` instance containing training-related arguments."""
        parser = argparse.ArgumentParser("Model Training", add_help=False)
        parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs.')
        parser.add_argument('--batch_size', type=int, default=4, help='Mini-batch size.')
        parser.add_argument('--num_workers', type=int, default=4, help='Number of CPU threads.')
        parser.add_argument('--optimizer', type=str, default='sgd', choices=('sgd', 'adamw'),
                            help='Optimization algorithm.')
        parser.add_argument('--learning_rate', type=float, default=0.01, help='Base learning rate to start from.')
        parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay factor.')
        parser.add_argument('--cosine_warmup', type=int, default=0,
                            help='Number of warmups before cosine LR scheduling (-1 to disable.)')
        parser.add_argument('--cosine_cycles', type=int, default=1,
                            help='Number of hard cosine LR cycles with hard restarts.')
        parser.add_argument('--cosine_min_lr', type=float, default=0.0,
                            help='LR lower bound when cosine scheduling is used.')
        parser.add_argument('--mixed_precision', action='store_true', help='Use float16 precision.')
        return parser

    @staticmethod
    def logging_parser() -> argparse.ArgumentParser:
        """Returns an `argparse.ArgumentParser` instance containing logging-related arguments."""
        parser = argparse.ArgumentParser("Logging", add_help=False)
        parser.add_argument('--checkpoint_root', type=str, default='./checkpoints/',
                            help='Top-level directory of checkpoints.')
        parser.add_argument('--save_every', type=int, default=100,
                            help='Save model checkpoint every `save_every` epochs.')
        parser.add_argument('--enable_wandb', action='store_true', help='Use Weights & Biases plugin.')
        return parser

    @staticmethod
    def finetune_parser() -> argparse.ArgumentParser:
        """Returns an `argparse.ArgumentParser` instance containing logging-related arguments."""
        parser = argparse.ArgumentParser("Finetune", add_help=False)
        parser.add_argument('--pretrained_dir', type=str, default=None, help='Path to pretrained model file (.pt).')
        parser.add_argument('--freeze_bn', action='store_true', help='Freeze BN weights of CNN backbone.')
        parser.add_argument('--balance', action='store_true', help='apply class balance weight')
        parser.add_argument('--finetune_trans', type=str, default='test', choices=('train', 'test'))

        parser.add_argument('--hidden', type=str, default="3")
        parser.add_argument('--add_apoe', action='store_true')
        parser.add_argument('--add_volume', action='store_true')


        # data
        parser.add_argument('--root', type=str, default='/raidWorkspace/mingu/Data/AIBL')
        parser.add_argument('--data_info', type=str, default='data_info.csv')
        parser.add_argument('--time_window', type=int, default=36, choices=(18, 36))
        parser.add_argument('--random_state', type=int, default=2021)
        parser.add_argument('--n_splits', type=int, default=5)
        parser.add_argument('--n_cv', type=int, default=0)

        parser.add_argument('--train_mode', type=str, default='train', choices=('train', 'test'))

        return parser
