# -*- coding: utf-8 -*-

import os
import copy
import json
import argparse
import datetime


def str2bool(v):
    """
    string to boolean
    ex) parser.add_argument('--argument', type=str2bool, default=True)
    """
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


class ConfigSimulation(object):
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
            cls.data_parser(),           # task-agnostic
            cls.model_parser(),          # task-agnostic
            cls.train_parser(),          # task-agnostic
            cls.logging_parser(),        # task-agnostic
            cls.task_specific_parser(),  # task-specific
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
    def task_specific_parser() -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser("Simulation Task", add_help=False)
        parser.add_argument('--alpha_ce', type=float, default=1.0)
        parser.add_argument('--alpha_sim', type=float, default=10.0)
        parser.add_argument('--alpha_diff', type=float, default=5.0)
        parser.add_argument('--alpha_recon', type=float, default=0.1)
        parser.add_argument('--alpha_kd_clf', type=float, default=100.0)
        parser.add_argument('--alpha_kd_repr', type=float, default=500.0)

        parser.add_argument('--temperature', type=float, default=5.0)

        return parser

    @staticmethod
    def convert_arg_line_to_args(arg_line):
        for arg in arg_line.split():
            if not arg.strip():
                continue
            yield arg

    @staticmethod
    def ddp_parser() -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser("Data Distributed Training", add_help=False)
        parser.add_argument('--gpus', type=str, nargs='+', default=None, help='')
        parser.add_argument('--server', type=str, choices=('main', 'workstation1', 'workstation2', 'workstation3'))
        parser.add_argument('--num_nodes', type=int, default=1, help='')
        parser.add_argument('--node_rank', type=int, default=0, help='')
        parser.add_argument('--dist_url', type=str, default='tcp://127.0.0.1:3500', help='')
        parser.add_argument('--dist_backend', type=str, default='nccl', help='')
        return parser

    @staticmethod
    def data_parser() -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser("Simulation Data", add_help=False)
        parser.add_argument('--n_train', type=int, default=200)
        parser.add_argument('--n_test', type=int, default=1000)

        parser.add_argument('--x_dim', type=int, default=50)
        parser.add_argument('--xs_dim', type=int, default=10)
        # parser.add_argument('--x1_dim', type=int, default=50)
        # parser.add_argument('--x2_dim', type=int, default=50)
        # parser.add_argument('--xs1_dim', type=int, default=10)
        # parser.add_argument('--xs2_dim', type=int, default=10)
        parser.add_argument('--overlap_dim', type=int, default=10)

        parser.add_argument('--hyperplane_dim', type=int, default=500)
        parser.add_argument('--missing_rate', type=float, default=0.5, help='-1 means None')

        parser.add_argument('--random_state', type=int, default=2021)

        return parser

    @staticmethod
    def train_parser() -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser("Model Training", add_help=False)

        # common
        parser.add_argument('--batch_size', type=int, default=16, help='Mini-batch size.')
        parser.add_argument('--num_workers', type=int, default=4, help='Number of CPU threads.')
        parser.add_argument('--optimizer', type=str, default='adamw', choices=('sgd', 'adamw'),
                            help='Optimization algorithm.')

        parser.add_argument('--cosine_warmup', type=int, default=0,
                            help='Number of warmups before cosine LR scheduling (-1 to disable.)')
        parser.add_argument('--cosine_cycles', type=int, default=1,
                            help='Number of hard cosine LR cycles with hard restarts.')
        parser.add_argument('--cosine_min_lr', type=float, default=0.0,
                            help='LR lower bound when cosine scheduling is used.')
        parser.add_argument('--mixed_precision', type=str2bool, default=False, help='Use float16 precision.')

        # train level
        parser.add_argument('--train_level', type=int, default=2,
                            choices=(0, 1, 2),
                            help='0: teacher, 1: teacher+kd, 2: teacher+kd+final')

        # teacher
        parser.add_argument('--epochs_teacher', type=int, default=1000, help='Number of training epochs.')
        parser.add_argument('--learning_rate_teacher', type=float, default=0.1, help='Base learning rate to start from.')
        parser.add_argument('--weight_decay_teacher', type=float, default=0.0001, help='Weight decay factor.')

        # teacher
        parser.add_argument('--epochs_kd', type=int, default=10, help='Number of training epochs.')
        parser.add_argument('--learning_rate_kd', type=float, default=0.01, help='Base learning rate to start from.')
        parser.add_argument('--weight_decay_kd', type=float, default=0.0001, help='Weight decay factor.')

        # teacher
        parser.add_argument('--epochs_final', type=int, default=1000, help='Number of training epochs.')
        parser.add_argument('--learning_rate_final', type=float, default=0.1, help='Base learning rate to start from.')
        parser.add_argument('--weight_decay_final', type=float, default=0.0001, help='Weight decay factor.')

        return parser

    @staticmethod
    def model_parser() -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser("Simulation Data", add_help=False)
        parser.add_argument('--hidden', type=int, default=25)

        return parser

    @staticmethod
    def logging_parser() -> argparse.ArgumentParser:
        """Returns an `argparse.ArgumentParser` instance containing logging-related arguments."""
        parser = argparse.ArgumentParser("Logging", add_help=False)
        parser.add_argument('--checkpoint_root', type=str, default='./checkpoints/', help='Top-level directory of checkpoints.')
        parser.add_argument('--save_every', type=int, default=2000, help='Save model checkpoint every `save_every` epochs.')
        parser.add_argument('--enable_wandb', type=str2bool, default=True, help='Use Weights & Biases plugin.')
        parser.add_argument('--save_log', type=str2bool, default=True)
        return parser
