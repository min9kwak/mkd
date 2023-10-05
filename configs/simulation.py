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


def parse_numbers(numbers):
    return [int(num) for num in numbers.split(",")]


def handle_none(string: str):
    if string.lower() == "none":
        return None
    else:
        return string


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
        parser.add_argument('--alpha_sim_smt', type=float, default=5.0)
        parser.add_argument('--alpha_sim_final', type=float, default=5.0)
        parser.add_argument('--alpha_diff', type=float, default=1.0)
        parser.add_argument('--alpha_recon', type=float, default=10.0)
        parser.add_argument('--alpha_kd_clf', type=float, default=10.0)
        parser.add_argument('--alpha_kd_repr', type=float, default=10.0)

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
        parser.add_argument('--gpus', type=str, nargs='+', default='0', help='')
        parser.add_argument('--server', type=str, default='main',
                            choices=('main', 'workstation1', 'workstation2', 'workstation3'))
        parser.add_argument('--num_nodes', type=int, default=1, help='')
        parser.add_argument('--node_rank', type=int, default=0, help='')
        parser.add_argument('--dist_url', type=str, default='tcp://127.0.0.1:3500', help='')
        parser.add_argument('--dist_backend', type=str, default='nccl', help='')
        return parser

    @staticmethod
    def data_parser() -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser("Simulation Data", add_help=False)

        # representation
        parser.add_argument('--zs_dim', type=int, default=10)
        parser.add_argument('--z1_dim', type=int, default=10)
        parser.add_argument('--z2_dim', type=int, default=10)
        parser.add_argument('--rho', type=float, default=0.5)
        parser.add_argument('--sigma', type=float, default=1.0)

        # input
        parser.add_argument('--mu_0', type=float, default=0.0)
        parser.add_argument('--mu_1', type=float, default=1.0)
        parser.add_argument('--xs_dim', type=int, default=20)
        parser.add_argument('--x1_dim', type=int, default=20)
        parser.add_argument('--x2_dim', type=int, default=20)
        parser.add_argument('--slope', type=float, default=0.5)

        # samples
        parser.add_argument('--n_complete', type=int, default=1000)
        parser.add_argument('--n_incomplete', type=int, default=0)
        parser.add_argument('--n_validation', type=int, default=1000)
        parser.add_argument('--n_test', type=int, default=1000)

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
        parser.add_argument('--train_level', type=parse_numbers, default="1,2,3",
                            help='1: Single from scratch,'
                                 '2: SMT & SMT-Student & Final'
                                 '3: Multi & Multi-Student')

        # single from scratch
        parser.add_argument('--epochs_single', type=int, default=30, help='Number of training epochs.')
        parser.add_argument('--learning_rate_single', type=float, default=0.0001,
                            help='Base learning rate to start from.')

        # smt
        parser.add_argument('--epochs_smt', type=int, default=100, help='Number of training epochs.')
        parser.add_argument('--learning_rate_smt', type=float, default=0.001, help='Base learning rate to start from.')

        # smt-student
        parser.add_argument('--epochs_smt_student', type=int, default=30, help='Number of training epochs.')
        parser.add_argument('--learning_rate_smt_student', type=float, default=0.0001, help='Base learning rate to start from.')

        # final
        parser.add_argument('--epochs_final', type=int, default=100, help='Number of training epochs.')
        parser.add_argument('--learning_rate_final', type=float, default=0.001, help='Base learning rate to start from.')

        # multi
        parser.add_argument('--epochs_multi', type=int, default=30, help='Number of training epochs.')
        parser.add_argument('--learning_rate_multi', type=float, default=0.0001, help='Base learning rate to start from.')

        # multi-student
        parser.add_argument('--epochs_multi_student', type=int, default=30, help='Number of training epochs.')
        parser.add_argument('--learning_rate_multi_student', type=float, default=0.0001, help='Base learning rate to start from.')

        parser.add_argument('--weight_decay', type=float, default=0.0001, help='Weight decay factor.')

        return parser

    @staticmethod
    def model_parser() -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser("Simulation Data", add_help=False)
        parser.add_argument('--hidden', type=int, default=25)
        parser.add_argument('--simple', type=str2bool, default=False)
        parser.add_argument('--encoder_act', type=handle_none, default='relu')

        return parser

    @staticmethod
    def logging_parser() -> argparse.ArgumentParser:
        """Returns an `argparse.ArgumentParser` instance containing logging-related arguments."""
        parser = argparse.ArgumentParser("Logging", add_help=False)
        parser.add_argument('--checkpoint_root', type=str, default='./checkpoints/', help='Top-level directory of checkpoints.')
        parser.add_argument('--save_every', type=int, default=2000, help='Save model checkpoint every `save_every` epochs.')
        parser.add_argument('--enable_wandb', type=str2bool, default=True, help='Use Weights & Biases plugin.')
        parser.add_argument('--save_log', type=str2bool, default=True)
        parser.add_argument('--note', type=str, default=None)
        return parser


class ConfigSimulationDisc(object):
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
        parser.add_argument('--alpha_disc', type=float, default=1.0)
        parser.add_argument('--alpha_recon', type=float, default=100.0)
        parser.add_argument('--alpha_kd_clf', type=float, default=100.0)
        parser.add_argument('--alpha_kd_repr', type=float, default=100.0)

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
        parser.add_argument('--gpus', type=str, nargs='+', default='0', help='')
        parser.add_argument('--server', type=str, default='main',
                            choices=('main', 'workstation1', 'workstation2', 'workstation3'))
        parser.add_argument('--num_nodes', type=int, default=1, help='')
        parser.add_argument('--node_rank', type=int, default=0, help='')
        parser.add_argument('--dist_url', type=str, default='tcp://127.0.0.1:3500', help='')
        parser.add_argument('--dist_backend', type=str, default='nccl', help='')
        return parser

    @staticmethod
    def data_parser() -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser("Simulation Data", add_help=False)

        # representation
        parser.add_argument('--zs_dim', type=int, default=10)
        parser.add_argument('--z1_dim', type=int, default=10)
        parser.add_argument('--z2_dim', type=int, default=10)
        parser.add_argument('--rho', type=float, default=0.5)
        parser.add_argument('--sigma', type=float, default=1.0)

        # input
        parser.add_argument('--mu_0', type=float, default=0.0)
        parser.add_argument('--mu_1', type=float, default=1.0)
        parser.add_argument('--xs_dim', type=int, default=20)
        parser.add_argument('--x1_dim', type=int, default=20)
        parser.add_argument('--x2_dim', type=int, default=20)
        parser.add_argument('--slope', type=float, default=0.5)

        # samples
        parser.add_argument('--n_complete', type=int, default=1000)
        parser.add_argument('--n_incomplete', type=int, default=0)
        parser.add_argument('--n_test', type=int, default=1000)

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
        parser.add_argument('--train_level', type=parse_numbers, default="1,2,3",
                            help='1: Single from scratch,'
                                 '2: SMT & SMT-Student & Final'
                                 '3: Multi & Multi-Student')

        # single from scratch
        parser.add_argument('--epochs_single', type=int, default=30, help='Number of training epochs.')
        parser.add_argument('--learning_rate_single', type=float, default=0.0001,
                            help='Base learning rate to start from.')

        # smt
        parser.add_argument('--epochs_smt', type=int, default=100, help='Number of training epochs.')
        parser.add_argument('--learning_rate_smt', type=float, default=0.001, help='Base learning rate to start from.')

        # smt-student
        parser.add_argument('--epochs_smt_student', type=int, default=30, help='Number of training epochs.')
        parser.add_argument('--learning_rate_smt_student', type=float, default=0.0001, help='Base learning rate to start from.')

        # final
        parser.add_argument('--epochs_final', type=int, default=100, help='Number of training epochs.')
        parser.add_argument('--learning_rate_final', type=float, default=0.001, help='Base learning rate to start from.')

        # multi
        parser.add_argument('--epochs_multi', type=int, default=30, help='Number of training epochs.')
        parser.add_argument('--learning_rate_multi', type=float, default=0.0001, help='Base learning rate to start from.')

        # multi-student
        parser.add_argument('--epochs_multi_student', type=int, default=30, help='Number of training epochs.')
        parser.add_argument('--learning_rate_multi_student', type=float, default=0.0001, help='Base learning rate to start from.')

        parser.add_argument('--weight_decay', type=float, default=0.0001, help='Weight decay factor.')

        return parser

    @staticmethod
    def model_parser() -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser("Simulation Data", add_help=False)
        parser.add_argument('--hidden', type=int, default=25)
        parser.add_argument('--simple', type=str2bool, default=False)
        parser.add_argument('--encoder_act', type=handle_none, default='relu')

        return parser

    @staticmethod
    def logging_parser() -> argparse.ArgumentParser:
        """Returns an `argparse.ArgumentParser` instance containing logging-related arguments."""
        parser = argparse.ArgumentParser("Logging", add_help=False)
        parser.add_argument('--checkpoint_root', type=str, default='./checkpoints/', help='Top-level directory of checkpoints.')
        parser.add_argument('--save_every', type=int, default=2000, help='Save model checkpoint every `save_every` epochs.')
        parser.add_argument('--enable_wandb', type=str2bool, default=True, help='Use Weights & Biases plugin.')
        parser.add_argument('--save_log', type=str2bool, default=True)
        parser.add_argument('--note', type=str, default=None)
        return parser
