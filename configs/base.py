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


class ConfigBase(object):
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
            save_path = os.path.join(self.checkpoint_dir, 'configs.json')
        else:
            save_path = os.path.join(self.checkpoint_dir, f'configs_{path}.json')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        attrs = copy.deepcopy(vars(self))
        attrs['task'] = self.task
        attrs['checkpoint_dir'] = self.checkpoint_dir

        with open(save_path, 'w') as f:
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
        raise NotImplementedError

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
        raise NotImplementedError

    @staticmethod
    def train_parser() -> argparse.ArgumentParser:
        raise NotImplementedError

    @staticmethod
    def model_parser() -> argparse.ArgumentParser:
        raise NotImplementedError

    @staticmethod
    def logging_parser() -> argparse.ArgumentParser:
        """Returns an `argparse.ArgumentParser` instance containing logging-related arguments."""
        parser = argparse.ArgumentParser("Logging", add_help=False)
        parser.add_argument('--checkpoint_root', type=str, default='./checkpoints/', help='Top-level directory of checkpoints.')
        parser.add_argument('--save_every', type=int, default=200, help='Save model checkpoint every `save_every` epochs.')
        parser.add_argument('--enable_wandb', type=str2bool, default=True, help='Use Weights & Biases plugin.')
        return parser
