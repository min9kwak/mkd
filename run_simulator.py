# -*- coding: utf-8 -*-

import os
import sys
import json
import time
import rich
import numpy as np
import wandb

import torch
import torch.nn as nn

from configs.simulation import ConfigSimulation
from tasks.simulation.simulator import Simulator
from utils.simulation import DataGenerator, MultiModalDataset

from utils.loss import SimCosineLoss
from utils.loss import DiffCosineLoss
from utils.logging import get_rich_logger
from utils.gpu import set_gpu

import argparse


def main():
    """Main function for single/distributed linear classification."""

    config = ConfigSimulation.parse_arguments()

    denom = config.z1_dim + config.z2_dim + config.zs_dim
    alpha = config.z1_dim / denom
    beta = config.z2_dim / denom
    gamma = config.zs_dim / denom

    setattr(config, 'alpha', alpha)
    setattr(config, 'beta', beta)
    setattr(config, 'gamma', gamma)

    missing_rate = config.n_incomplete / (config.n_complete + config.n_incomplete)
    setattr(config, 'missing_rate', missing_rate)

    config.task = 'simulation'

    set_gpu(config)
    num_gpus_per_node = len(config.gpus)
    world_size = config.num_nodes * num_gpus_per_node
    distributed = world_size > 1
    setattr(config, 'num_gpus_per_node', num_gpus_per_node)
    setattr(config, 'world_size', world_size)
    setattr(config, 'distributed', distributed)

    np.random.seed(config.random_state)
    torch.manual_seed(config.random_state)

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

    if config.distributed:
        raise NotImplementedError
    else:
        rich.print(f"Single GPU training.")
        main_worker(0, config=config)  # single machine, single gpu


def main_worker(local_rank: int, config: argparse.Namespace):
    """Single process."""

    torch.cuda.set_device(local_rank)
    if config.distributed:
        raise NotImplementedError

    config.batch_size = config.batch_size // config.world_size
    config.num_workers = config.num_workers // config.num_gpus_per_node

    # Dataset
    generator = DataGenerator(zs_dim=config.zs_dim, z1_dim=config.z1_dim, z2_dim=config.z2_dim,
                              rho=config.rho, sigma=config.sigma, random_state=config.random_state)
    dataset = generator.generate_data(mu_0=config.mu_0, mu_1=config.mu_1,
                                      xs_dim=config.xs_dim, x1_dim=config.x1_dim, x2_dim=config.x2_dim,
                                      slope=config.slope,
                                      n_complete=config.n_complete, n_incomplete=config.n_incomplete,
                                      n_validation=config.n_validation, n_test=config.n_test)

    train_complete_set = MultiModalDataset(x1=dataset['x1_train_complete'],
                                           x2=dataset['x2_train_complete'],
                                           y=dataset['y_train_complete'])
    validation_set = MultiModalDataset(x1=dataset['x1_validation'],
                                       x2=dataset['x2_validation'],
                                       y=dataset['y_validation'])
    test_set = MultiModalDataset(x1=dataset['x1_test'],
                                 x2=dataset['x2_test'],
                                 y=dataset['y_test'])

    if config.n_incomplete > 0:
        train_incomplete_set = MultiModalDataset(x1=dataset['x1_train_incomplete'],
                                                 x2=None,
                                                 y=dataset['y_train_incomplete'])
        setattr(config, 'use_incomplete', True)
    else:
        train_incomplete_set = None
        setattr(config, 'use_incomplete', False)

    # dataset for scratch
    if config.use_incomplete:
        x1_total = np.concatenate([dataset['x1_train_complete'], dataset['x1_train_incomplete']], axis=0)
        y_total = np.concatenate([dataset['y_train_complete'], dataset['y_train_incomplete']], axis=0)
    else:
        x1_total = dataset['x1_train_complete']
        y_total = dataset['y_train_complete']

    train_total_set = MultiModalDataset(x1=x1_total, x2=None, y=y_total)
    datasets = {'train_complete': train_complete_set,
                'train_incomplete': train_incomplete_set,
                'train_total': train_total_set,
                'validation': validation_set,
                'test': test_set}

    # Logging
    logfile = os.path.join(config.checkpoint_dir, 'main.log')
    logger = get_rich_logger(logfile=logfile)
    if config.enable_wandb:
        wandb.init(
            name=f'{config.task} : {config.hash}',
            project='incomplete-kd-simulation',
            config=config.__dict__,
            settings=wandb.Settings(code_dir=".")
        )
    if local_rank == 0:
        rich.print(config.__dict__)
        config.save()

    # create train_params
    train_params = {
        'single': {'epochs': config.epochs_single,
                   'learning_rate': config.learning_rate_single},
        'smt': {'epochs': config.epochs_smt,
                'learning_rate': config.learning_rate_smt},
        'smt_student': {'epochs': config.epochs_smt_student,
                        'learning_rate': config.learning_rate_smt_student},
        'final': {'epochs': config.epochs_final,
                  'learning_rate': config.learning_rate_final},
        'final_single': {'epochs': config.epochs_final,
                         'learning_rate': config.learning_rate_final},
        'multi': {'epochs': config.epochs_multi,
                  'learning_rate': config.learning_rate_multi},
        'multi_student': {'epochs': config.epochs_multi_student,
                          'learning_rate': config.learning_rate_multi_student},
    }
    for mode in train_params.keys():
        train_params[mode]['weight_decay'] = config.weight_decay
    setattr(config, 'train_params', train_params)

    # Loss Function
    # Cross Entropy
    loss_function_ce = nn.CrossEntropyLoss(reduction='mean')
    loss_function_sim = SimCosineLoss()
    loss_function_diff = DiffCosineLoss()
    loss_function_recon = nn.MSELoss(reduction='mean')

    # Model (Task)
    model = Simulator()
    model.prepare(config=config,
                  loss_function_ce=loss_function_ce,
                  loss_function_sim=loss_function_sim,
                  loss_function_diff=loss_function_diff,
                  loss_function_recon=loss_function_recon,
                  save_log=config.save_log,
                  local_rank=local_rank)

    # Train & evaluate
    start = time.time()
    model.run(
        datasets=datasets,
        save_every=config.save_every,
        logger=logger
    )
    elapsed_sec = time.time() - start

    if logger is not None:
        elapsed_mins = elapsed_sec / 60
        logger.info(f'Total training time: {elapsed_mins:,.2f} minutes.')
        logger.handlers.clear()


if __name__ == '__main__':

    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
