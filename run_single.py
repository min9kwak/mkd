# -*- coding: utf-8 -*-

import os
import sys
import time
import rich
import numpy as np
import wandb

import torch
import torch.nn as nn

from configs.single import SingleConfig
from tasks.single import Single

from dataset.brain import BrainProcessor, Brain
from dataset.transforms import make_mri_transforms, make_pet_transforms

from models.build import build_network_single

from utils.logging import get_rich_logger
from utils.gpu import set_gpu


def main():
    """Main function for single/distributed linear classification."""

    config = SingleConfig.parse_arguments()
    config.task = 'Single-' + config.data_type.upper()

    set_gpu(config)
    num_gpus_per_node = len(config.gpus)
    world_size = config.num_nodes * num_gpus_per_node
    distributed = world_size > 1
    setattr(config, 'num_gpus_per_node', num_gpus_per_node)
    setattr(config, 'world_size', world_size)
    setattr(config, 'distributed', distributed)

    rich.print(config.__dict__)
    config.save()

    np.random.seed(config.random_state)
    torch.manual_seed(config.random_state)

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

    if config.distributed:
        raise NotImplementedError
    else:
        rich.print(f"Single GPU training.")
        main_worker(0, config=config)  # single machine, single gpu


def main_worker(local_rank: int, config: object):
    """Single process."""

    torch.cuda.set_device(local_rank)
    if config.distributed:
        raise NotImplementedError

    config.batch_size = config.batch_size // config.world_size
    config.num_workers = config.num_workers // config.num_gpus_per_node

    # Logging
    logfile = os.path.join(config.checkpoint_dir, 'main.log')
    logger = get_rich_logger(logfile=logfile)
    if config.enable_wandb:
        wandb.init(
            name=f'{config.task} : {config.hash}',
            project='IMFi',
            config=config.__dict__,
            settings=wandb.Settings(code_dir=".")
        )

    # Transform
    if config.data_type == 'mri':
        train_transform, test_transform = make_mri_transforms(image_size=config.image_size)
    elif config.data_type == 'pet':
        train_transform, test_transform = make_pet_transforms(image_size=config.image_size)
    else:
        raise ValueError('data_type must be either mri or pet')

    # Dataset
    processor = BrainProcessor(root=config.root,
                               data_file=config.data_file,
                               pet_type=config.pet_type,
                               random_state=config.random_state)
    datasets_dict = processor.process(validation_size=config.validation_size,
                                      test_size=config.test_size,
                                      missing_rate=config.missing_rate)

    if config.data_type == 'mri':
        train_set = Brain(dataset=datasets_dict['mri_total_train'],
                          data_type='mri',
                          mri_transform=train_transform,
                          pet_transform=None)
        validation_set = Brain(dataset=datasets_dict['mri_pet_complete_validation'],
                               data_type='mri',
                               mri_transform=train_transform,
                               pet_transform=None)
        test_set = Brain(dataset=datasets_dict['mri_pet_complete_test'],
                         data_type='mri',
                         mri_transform=test_transform,
                         pet_transform=None)
    elif config.data_type == 'pet':
        train_set = Brain(dataset=datasets_dict['mri_pet_complete_train'],
                          data_type='pet',
                          mri_transform=None,
                          pet_transform=train_transform)
        validation_set = Brain(dataset=datasets_dict['mri_pet_complete_validation'],
                               data_type='pet',
                               mri_transform=None,
                               pet_transform=train_transform)
        test_set = Brain(dataset=datasets_dict['mri_pet_complete_test'],
                         data_type='pet',
                         mri_transform=None,
                         pet_transform=test_transform)
    else:
        raise ValueError('data_type must be either mri or pet')

    datasets = {'train': train_set, 'validation': validation_set, 'test': test_set}

    # Networks: mri_encoder, mri_projector, classifier
    networks = build_network_single(config=config)

    # Cross Entropy Loss Function
    class_weight = None
    if config.balance:
        class_weight = torch.tensor(processor.class_weight, dtype=torch.float).to(local_rank)
    loss_function_ce = nn.CrossEntropyLoss(weight=class_weight, reduction='mean')

    # Model (Task)
    model = Single(networks=networks, data_type=config.data_type)
    model.prepare(
        checkpoint_dir=config.checkpoint_dir,
        loss_function_ce=loss_function_ce,
        optimizer=config.optimizer,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        cosine_warmup=config.cosine_warmup,
        cosine_cycles=config.cosine_cycles,
        cosine_min_lr=config.cosine_min_lr,
        epochs=config.epochs,
        batch_size=config.batch_size,
        accumulate=config.accumulate,
        num_workers=config.num_workers,
        distributed=config.distributed,
        local_rank=local_rank,
        mixed_precision=config.mixed_precision,
        enable_wandb=config.enable_wandb
    )

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
