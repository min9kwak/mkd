# -*- coding: utf-8 -*-

import os
import sys
import time
import rich
import numpy as np
import wandb

import torch
import torch.nn as nn

from configs.swap import SwapConfig
from tasks.swap import Swap

from dataset.brain import BrainProcessor, Brain
from dataset.transforms import make_mri_transforms, make_pet_transforms

from models.build import build_network

from utils.logging import get_rich_logger
from utils.gpu import set_gpu

# TODO: DDP - wrap, spawn, and make DDPBatchSampler
def main():
    """Main function for single/distributed linear classification."""

    config = SwapConfig.parse_arguments()

    if config.swap:
        config.task = 'Swap'
    else:
        config.task = 'Non-Swap'

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
    mri_train_transform, mri_test_transform = make_mri_transforms(image_size=config.mri_image_size)
    pet_train_transform, pet_test_transform = make_pet_transforms(image_size=config.pet_image_size)

    # Dataset
    processor = BrainProcessor(root=config.root,
                               data_file=config.data_file,
                               pet_type=config.pet_type,
                               random_state=config.random_state)
    datasets_dict = processor.process(validation_size=config.validation_size,
                                      test_size=config.test_size,
                                      missing_rate=config.missing_rate)
    datasets = {
        'mri_pet_complete_train': Brain(dataset=datasets_dict['mri_pet_complete_train'],
                                        data_type='multi',
                                        mri_transform=mri_train_transform,
                                        pet_transform=pet_train_transform),
        'mri_incomplete_train': Brain(dataset=datasets_dict['mri_incomplete_train'],
                                      data_type='mri',
                                      mri_transform=mri_train_transform,
                                      pet_transform=None),
        'mri_pet_complete_validation': Brain(dataset=datasets_dict['mri_pet_complete_validation'],
                                             data_type='multi',
                                             mri_transform=mri_test_transform,
                                             pet_transform=pet_test_transform),
        'mri_pet_complete_test': Brain(dataset=datasets_dict['mri_pet_complete_test'],
                                       data_type='multi',
                                       mri_transform=mri_test_transform,
                                       pet_transform=pet_test_transform)
    }

    # Networks: mri_encoder, pet_encoder, mri_projector, pet_projector, common_projector, predictor, classifier
    networks = build_network(config=config)

    # Cross Entropy Loss Function
    class_weight = None
    if config.balance:
        class_weight = torch.tensor(processor.class_weight, dtype=torch.float).to(local_rank)
    loss_function_ce = nn.CrossEntropyLoss(weight=class_weight, reduction='sum')

    # Model (Task)
    model = Swap(networks=networks)
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
        incomplete_ratio=config.incomplete_ratio,
        add_type=config.add_type,
        alpha=config.alpha,
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
