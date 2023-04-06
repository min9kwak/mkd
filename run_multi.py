# -*- coding: utf-8 -*-

import os
import sys
import time
import rich
import numpy as np
import wandb

import torch
import torch.nn as nn

from configs.slice.multi import SliceMultiConfig
from tasks.slice.multi import Multi

from datasets.brain import BrainProcessor, BrainMulti
from datasets.slice.transforms import make_mri_transforms, make_pet_transforms
from models.slice.build import build_networks

from utils.logging import get_rich_logger
from utils.gpu import set_gpu


def main():
    """Main function for single/distributed linear classification."""

    config = SliceMultiConfig.parse_arguments()
    config.task = 'Multi-' + config.data_type.upper() + '-' + config.pet_type.upper()

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
            project='incomplete-kd',
            config=config.__dict__,
            settings=wandb.Settings(code_dir=".")
        )

    # Transform
    train_transform_mri, test_transform_mri = make_mri_transforms(
        image_size=config.image_size, intensity=config.intensity, crop_size=config.crop_size,
        rotate=config.rotate, flip=config.flip, affine=config.affine, blur_std=config.blur_std,
        train_slices=config.train_slices, num_slices=config.num_slices, slice_range=config.slice_range,
        prob=config.prob
    )
    train_transform_pet, test_transform_pet = make_pet_transforms(
        image_size=config.image_size, intensity=config.intensity, crop_size=config.crop_size,
        rotate=config.rotate, flip=config.flip, affine=config.affine, blur_std=config.blur_std,
        train_slices=config.train_slices, num_slices=config.num_slices, slice_range=config.slice_range,
        prob=config.prob
    )

    # Dataset
    processor = BrainProcessor(root=config.root,
                               data_file=config.data_file,
                               pet_type=config.pet_type,
                               mci_only=config.mci_only,
                               random_state=config.random_state)
    datasets_dict = processor.process(validation_size=config.validation_size,
                                      test_size=config.test_size,
                                      missing_rate=config.missing_rate)

    train_set = BrainMulti(dataset=datasets_dict['mri_total_train'],
                           mri_transform=train_transform_mri,
                           pet_transform=train_transform_pet)
    validation_set = BrainMulti(dataset=datasets_dict['mri_pet_complete_validation'],
                                mri_transform=test_transform_mri,
                                pet_transform=test_transform_pet)
    test_set = BrainMulti(dataset=datasets_dict['mri_pet_complete_test'],
                          mri_transform=test_transform_mri,
                          pet_transform=test_transform_pet)

    datasets = {'train': train_set, 'validation': validation_set, 'test': test_set}

    # Networks
    networks = build_networks(config)
    networks = {'encoder_t': networks['encoder_pet'],
                'encoder_s': networks['encoder_mri'],
                'classifier': networks['classifier']}

    # Cross Entropy Loss Function
    class_weight = None
    if config.balance:
        class_weight = torch.tensor(processor.class_weight, dtype=torch.float).to(local_rank)
    loss_function_ce = nn.CrossEntropyLoss(weight=class_weight, reduction='mean')

    # Model (Task)
    model = Multi(networks=networks)
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
        num_workers=config.num_workers,
        distributed=config.distributed,
        local_rank=local_rank,
        add_type=config.add_type,
        mixed_precision=config.mixed_precision,
        enable_wandb=config.enable_wandb,
        config=config
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
