# -*- coding: utf-8 -*-
import os
import sys
import time
import rich
import numpy as np
import wandb

import argparse
from copy import deepcopy

import torch
import torch.nn as nn

from configs.slice.single import SliceSingleConfig
from tasks.slice.single import Single

from datasets.brain import BrainProcessor, BrainMRI, BrainPET
from datasets.slice.transforms import make_mri_transforms
from models.slice.build import build_networks_general_teacher

from utils.logging import get_rich_logger
from utils.gpu import set_gpu


def main():
    """Main function for training single-modality baseline (MRI-only or PET-only)."""

    # Parse configuration arguments
    config = SliceSingleConfig.parse_arguments()
    config.task = 'Single-' + config.data_type.upper()

    # Set data root path based on server configuration
    if config.server == 'main':
        setattr(config, 'root', 'D:/data/ADNI')
    else:
        setattr(config, 'root', '/raidWorkspace/mingu/Data/ADNI')

    # Configure GPU settings and distributed training
    set_gpu(config)
    num_gpus_per_node = len(config.gpus)
    world_size = config.num_nodes * num_gpus_per_node
    distributed = world_size > 1
    setattr(config, 'num_gpus_per_node', num_gpus_per_node)
    setattr(config, 'world_size', world_size)
    setattr(config, 'distributed', distributed)

    # Set random seeds for reproducibility
    np.random.seed(config.random_state)
    torch.manual_seed(config.random_state)

    # Configure CUDNN for faster training
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

    if config.distributed:
        raise NotImplementedError
    else:
        rich.print(f"Single GPU training.")
        main_worker(0, config=config)  # single machine, single gpu


def main_worker(local_rank: int, config: argparse.Namespace):
    """Single process worker for training single-modality model."""

    # Set device for current process
    torch.cuda.set_device(local_rank)
    if config.distributed:
        raise NotImplementedError

    # Adjust batch size and workers for distributed setting
    config.batch_size = config.batch_size // config.world_size
    config.num_workers = config.num_workers // config.num_gpus_per_node

    # Determine number of slices based on training strategy
    if config.train_slices == 'random':
        pass
    elif config.train_slices == 'fixed':
        num_slices = 3 * (2 * config.n_points + 1)
        setattr(config, 'num_slices', num_slices)
    elif config.train_slices in ['sagittal', 'coronal', 'axial']:
        setattr(config, 'num_slices', 1)
    else:
        raise ValueError

    # Create data augmentation transforms for selected modality (MRI or PET)
    assert config.data_type in ['mri', 'pet']
    train_transform, test_transform = make_mri_transforms(
        image_size_mri=config.image_size_mri, intensity_mri=config.intensity_mri, crop_size_mri=config.crop_size_mri,
        rotate_mri=config.rotate_mri, flip_mri=config.flip_mri, affine_mri=config.affine_mri,
        blur_std_mri=config.blur_std_mri, train_slices=config.train_slices, num_slices=config.num_slices,
        slice_range=config.slice_range, space=config.space, n_points=config.n_points, prob=config.prob)

    # Dataset
    if config.missing_rate == -1.0:
        setattr(config, 'missing_rate', None)

    assert not config.use_unlabeled, "only supervised learning"

    processor = BrainProcessor(root=config.root,
                               data_file=config.data_file,
                               mri_type=config.mri_type,
                               mci_only=config.mci_only,
                               random_state=config.random_state)
    datasets_dict = processor.process(validation_size=config.validation_size,
                                      test_size=config.test_size,
                                      missing_rate=config.missing_rate)

    if config.missing_rate is None:
        setattr(config, 'missing_rate', -1.0)
    setattr(config, 'current_missing_rate', processor.current_missing_rate)

    # TODO: BrainMRI and BrainPET: mri_pet_complete_train
    if config.data_type == 'mri':
        train_set = BrainMRI(dataset=datasets_dict['mri_total_train'], mri_transform=train_transform)
        validation_set = BrainMRI(dataset=datasets_dict['mri_pet_complete_validation'], mri_transform=test_transform)
        test_set = BrainMRI(dataset=datasets_dict['mri_pet_complete_test'], mri_transform=test_transform)
    elif config.data_type == 'pet':
        train_set = BrainPET(dataset=datasets_dict['mri_pet_complete_train'], pet_transform=train_transform)
        validation_set = BrainPET(dataset=datasets_dict['mri_pet_complete_validation'], pet_transform=test_transform)
        test_set = BrainPET(dataset=datasets_dict['mri_pet_complete_test'], pet_transform=test_transform)
    else:
        raise ValueError

    datasets = {'train': train_set, 'validation': validation_set, 'test': test_set}

    # Networks
    networks_ = build_networks_general_teacher(config=config)
    networks = {'extractor': deepcopy(networks_['extractor_mri']),
                'projector': deepcopy(networks_['projector_mri']),
                'encoder': deepcopy(networks_['encoder_mri']),
                'classifier': deepcopy(networks_['classifier'])}
    del networks_

    # Cross Entropy Loss Function
    class_weight = None
    if config.balance:
        if config.data_type == 'mri':
            class_weight = torch.tensor(processor.class_weight_mri, dtype=torch.float).to(local_rank)
        elif config.data_type == 'pet':
            class_weight = torch.tensor(processor.class_weight_pet, dtype=torch.float).to(local_rank)
        else:
            raise ValueError
    loss_function_ce = nn.CrossEntropyLoss(weight=class_weight, reduction='mean')

    # Logging
    logfile = os.path.join(config.checkpoint_dir, 'main.log')
    logger = get_rich_logger(logfile=logfile)
    if config.enable_wandb:
        wandb.init(
            name=f'{config.task} : {config.hash}',
            project=f'incomplete-kd-{config.data_type}',
            config=config.__dict__,
            settings=wandb.Settings(code_dir=".")
        )
    if local_rank == 0:
        rich.print(config.__dict__)
        config.save()

    # Model (Task)
    model = Single(networks=networks)
    model.prepare(config=config,
                  loss_function_ce=loss_function_ce,
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
