# -*- coding: utf-8 -*-

import os
import sys
import time
import rich
import numpy as np
import wandb

import torch
import torch.nn as nn

from configs.slice.mdt import MDTConfig
from tasks.slice.mdt import MDT

from datasets.brain import BrainProcessor, BrainMulti
from datasets.slice.transforms import make_mri_transforms, make_pet_transforms
from models.slice.build import build_networks_general_teacher

from utils.loss import SimCosineLoss, SimCMDLoss, SimL2Loss
from utils.loss import DiffCosineLoss, DiffFrobeniusLoss, DiffMSELoss
from utils.logging import get_rich_logger
from utils.gpu import set_gpu

import argparse


def main():
    """Main function for training MDT (Modality-Disentangling Teacher)."""

    # Parse configuration arguments
    config = MDTConfig.parse_arguments()
    config.task = 'MDT'

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
    """Single process."""

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

    # Create data augmentation transforms for MRI and PET
    train_transform_mri, test_transform_mri = make_mri_transforms(
        image_size_mri=config.image_size_mri, intensity_mri=config.intensity_mri, crop_size_mri=config.crop_size_mri,
        rotate_mri=config.rotate_mri, flip_mri=config.flip_mri, affine_mri=config.affine_mri,
        blur_std_mri=config.blur_std_mri, train_slices=config.train_slices, num_slices=config.num_slices,
        slice_range=config.slice_range, space=config.space, n_points=config.n_points, prob=config.prob)
    train_transform_pet, test_transform_pet = make_pet_transforms(
        image_size_pet=config.image_size_pet, intensity_pet=config.intensity_pet, crop_size_pet=config.crop_size_pet,
        rotate_pet=config.rotate_pet, flip_pet=config.flip_pet, affine_pet=config.affine_pet,
        blur_std_pet=config.blur_std_pet, train_slices=config.train_slices, num_slices=config.num_slices,
        slice_range=config.slice_range, space=config.space, n_points=config.n_points, prob=config.prob)

    # Process and load dataset with optional PET missing rate simulation
    if config.missing_rate == -1.0:
        setattr(config, 'missing_rate', None)

    # Initialize data processor for ADNI brain imaging dataset
    processor = BrainProcessor(root=config.root,
                               data_file=config.data_file,
                               mri_type=config.mri_type,
                               mci_only=config.mci_only,
                               use_unlabeled=config.use_unlabeled,
                               random_state=config.random_state)
    datasets_dict = processor.process(validation_size=config.validation_size,
                                      test_size=config.test_size,
                                      missing_rate=config.missing_rate)

    if config.missing_rate is None:
        setattr(config, 'missing_rate', -1.0)
    setattr(config, 'current_missing_rate', processor.current_missing_rate)

    # Create datasets for multi-modal training (complete MRI+PET pairs only)
    train_set = BrainMulti(dataset=datasets_dict['mri_pet_complete_train'],
                           mri_transform=train_transform_mri,
                           pet_transform=train_transform_pet)
    validation_set = BrainMulti(dataset=datasets_dict['mri_pet_complete_validation'],
                                mri_transform=test_transform_mri,
                                pet_transform=test_transform_pet)
    test_set = BrainMulti(dataset=datasets_dict['mri_pet_complete_test'],
                          mri_transform=test_transform_mri,
                          pet_transform=test_transform_pet)

    datasets = {'train': train_set, 'validation': validation_set, 'test': test_set}

    # Build neural network modules (extractors, projectors, encoders, decoders, classifier)
    networks = build_networks_general_teacher(config=config)

    # Initialize logger for training progress
    logfile = os.path.join(config.checkpoint_dir, 'main.log')
    logger = get_rich_logger(logfile=logfile)
    if config.enable_wandb:
        wandb.init(
            name=f'{config.task} : {config.hash}',
            project='incomplete-kd-teacher',
            config=config.__dict__,
            settings=wandb.Settings(code_dir=".")
        )
    if local_rank == 0:
        rich.print(config.__dict__)
        config.save()

    # Configure loss functions for multi-modal teacher training
    # Cross Entropy loss for classification with optional class balancing
    class_weight = None
    if config.balance:
        class_weight = torch.tensor(processor.class_weight_pet, dtype=torch.float).to(local_rank)
    loss_function_ce = nn.CrossEntropyLoss(weight=class_weight, reduction='sum', ignore_index=-1)

    # Similarity loss: encourages common representations across modalities
    if config.loss_sim == 'cmd':
        loss_function_sim = SimCMDLoss(n_moments=config.n_moments)
    elif config.loss_sim == 'cosine':
        loss_function_sim = SimCosineLoss()
    elif config.loss_sim == 'l2':
        loss_function_sim = SimL2Loss()
    elif config.loss_sim == 'mse':
        loss_function_sim = nn.MSELoss(reduction='mean')
    else:
        raise ValueError

    # Difference loss: separates modality-specific representations
    if config.loss_diff == 'cosine':
        loss_function_diff = DiffCosineLoss()
    elif config.loss_diff == 'fro':
        loss_function_diff = DiffFrobeniusLoss()
    elif config.loss_diff == 'mse':
        loss_function_diff = DiffMSELoss()
    else:
        raise ValueError

    # Reconstruction loss: ensures information preservation
    loss_function_recon = nn.MSELoss(reduction='mean')

    # Initialize MDT model with all loss functions
    model = MDT(networks=networks)
    model.prepare(config=config,
                  loss_function_ce=loss_function_ce,
                  loss_function_sim=loss_function_sim,
                  loss_function_diff=loss_function_diff,
                  loss_function_recon=loss_function_recon,
                  swap=config.swap,
                  local_rank=local_rank)

    # Start training and evaluation loop
    start = time.time()
    model.run(
        datasets=datasets,
        save_every=config.save_every,
        logger=logger
    )
    elapsed_sec = time.time() - start

    # Log total training time
    if logger is not None:
        elapsed_mins = elapsed_sec / 60
        logger.info(f'Total training time: {elapsed_mins:,.2f} minutes.')
        logger.handlers.clear()


if __name__ == '__main__':

    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
