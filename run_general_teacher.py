# -*- coding: utf-8 -*-

import os
import sys
import time
import rich
import numpy as np
import wandb

import torch
import torch.nn as nn

from configs.slice.general_teacher import SliceGeneralTeacher
from tasks.slice.general_teacher import GeneralTeacher

from datasets.brain import BrainProcessor, BrainMulti
from datasets.slice.transforms import make_mri_transforms, make_pet_transforms
from models.slice.build import build_networks_general_teacher

from utils.loss import DiffLoss, CMDLoss
from utils.logging import get_rich_logger
from utils.gpu import set_gpu


def main():
    """Main function for single/distributed linear classification."""

    config = SliceGeneralTeacher.parse_arguments()
    config.task = 'GeneralTeacher-' + config.pet_type.upper()

    if config.server == 'main':
        setattr(config, 'root', 'D:/data/ADNI')
    else:
        setattr(config, 'root', '/raidWorkspace/mingu/Data/ADNI')

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


def main_worker(local_rank: int, config: object):
    """Single process."""

    torch.cuda.set_device(local_rank)
    if config.distributed:
        raise NotImplementedError

    config.batch_size = config.batch_size // config.world_size
    config.num_workers = config.num_workers // config.num_gpus_per_node

    # Transform
    if config.train_slices == 'random':
        pass
    elif config.train_slices == 'fixed':
        setattr(config, 'num_slices', 3)
    elif config.train_slices in ['sagittal', 'coronal', 'axial']:
        setattr(config, 'num_slices', 1)
    else:
        raise ValueError

    train_transform_mri, test_transform_mri = make_mri_transforms(
        image_size_mri=config.image_size_mri, intensity_mri=config.intensity_mri, crop_size_mri=config.crop_size_mri,
        rotate_mri=config.rotate_mri, flip_mri=config.flip_mri, affine_mri=config.affine_mri,
        blur_std_mri=config.blur_std_mri, train_slices=config.train_slices, num_slices=config.num_slices,
        slice_range=config.slice_range, prob=config.prob)
    train_transform_pet, test_transform_pet = make_pet_transforms(
        image_size_pet=config.image_size_pet, intensity_pet=config.intensity_pet, crop_size_pet=config.crop_size_pet,
        rotate_pet=config.rotate_pet, flip_pet=config.flip_pet, affine_pet=config.affine_pet,
        blur_std_pet=config.blur_std_pet, train_slices=config.train_slices, num_slices=config.num_slices,
        slice_range=config.slice_range, prob=config.prob)

    # Dataset
    if config.missing_rate == -1.0:
        setattr(config, 'missing_rate', None)

    processor = BrainProcessor(root=config.root,
                               data_file=config.data_file,
                               mri_type=config.mri_type,
                               pet_type=config.pet_type,
                               mci_only=config.mci_only,
                               random_state=config.random_state)
    datasets_dict = processor.process(validation_size=config.validation_size,
                                      test_size=config.test_size,
                                      missing_rate=config.missing_rate)
    setattr(config, 'current_missing_rate', processor.current_missing_rate)

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

    # Networks
    networks = build_networks_general_teacher(config=config)

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
    if local_rank == 0:
        rich.print(config.__dict__)
        config.save()

    # Loss Function
    # Cross Entropy
    class_weight = None
    if config.balance:
        class_weight = torch.tensor(processor.class_weight_pet, dtype=torch.float).to(local_rank)
    loss_function_ce = nn.CrossEntropyLoss(weight=class_weight, reduction='mean')

    # Similarity, Difference, and Reconstruction
    loss_function_sim = CMDLoss(n_moments=config.n_moments)
    if config.loss_diff == 'diff':
        loss_function_diff = DiffLoss()
    elif config.loss_diff == 'mse':
        loss_function_diff = nn.MSELoss(reduction='mean')
    else:
        raise ValueError
    loss_function_recon = nn.MSELoss(reduction='mean')

    # Model (Task)
    model = GeneralTeacher(networks=networks)
    model.prepare(config=config,
                  loss_function_ce=loss_function_ce,
                  loss_function_sim=loss_function_sim,
                  loss_function_diff=loss_function_diff,
                  loss_function_recon=loss_function_recon,
                  swap=config.swap,
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
