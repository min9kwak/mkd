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

from configs.slice.smt_student import SMTStudentConfig
from tasks.slice.smt_student import SMTStudent

from datasets.brain import BrainProcessor, BrainMulti, BrainMRI
from datasets.slice.transforms import make_mri_transforms, make_pet_transforms
from models.slice.build import build_networks_general_teacher

from utils.loss import SimCosineLoss, SimCMDLoss, SimL2Loss
from utils.loss import DiffCosineLoss, DiffFrobeniusLoss, DiffMSELoss
from utils.logging import get_rich_logger
from utils.gpu import set_gpu

import argparse
from copy import deepcopy


def main():
    """Main function for training SMT-Student via knowledge distillation."""

    # Parse configuration arguments
    config = SMTStudentConfig.parse_arguments()

    # Load teacher model checkpoint path
    teacher_file = os.path.join(config.teacher_dir, f"ckpt.{config.teacher_position}.pth.tar")
    setattr(config, 'teacher_file', teacher_file)

    # Load teacher's configuration to inherit hyperparameters
    teacher_config = os.path.join(config.teacher_dir, "configs.json")
    with open(teacher_config, 'rb') as fb:
        teacher_config = json.load(fb)

    # Inherit pretrained teacher configs and mark conflicting keys with '_t' suffix
    for key in teacher_config.keys():
        if key not in config.__dict__.keys():
            # property cannot be done
            try:
                setattr(config, key, teacher_config[key])
            except:
                pass
        else:
            try:
                setattr(config, f'{key}_t', teacher_config[key])
            except:
                pass
    setattr(config, 'hash_t', teacher_config['hash'])

    config.task = 'SMT-Student'

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


def main_worker(local_rank: int, config: argparse.Namespace):
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
        num_slices = 3 * (2 * config.n_points + 1)
        setattr(config, 'num_slices', num_slices)
    elif config.train_slices in ['sagittal', 'coronal', 'axial']:
        setattr(config, 'num_slices', 1)
    else:
        raise ValueError

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

    # Dataset
    if config.missing_rate == -1.0:
        setattr(config, 'missing_rate', None)

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

    # Create datasets: complete pairs for teacher, MRI-only for student training
    train_set = BrainMulti(dataset=datasets_dict['mri_pet_complete_train'],
                           mri_transform=train_transform_mri,
                           pet_transform=train_transform_pet)
    train_mri_set = BrainMRI(dataset=datasets_dict['mri_incomplete_train'],
                             mri_transform=train_transform_mri)
    validation_set = BrainMulti(dataset=datasets_dict['mri_pet_complete_validation'],
                                mri_transform=test_transform_mri,
                                pet_transform=test_transform_pet)
    test_set = BrainMulti(dataset=datasets_dict['mri_pet_complete_test'],
                          mri_transform=test_transform_mri,
                          pet_transform=test_transform_pet)

    datasets = {'train': train_set, 'train_mri': train_mri_set, 'validation': validation_set, 'test': test_set}

    # Define which networks belong to teacher and student
    teacher_network_names = ['extractor_mri', 'extractor_pet', 'projector_mri', 'projector_pet',
                             'encoder_general', 'classifier']
    student_network_names = ['extractor_mri', 'projector_mri',
                             'encoder_general', 'classifier']
    # Add modality-specific encoders if enabled
    teacher_network_names += ['encoder_mri']
    student_network_names += ['encoder_mri']

    # Build networks and filter by teacher/student requirements
    networks = build_networks_general_teacher(config=config)
    networks = {k: v for k, v in networks.items() if k in teacher_network_names and v is not None}
    networks_student = {f'{k}_s': deepcopy(v) for k, v in networks.items()
                        if k in student_network_names and v is not None}

    # Load pre-trained teacher weights from checkpoint
    for name, network in networks.items():
        network.load_weights_from_checkpoint(path=config.teacher_file, key=name)
    # Initialize student with teacher weights (optional)
    for name_s, network_s in networks_student.items():
        if config.use_teacher:
            if name_s == 'classifier':
                if config.inherit_classifier:
                    network_s.load_weights_from_checkpoint(path=config.teacher_file, key=name_s.replace('_s', ''))
                else:
                    pass
            else:
                network_s.load_weights_from_checkpoint(path=config.teacher_file, key=name_s.replace('_s', ''))
        networks[name_s] = network_s
    del networks_student

    # Logging
    logfile = os.path.join(config.checkpoint_dir, 'main.log')
    logger = get_rich_logger(logfile=logfile)
    if config.enable_wandb:
        wandb.init(
            name=f'{config.task} : {config.hash}',
            project='incomplete-kd-distillation',
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
    loss_function_ce = nn.CrossEntropyLoss(weight=class_weight, reduction='sum', ignore_index=-1)

    # Model (Task)
    model = SMTStudent(networks=networks)
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
