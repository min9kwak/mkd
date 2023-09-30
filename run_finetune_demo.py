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

from configs.slice.finetune_demo import SliceFinetuneDemo
from tasks.slice.finetune_demo import DemoClassification

from datasets.brain import BrainProcessor, BrainMulti, BrainMRI
from datasets.slice.transforms import make_mri_transforms, make_pet_transforms
from models.slice.build import build_networks_general_teacher
from models.slice.demo import DemoEncoder, LinearDemoClassifier

from utils.logging import get_rich_logger
from utils.gpu import set_gpu

import argparse
from copy import deepcopy
from easydict import EasyDict as edict


def main():
    """Main function for single/distributed linear classification."""

    config = SliceFinetuneDemo.parse_arguments()

    pretrained_config = os.path.join(config.pretrained_dir, "configs.json")
    with open(pretrained_config, 'rb') as fb:
        pretrained_config = json.load(fb)
        pretrained_config = edict(pretrained_config)

    pretrained_file = os.path.join(config.pretrained_dir, "ckpt.last.pth.tar")
    setattr(pretrained_config, 'pretrained_file', pretrained_file)

    if config.task_type == 'multi':
        if hasattr(pretrained_config, 'use_specific_final'):
            setattr(config, 'use_specific_final', pretrained_config.use_specific_final)
        else:
            setattr(pretrained_config, 'use_specific_final', False)
            setattr(config, 'use_specific_final', False)

    # inherit
    setattr(config, 'random_state', pretrained_config.random_state)

    # define task
    config.task = f'{config.task_type}_demo'

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

    np.random.seed(pretrained_config.random_state)
    torch.manual_seed(pretrained_config.random_state)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.allow_tf32 = True

    if config.distributed:
        raise NotImplementedError
    else:
        rich.print(f"Single GPU training.")
        main_worker(0, config=config, pretrained_config=pretrained_config)  # single machine, single gpu


def main_worker(local_rank: int, config: argparse.Namespace, pretrained_config: edict):
    """Single process."""

    torch.cuda.set_device(local_rank)
    if config.distributed:
        raise NotImplementedError

    config.num_workers = config.num_workers // config.num_gpus_per_node

    # Transform
    if pretrained_config.train_slices == 'random':
        pass
    elif pretrained_config.train_slices == 'fixed':
        num_slices = 3 * (2 * pretrained_config.n_points + 1)
        setattr(pretrained_config, 'num_slices', num_slices)
    elif pretrained_config.train_slices in ['sagittal', 'coronal', 'axial']:
        setattr(pretrained_config, 'num_slices', 1)
    else:
        raise ValueError

    train_transform_mri, test_transform_mri = make_mri_transforms(
        image_size_mri=pretrained_config.image_size_mri, intensity_mri=pretrained_config.intensity_mri,
        crop_size_mri=pretrained_config.crop_size_mri, rotate_mri=pretrained_config.rotate_mri,
        flip_mri=pretrained_config.flip_mri, affine_mri=pretrained_config.affine_mri,
        blur_std_mri=pretrained_config.blur_std_mri, train_slices=pretrained_config.train_slices,
        num_slices=pretrained_config.num_slices, slice_range=pretrained_config.slice_range,
        space=pretrained_config.space, n_points=pretrained_config.n_points, prob=pretrained_config.prob)
    train_transform_pet, test_transform_pet = make_pet_transforms(
        image_size_pet=pretrained_config.image_size_pet, intensity_pet=pretrained_config.intensity_pet,
        crop_size_pet=pretrained_config.crop_size_pet, rotate_pet=pretrained_config.rotate_pet,
        flip_pet=pretrained_config.flip_pet, affine_pet=pretrained_config.affine_pet,
        blur_std_pet=pretrained_config.blur_std_pet, train_slices=pretrained_config.train_slices,
        num_slices=pretrained_config.num_slices, slice_range=pretrained_config.slice_range,
        space=pretrained_config.space, n_points=pretrained_config.n_points, prob=pretrained_config.prob)

    # Dataset
    if pretrained_config.missing_rate == -1.0:
        setattr(pretrained_config, 'missing_rate', None)

    processor = BrainProcessor(root=pretrained_config.root,
                               data_file=pretrained_config.data_file,
                               mri_type=pretrained_config.mri_type,
                               pet_type=pretrained_config.pet_type,
                               mci_only=pretrained_config.mci_only,
                               use_unlabeled=False,
                               use_cdr=config.use_cdr,
                               scale_demo=config.scale_demo,
                               random_state=pretrained_config.random_state)
    datasets_dict = processor.process(validation_size=pretrained_config.validation_size,
                                      test_size=pretrained_config.test_size,
                                      missing_rate=pretrained_config.missing_rate)

    if pretrained_config.missing_rate is None:
        setattr(pretrained_config, 'missing_rate', -1.0)
    setattr(pretrained_config, 'current_missing_rate', processor.current_missing_rate)

    if config.task_type == 'multi':
        train_set = BrainMulti(dataset=datasets_dict['mri_pet_complete_train'],
                               mri_transform=train_transform_mri,
                               pet_transform=train_transform_pet)
        validation_set = BrainMulti(dataset=datasets_dict['mri_pet_complete_validation'],
                                    mri_transform=test_transform_mri,
                                    pet_transform=test_transform_pet)
        test_set = BrainMulti(dataset=datasets_dict['mri_pet_complete_test'],
                              mri_transform=test_transform_mri,
                              pet_transform=test_transform_pet)
    elif config.task_type == 'single':
        train_set = BrainMRI(dataset=datasets_dict['mri_total_train'], mri_transform=train_transform_mri)
        validation_set = BrainMRI(dataset=datasets_dict['mri_pet_complete_validation'], mri_transform=test_transform_mri)
        test_set = BrainMRI(dataset=datasets_dict['mri_pet_complete_test'], mri_transform=test_transform_mri)
    else:
        raise ValueError

    datasets = {'train': train_set, 'validation': validation_set, 'test': test_set}

    # Networks
    networks = build_networks_general_teacher(config=pretrained_config)

    encoder_demo = DemoEncoder(in_channels=len(processor.demo_columns), hidden=config.hidden_demo)
    classifier = LinearDemoClassifier(image_dims=pretrained_config.hidden // 2, demo_dims=encoder_demo.out_features,
                                      num_classes=2)

    # load from teacher and student
    if config.task_type == 'multi':
        teacher_network_names = ['extractor_pet', 'projector_pet', 'extractor_mri', 'projector_mri',
                                 'encoder_general', 'encoder_pet', 'encoder_mri']
        student_network_names = ['extractor_mri', 'projector_mri']

        for name in teacher_network_names:
            networks[name].load_weights_from_checkpoint(path=pretrained_config.pretrained_file, key=name)
        for name in student_network_names:
            network = deepcopy(networks[name])
            network.load_weights_from_checkpoint(path=pretrained_config.pretrained_file, key=name + '_s')
            networks[name + '_s'] = network
            del network

    elif config.task_type == 'single':
        student_network_names = ['extractor_mri_s', 'projector_mri_s', 'encoder_general_s']
        for name in student_network_names:
            network = deepcopy(networks[name.replace('_s', '')])
            network.load_weights_from_checkpoint(path=pretrained_config.pretrained_file, key=name)
            networks[name] = network
            del network

    networks['encoder_demo'] = deepcopy(encoder_demo)
    networks['classifier_demo'] = deepcopy(classifier)
    del encoder_demo, classifier

    # remove useless networks
    if config.task_type == 'multi':
        target_network_names = ['extractor_mri', 'projector_mri', 'extractor_pet', 'projector_pet',
                                'encoder_general', 'encoder_mri', 'encoder_pet',
                                'encoder_demo', 'classifier_demo']
    elif config.task_type == 'single':
        target_network_names = ['extractor_mri_s', 'projector_mri_s', 'encoder_general_s',
                                'encoder_demo', 'classifier_demo']
    else:
        raise ValueError

    network_names = list(networks.keys())
    for name in network_names:
        if name not in target_network_names:
            del networks[name]

    # Logging
    logfile = os.path.join(config.checkpoint_dir, 'main.log')
    logger = get_rich_logger(logfile=logfile)
    if config.enable_wandb:
        wandb.init(
            name=f'{config.task} : {config.hash}',
            project='incomplete-demo',
            config=config.__dict__,
            settings=wandb.Settings(code_dir=".")
        )
    if local_rank == 0:
        rich.print(config.__dict__)
        config.save()

        # save pretrained config
        save_path = os.path.join(config.checkpoint_dir, 'configs_pretrained.json')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump(pretrained_config, f, indent=2)


    # Loss Function
    # Cross Entropy
    class_weight = None
    if pretrained_config.balance:
        class_weight = torch.tensor(processor.class_weight_pet, dtype=torch.float).to(local_rank)
    loss_function_ce = nn.CrossEntropyLoss(weight=class_weight, reduction='mean', ignore_index=-1)

    # Model (Task)
    model = DemoClassification(networks=networks, task_type=config.task_type)
    model.prepare(config=config,
                  pretrained_config=pretrained_config,
                  loss_function=loss_function_ce,
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
