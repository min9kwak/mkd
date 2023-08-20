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

from configs.slice.external_test import SliceExternalTestConfig
from tasks.slice.external_test import ExternalTest

from datasets.aibl_oasis import AOProcessor, AODataset
from datasets.slice.transforms import make_mri_transforms, make_pet_transforms
from models.slice.build import build_networks_general_teacher
from models.slice.demo import DemoEncoder, LinearDemoClassifier

from utils.metrics import classification_result_by_source, flatten_results
from utils.logging import get_rich_logger
from utils.gpu import set_gpu

import argparse
from copy import deepcopy
from easydict import EasyDict as edict


def main():

    config = SliceExternalTestConfig.parse_arguments()

    # main
    if '_demo' in config.pretrained_task:
        demo_config = os.path.join(config.pretrained_dir, "configs.json")
        with open(demo_config, 'rb') as fb:
            demo_config = json.load(fb)
            demo_config = edict(demo_config)

        pretrained_config = os.path.join(config.pretrained_dir, "configs_pretrained.json")
        with open(pretrained_config, 'rb') as fb:
            pretrained_config = json.load(fb)
            pretrained_config = edict(pretrained_config)
    else:
        demo_config = None
        pretrained_config = os.path.join(config.pretrained_dir, "configs.json")
        with open(pretrained_config, 'rb') as fb:
            pretrained_config = json.load(fb)
            pretrained_config = edict(pretrained_config)

    pretrained_file = os.path.join(config.pretrained_dir, "ckpt.last.pth.tar")
    setattr(pretrained_config, 'pretrained_file', pretrained_file)

    if '+' in config.external_data_type:
        assert 'multi' in pretrained_config.task.lower()

    if hasattr(pretrained_config, 'use_specific_final'):
        setattr(config, 'use_specific_final', pretrained_config.use_specific_final)
    else:
        setattr(pretrained_config, 'use_specific_final', False)
        setattr(config, 'use_specific_final', False)

    # inherit
    setattr(config, 'random_state', pretrained_config.random_state)

    # define task
    config.task = f'ExternalTest'

    if config.server == 'main':
        setattr(config, 'root', 'D:/data')
    else:
        setattr(config, 'root', '/raidWorkspace/mingu/Data')

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

        if config.train_mode == 'train':
            y_true_final, y_pred_final, sources_final = [], [], []
            for n_cv in range(config.n_splits):
                # single machine, single gpu
                setattr(config, 'n_cv', n_cv)
                y_true, y_pred, sources = main_worker(0, config, pretrained_config, demo_config)
                y_true_final.append(y_true)
                y_pred_final.append(y_pred)
                sources_final.append(sources)

            y_true_final = torch.cat(y_true_final, dim=0)
            y_pred_final = torch.cat(y_pred_final, dim=0).to(torch.float32)
            sources_final = torch.cat(sources_final)

            clf_result = classification_result_by_source(y_true=y_true_final.cpu().numpy(),
                                                         y_pred=y_pred_final.softmax(1).detach().cpu().numpy(),
                                                         adjusted=False,
                                                         source=sources_final.detach().cpu().numpy())
            clf_result_adj = classification_result_by_source(y_true=y_true_final.cpu().numpy(),
                                                             y_pred=y_pred_final.softmax(1).detach().cpu().numpy(),
                                                             adjusted=True,
                                                             source=sources_final.detach().cpu().numpy())

            final_history = flatten_results(results={'final': clf_result, 'final-adjusted': clf_result_adj})

            if config.enable_wandb:
                wandb.init(
                    name=f'{config.task} : {config.hash} : {config.n_cv}',
                    project=f'incomplete-external_test',
                    config=config.__dict__
                )
                wandb.log(final_history)

        else:
            _, _, _ = main_worker(0, config, pretrained_config, demo_config)


def main_worker(local_rank: int, config: argparse.Namespace, pretrained_config: edict, demo_config: edict = None):

    torch.cuda.set_device(local_rank)
    if config.distributed:
        raise NotImplementedError

    config.batch_size = config.batch_size // config.world_size
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
    if demo_config is None:
        setattr(config, 'use_cdr', False)
        setattr(config, 'scale_demo', True)
    else:
        setattr(config, 'use_cdr', demo_config.use_cdr)
        setattr(config, 'scale_demo', demo_config.scale_demo)

    processor = AOProcessor(root=config.root, data_info='aibl_oasis_data_info.csv',
                            external_data_type=config.external_data_type, use_cdr=config.use_cdr,
                            scale_demo=config.scale_demo, random_state=config.random_state)
    test_only = True if config.train_mode == 'test' else False
    datasets_dict = processor.process(n_splits=config.n_splits, n_cv=config.n_cv, test_only=test_only)

    if not test_only:
        train_set = AODataset(dataset=datasets_dict['train'], external_data_type=config.external_data_type,
                              mri_transform=train_transform_mri, pet_transform=train_transform_pet)
    else:
        train_set = None
    test_set = AODataset(dataset=datasets_dict['test'], external_data_type=config.external_data_type,
                         mri_transform=test_transform_mri, pet_transform=test_transform_pet)

    datasets = {'train': train_set, 'test': test_set}

    # Networks
    if demo_config is not None:

        ckpt = torch.load(pretrained_config.pretrained_file)
        pretrained_networks_names = list(ckpt.keys())
        del ckpt

        networks = build_networks_general_teacher(config=pretrained_config)

        encoder_demo = DemoEncoder(in_channels=len(processor.demo_columns), hidden=demo_config.hidden_demo)
        classifier_demo = LinearDemoClassifier(image_dims=pretrained_config.hidden // 2,
                                               demo_dims=encoder_demo.out_features,
                                               num_classes=2)

        networks['encoder_demo'] = deepcopy(encoder_demo)
        networks['classifier_demo'] = deepcopy(classifier_demo)

        # single
        if '+' not in config.external_data_type:
            networks['extractor_mri_s'] = deepcopy(networks['extractor_mri'])
            networks['projector_mri_s'] = deepcopy(networks['projector_mri'])
            networks['encoder_general_s'] = deepcopy(networks['encoder_general'])

        network_names = list(networks.keys())
        for name in network_names:
            if name not in pretrained_networks_names:
                del networks[name]
            else:
                networks[name].load_weights_from_checkpoint(path=pretrained_config.pretrained_file, key=name)

    else:

        networks = build_networks_general_teacher(config=pretrained_config)
        network_names = list(networks.keys())

        if '+' in config.external_data_type:
            # final multi
            target_network_names = ['extractor_mri', 'projector_mri', 'extractor_pet', 'projector_pet',
                                    'encoder_general', 'encoder_mri', 'encoder_pet', 'classifier']
            for name in network_names:
                if name not in target_network_names:
                    del networks[name]
                else:
                    networks[name].load_weights_from_checkpoint(path=pretrained_config.pretrained_file, key=name)
        else:
            # single
            target_network_names = ['extractor_mri_s', 'projector_mri_s', 'encoder_general_s','classifier_s']
            for name in network_names:
                if name + '_s' not in target_network_names:
                    del networks[name]
                else:
                    networks[name].load_weights_from_checkpoint(path=pretrained_config.pretrained_file, key=name + '_s')
                    networks[name + '_s'] = deepcopy(networks[name])
                    del networks[name]

    # Logging
    if local_rank == 0:
        logfile = os.path.join(config.checkpoint_dir, 'main.log')
        logger = get_rich_logger(logfile=logfile)
        if config.enable_wandb:
            wandb.init(
                name=f'{config.task} : {config.hash} : {config.n_cv}',
                project=f'incomplete-external_test',
                config=config.__dict__
            )
    else:
        logger = None

    # Loss Function
    # Cross Entropy
    class_weight = None
    if pretrained_config.balance:
        class_weight = torch.tensor(processor.class_weight, dtype=torch.float).to(local_rank)
    loss_function_ce = nn.CrossEntropyLoss(weight=class_weight, reduction='mean', ignore_index=-1)

    # Model (Task)
    task_type = 'multi' if '+' in config.external_data_type else 'single'
    model = ExternalTest(networks=networks,
                         config=config,
                         demo_config=demo_config,
                         pretrained_config=pretrained_config,
                         task_type=task_type)
    model.prepare(
        loss_function=loss_function_ce,
        local_rank=local_rank,
    )

    # Train & evaluate
    start = time.time()
    y_true, y_pred, sources = model.run(
        datasets=datasets,
        save_every=config.save_every,
        logger=logger
    )
    elapsed_sec = time.time() - start

    if logger is not None:
        elapsed_mins = elapsed_sec / 60
        logger.info(f'Total training time: {elapsed_mins:,.2f} minutes.')
        logger.handlers.clear()

    del model, train_set, test_set, networks

    return y_true, y_pred, sources


if __name__ == '__main__':

    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
