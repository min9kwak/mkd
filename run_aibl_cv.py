# -*- coding: utf-8 -*-
import argparse
import collections
import copy
import os
import sys
import json
import time
import rich
import numpy as np
import pickle
import wandb

import torch
import torch.nn as nn

from configs.slice.aibl import AIBLConfig
from tasks.slice.aibl_cv import AIBLCV

from models.slice.build import build_networks_general_teacher

from datasets.aibl import AIBLProcessor, AIBLDataset
from datasets.slice.transforms import make_mri_transforms

from utils.logging import get_rich_logger
from utils.gpu import set_gpu
from utils.metrics import classification_result


def main():
    """Main function for single/distributed linear classification."""

    config = AIBLConfig.parse_arguments()

    student_file = os.path.join(config.student_dir, f"ckpt.{config.student_position}.pth.tar")
    setattr(config, 'student_file', student_file)

    student_config = os.path.join(config.student_dir, "configs.json")
    with open(student_config, 'rb') as fb:
        student_config = json.load(fb)

    # inherit pretrained configs
    for key in student_config.keys():
        if key not in config.__dict__.keys():
            # property cannot be done
            try:
                setattr(config, key, student_config[key])
            except:
                pass
        else:
            try:
                setattr(config, f'{key}_s', student_config[key])
            except:
                pass
    setattr(config, 'hash_s', student_config['hash'])

    config.task = 'AIBL_CV'

    if config.server == 'main':
        setattr(config, 'root', 'D:/data/AIBL')
    else:
        setattr(config, 'root', '/raidWorkspace/mingu/Data/AIBL')

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

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.allow_tf32 = True

    if config.distributed:
        raise NotImplementedError
    else:
        rich.print(f"Single GPU training.")

        y_true_final, y_pred_final = [], []
        for n_cv in range(config.n_splits):
            # single machine, single gpu
            setattr(config, 'n_cv', n_cv)
            y_true, y_pred = main_worker(0, config=config)
            y_true_final.append(y_true)
            y_pred_final.append(y_pred)

        y_true_final = torch.cat(y_true_final, dim=0)
        y_pred_final = torch.cat(y_pred_final, dim=0).to(torch.float32)

        clf_result = classification_result(y_true=y_true_final.numpy(),
                                           y_pred=y_pred_final.softmax(1).numpy(),
                                           adjusted=False)
        clf_result_adj = classification_result(y_true=y_true_final.numpy(),
                                               y_pred=y_pred_final.softmax(1).numpy(),
                                               adjusted=True)

        final_history = collections.defaultdict(dict)
        # TODO: final-{suffix}
        for k, v in clf_result.items():
            final_history[f'adjusted-plain/{k}'] = v
        for k, v in clf_result_adj.items():
            final_history[f'adjusted-adjusted/{k}'] = v

        if config.enable_wandb:
            wandb.init(
                name=f'{config.task} : {config.hash} : final',
                project=f'incomplete-kd-{config.task}',
                config=config.__dict__
            )
            wandb.log(final_history)


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

    # Dataset
    processor = AIBLProcessor(root=config.root,
                              data_info='data_info_mri.csv',
                              time_window=36,
                              random_state=config.random_state)
    test_only = True if config.train_mode == 'test' else False
    datasets_dict = processor.process(n_splits=config.n_splits, n_cv=config.n_cv, test_only=test_only)

    if not test_only:
        train_set = AIBLDataset(dataset=datasets_dict['train'], transform=train_transform_mri)
    else:
        train_set = None
    test_set = AIBLDataset(dataset=datasets_dict['test'], transform=test_transform_mri)

    # cross-entropy loss function
    if (config.balance) and (not test_only):
        class_weight = torch.tensor(processor.class_weight, dtype=torch.float).to(local_rank)
        loss_function = nn.CrossEntropyLoss(weight=class_weight)
    else:
        loss_function = nn.CrossEntropyLoss()

    # student networks
    architectures = build_networks_general_teacher(config=config)
    student_network_names = ['extractor_mri', 'projector_mri', 'encoder_general', 'classifier']
    networks = {f'{k}_s': copy.deepcopy(v) for k, v in architectures.items()
                if f'{k}_s' in student_network_names}
    for name, network in networks.items():
        network.load_weights_from_checkpoint(path=config.student_file, key=name)
    del architectures

    if local_rank == 0:
        logfile = os.path.join(config.checkpoint_dir, 'main.log')
        logger = get_rich_logger(logfile=logfile)
        if config.enable_wandb:
            wandb.init(
                name=f'{config.task} : {config.hash} : {config.n_cv}',
                project=f'incomplete-kd-{config.task}',
                config=config.__dict__
            )
    else:
        logger = None

    # Model (Task)
    model = AIBLCV(networks=networks, config=config)
    model.prepare(
        loss_function=loss_function,
        local_rank=local_rank,
    )

    # Train & evaluate
    start = time.time()
    y_true, y_pred = model.run(
        train_set=train_set,
        test_set=test_set,
        save_every=config.save_every,
        logger=logger
    )
    elapsed_sec = time.time() - start

    if logger is not None:
        elapsed_mins = elapsed_sec / 60
        logger.info(f'Total training time: {elapsed_mins:,.2f} minutes.')
        logger.handlers.clear()

    del model, train_set, test_set, networks

    return y_true, y_pred


if __name__ == '__main__':

    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
