import argparse
import os
import collections
import torch
import torch.nn as nn
import numpy as np

from torch.utils.data import DataLoader
from utils.metrics import classification_result_by_source, flatten_results

import wandb
from utils.logging import make_epoch_description, get_rich_pbar
from datasets.samplers import ImbalancedDatasetSampler, StratifiedSampler
from utils.optimization import get_optimizer
from utils.optimization import get_cosine_scheduler

from easydict import EasyDict as edict


class ExternalTest(object):
    def __init__(self,
                 networks: dict,
                 config: argparse.Namespace,
                 demo_config: edict = None,
                 pretrained_config: edict = None,
                 task_type: str = 'multi',
    ):

        # network
        self.networks = networks

        # optimizer
        self.scaler = None
        self.optimizer = None
        self.scheduler = None
        self.loss_function = None

        # demo_config exists only if model is trained with demographic information
        self.config = config
        self.demo_config = demo_config
        self.pretrained_config = pretrained_config

        # prepared
        self.prepared = False

        self.networks = {k: v for k, v in self.networks.items() if v is not None}
        self.network_names = list(self.networks.keys())

        assert task_type in ['single', 'multi']
        self.task_type = task_type
        if self.task_type == 'single':
            self.train_step = self.train_step_single
        else:
            self.train_step = self.train_step_multi

        if self.demo_config is None:
            self.use_demo = False
        else:
            self.use_demo = True

    def prepare(self,
                loss_function: nn.Module,
                local_rank: int = 0,
                **kwargs):  # pylint: disable=unused-argument

        # Set attributes
        self.loss_function = loss_function

        self.checkpoint_dir = self.config.checkpoint_dir
        self.epochs = self.config.epochs
        self.batch_size = self.config.batch_size
        self.num_workers = self.config.num_workers
        self.local_rank = local_rank
        self.mixed_precision = self.config.mixed_precision
        self.enable_wandb = self.config.enable_wandb

        if self.pretrained_config.train_slices == 'random':
            self.test_num_slices = 3
        elif self.pretrained_config.train_slices == 'fixed':
            # Use all slices for test
            self.test_num_slices = self.pretrained_config.num_slices
        elif self.pretrained_config.train_slices in ['sagittal', 'coronal', 'axial']:
            self.test_num_slices = 1
        else:
            raise ValueError

        if self.config.distributed:
            raise NotImplementedError
        else:
            _ = [v.to(self.local_rank) for k, v in self.networks.items()]

        # Optimization
        params = []
        for name in self.networks.keys():
            params = params + [{'params': self.networks[name].parameters(),
                                'lr': self.config.learning_rate}]
        self.optimizer = get_optimizer(params=params, name=self.config.optimizer,
                                       lr=self.config.learning_rate, weight_decay=self.config.weight_decay)
        self.scheduler = get_cosine_scheduler(
            self.optimizer,
            epochs=self.epochs,
            warmup_steps=self.config.cosine_warmup,
            cycles=self.config.cosine_cycles,
            min_lr=self.config.cosine_min_lr,
            )
        self.scaler = torch.cuda.amp.GradScaler() if self.mixed_precision else None

        # Ready to train!
        self.prepared = True

    def run(self,
            datasets: dict,
            save_every: int = 1000,
            **kwargs):

        if not self.prepared:
            raise RuntimeError("Training not prepared.")

        # DataSet & DataLoader
        train_sampler = None
        if hasattr(datasets, 'train'):
            if self.pretrained_config.sampler_type == 'over':
                train_sampler = ImbalancedDatasetSampler(dataset=datasets['train'])
            elif self.pretrained_config.sampler_type == 'stratified':
                train_sampler = StratifiedSampler(class_vector=datasets['train'].y, batch_size=self.batch_size)

        if train_sampler is not None:
            loaders = {
                'train': DataLoader(dataset=datasets['train'], batch_size=self.batch_size,
                                    sampler=train_sampler, drop_last=True),
                'test': DataLoader(dataset=datasets['test'], batch_size=self.batch_size, drop_last=False)
            }
        else:
            loaders = {
                'test': DataLoader(dataset=datasets['test'], batch_size=self.batch_size, drop_last=False)
            }

        # Logging
        logger = kwargs.get('logger', None)

        # Find the best model by total loss
        best_eval_loss = float('inf')
        best_epoch = 0

        # Supervised training
        epoch = 1
        with torch.no_grad():
            test_history = self.train(loaders['test'], train=False, adjusted=False)
        epoch_history = flatten_results(results={'test': test_history})

        if self.enable_wandb:
            wandb.log({'epoch': epoch}, commit=False)
            if self.scheduler is not None:
                wandb.log({'lr': self.scheduler.get_last_lr()[0]}, commit=False)
            else:
                wandb.log({'lr': self.optimizer.param_groups[0]['lr']}, commit=False)
            wandb.log(epoch_history)

        # adjusted evaluation
        with torch.no_grad():
            test_history, y_true, y_pred, sources = self.train(loaders['test'], train=False,
                                                               adjusted=True, return_values=True)
        epoch_history = flatten_results(results={'adjusted': test_history})
        if self.enable_wandb:
            wandb.log(epoch_history)

        del loaders
        self.networks = None

        return y_true.detach().cpu(), y_pred.detach().cpu(), sources.detach().cpu()

    def train(self, data_loader, train=True, adjusted=False, return_values=False):
        """Training defined for a single epoch."""

        steps = len(data_loader)
        self._set_learning_phase(train=True)
        result = {
            'loss_ce': torch.zeros(steps, device=self.local_rank)
        }

        with get_rich_pbar(transient=True, auto_refresh=False) as pg:

            if self.local_rank == 0:
                task = pg.add_task(f"[bold red] Training...", total=steps)

            y_true, y_pred = [], []
            sources = []
            for i, batch in enumerate(data_loader):
                with torch.cuda.amp.autocast(self.mixed_precision):
                    loss_ce, y, logit, source = self.train_step(batch)
                if train:
                    self.update(loss_ce)
                result['loss_ce'][i] = loss_ce.detach()

                if self.local_rank == 0:
                    desc = f"[bold green] [{i+1}/{steps}]: "
                    for k, v in result.items():
                        desc += f" {k} : {v[:i+1].mean():.4f} |"
                    pg.update(task, advance=1., description=desc)
                    pg.refresh()

                # Save only labeled samples
                y = y.chunk(self.pretrained_config.num_slices)[0].long()
                y_true.append(y)

                num_classes = logit.shape[-1]
                logit = logit.reshape(self.pretrained_config.num_slices, -1, num_classes).mean(0)
                y_pred.append(logit)

                source = source.long()
                sources.append(source)

        result = {k: v.mean().item() for k, v in result.items()}

        # enforce to float32: accuracy and macro f1 score
        y_true = torch.cat(y_true, dim=0)
        y_pred = torch.cat(y_pred, dim=0).to(torch.float32)
        sources = torch.cat(sources)

        clf_result = classification_result_by_source(y_true=y_true.cpu().numpy(),
                                                     y_pred=y_pred.softmax(1).detach().cpu().numpy(),
                                                     adjusted=adjusted,
                                                     source=sources.detach().cpu().numpy())
        for k, v in clf_result.items():
            result[k] = v

        if return_values:
            return result, y_true, y_pred, sources
        else:
            return result

    def train_step_multi(self, batch):

        x_mri = torch.concat(batch['mri']).float().to(self.local_rank)
        x_pet = torch.concat(batch['pet']).float().to(self.local_rank)
        source = batch['source'].long()
        y = batch['y'].long().repeat(self.pretrained_config.num_slices).to(self.local_rank)

        # forward through final multi
        h_mri = self.networks['projector_mri'](self.networks['extractor_mri'](x_mri))
        h_pet = self.networks['projector_pet'](self.networks['extractor_pet'](x_pet))

        z_mri_general = self.networks['encoder_general'](h_mri)
        z_pet_general = self.networks['encoder_general'](h_pet)
        z_mri = self.networks['encoder_mri'](h_mri)
        z_pet = self.networks['encoder_pet'](h_pet)

        if self.pretrained_config.use_specific_final:
            z_image = z_mri_general + z_pet_general + z_mri + z_pet
        else:
            z_image = z_mri_general + z_pet_general

        # forward through demo
        if self.use_demo:
            demo = batch['demo'].float().repeat(self.pretrained_config.num_slices, 1).to(self.local_rank)
            z_demo = self.networks['encoder_demo'](demo)
            logit = self.networks['classifier_demo'](z_image, z_demo)
        else:
            logit = self.networks['classifier'](z_image)

        loss_ce = self.loss_function(logit, y)
        y = batch['y'].long().repeat(self.pretrained_config.num_slices).to(self.local_rank)

        return loss_ce, y, logit, source

    def train_step_single(self, batch):

        x_mri = torch.concat(batch['mri']).float().to(self.local_rank)
        source = batch['source'].long()
        y = batch['y'].long().repeat(self.pretrained_config.num_slices).to(self.local_rank)

        # forward through final multi
        h_mri = self.networks['projector_mri_s'](self.networks['extractor_mri_s'](x_mri))
        z_image = self.networks['encoder_general_s'](h_mri)

        # forward through demo
        if self.use_demo:
            demo = batch['demo'].float().repeat(self.pretrained_config.num_slices, 1).to(self.local_rank)
            z_demo = self.networks['encoder_demo'](demo)
            logit = self.networks['classifier_demo'](z_image * 2, z_demo)
        else:
            logit = self.networks['classifier_s'](z_image * 2)

        loss_ce = self.loss_function(logit, y)

        return loss_ce, y, logit, source

    def _set_learning_phase(self, train: bool = True):
        for name in self.networks.keys():
            if train:
                self.networks[name].train()
            else:
                self.networks[name].eval()

    @staticmethod
    def freeze_params(net: nn.Module):
        for p in net.parameters():
            p.requires_grad = False

    def save_checkpoint(self, path: str, **kwargs):
        ckpt = {k: v.state_dict() for k, v in self.networks.items()}
        if kwargs:
            ckpt.update(kwargs)
        torch.save(ckpt, path)

    def update(self, loss):
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            self.optimizer.step()
        self.optimizer.zero_grad()
