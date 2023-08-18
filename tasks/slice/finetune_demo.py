import argparse
import os
import collections
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from utils.metrics import classification_result

import wandb
from utils.logging import make_epoch_description, get_rich_pbar
from datasets.samplers import ImbalancedDatasetSampler, StratifiedSampler
from utils.optimization import get_optimizer
from utils.optimization import get_cosine_scheduler

from easydict import EasyDict as edict


class DemoClassification(object):
    def __init__(self,
                 networks: dict,
                 task_type: str,
                 ):

        self.networks = networks
        self.scaler = None
        self.optimizer = None
        self.scheduler = None
        self.prepared = False

        self.networks = {k: v for k, v in self.networks.items() if v is not None}
        self.network_names = list(self.networks.keys())

        assert task_type in ['single', 'multi']
        self.task_type = task_type

        if self.task_type == 'single':
            self.train_step = self.train_step_single
        elif self.task_type == 'multi':
            self.train_step = self.train_step_multi
        else:
            raise ValueError('Invalid task')

    def prepare(self,
                config: argparse.Namespace,
                pretrained_config: edict,
                loss_function: nn.Module,
                local_rank: int = 0,
                **kwargs):  # pylint: disable=unused-argument

        # Set attributes
        self.config = config
        self.pretrained_config = pretrained_config

        self.loss_function = loss_function

        self.checkpoint_dir = config.checkpoint_dir
        self.epochs = config.epochs
        self.batch_size = config.batch_size
        self.num_workers = config.num_workers
        self.distributed = config.distributed
        self.local_rank = local_rank

        self.mixed_precision = config.mixed_precision
        self.enable_wandb = config.enable_wandb

        if self.pretrained_config.train_slices == 'random':
            self.test_num_slices = 3
        elif self.pretrained_config.train_slices == 'fixed':
            # Use all slices for test
            self.test_num_slices = self.pretrained_config.num_slices
        elif self.pretrained_config.train_slices in ['sagittal', 'coronal', 'axial']:
            self.test_num_slices = 1
        else:
            raise ValueError

        if self.distributed:
            raise NotImplementedError
        else:
            _ = [v.to(self.local_rank) for k, v in self.networks.items()]

        params = []
        for name in self.networks.keys():
            if config.different_lr_demo and ('demo' in name):
                params = params + [{'params': self.networks[name].parameters(),
                                    'lr': self.config.learning_rate * 10}]
            else:
                params = params + [{'params': self.networks[name].parameters(), 'lr': self.config.learning_rate}]
        self.optimizer = get_optimizer(params=params, name=config.optimizer,
                                       lr=config.learning_rate, weight_decay=config.weight_decay)
        self.scheduler = get_cosine_scheduler(self.optimizer, epochs=self.epochs, warmup_steps=config.cosine_warmup,
                                              cycles=config.cosine_cycles, min_lr=config.cosine_min_lr)
        self.scaler = torch.cuda.amp.GradScaler() if config.mixed_precision else None

        # Ready to train!
        self.prepared = True

    def run(self, datasets, save_every: int = 20, **kwargs):

        if not self.prepared:
            raise RuntimeError("Training not prepared.")

        # DataSet & DataLoader
        train_sampler = None
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
                'train': DataLoader(dataset=datasets['train'], batch_size=self.batch_size, shuffle=True,
                                    sampler=train_sampler, drop_last=True),
                'test': DataLoader(dataset=datasets['test'], batch_size=self.batch_size, drop_last=False)
            }

        # Logging
        logger = kwargs.get('logger', None)

        # Find the best model by total loss
        best_eval_loss = float('inf')
        best_epoch = 0

        for epoch in range(1, self.epochs + 1):

            self.epoch = epoch

            # Train and Test
            epoch_history = collections.defaultdict(dict)
            train_history = self.train(loaders['train'], train=True, adjusted=False)
            with torch.no_grad():
                test_history = self.train(loaders['test'], train=False, adjusted=False)

            # Logging
            for mode, history in zip(['train', 'test'], [train_history, test_history]):
                for k, v in history.items():
                    epoch_history[f'{mode}/{k}'] = v

            log = make_epoch_description(
                history=epoch_history,
                current=epoch,
                total=self.epochs,
                best=best_epoch,
            )
            if logger is not None:
                logger.info(log)

            if self.enable_wandb:
                wandb.log({'epoch': epoch}, commit=False)
                if self.scheduler is not None:
                    wandb.log({'lr': self.scheduler.get_last_lr()[0]}, commit=False)
                else:
                    wandb.log({'lr': self.optimizer.param_groups[0]['lr']}, commit=False)
                wandb.log(epoch_history)

            # Skip the best model

            # Save intermediate model checkpoints
            if (epoch % save_every == 0) & (self.local_rank == 0):
                ckpt = os.path.join(self.checkpoint_dir, f"ckpt.{epoch}.pth.tar")
                self.save_checkpoint(ckpt, epoch=epoch)

            # Update learning rate
            if self.scheduler is not None:
                self.scheduler.step()

        # Save final model checkpoint
        ckpt = os.path.join(self.checkpoint_dir, f"ckpt.last.pth.tar")
        self.save_checkpoint(ckpt, epoch=epoch)

        # adjusted evaluation (last)
        with torch.no_grad():
            test_history = self.train(loaders['test'], train=False, adjusted=True)
        last_history = collections.defaultdict(dict)

        for k, v in test_history.items():
            last_history[f'adjusted-last/test/{k}'] = v
        if self.enable_wandb:
            wandb.log(last_history)

    def train(self, data_loader, train=True, adjusted=False):
        """Training defined for a single epoch."""

        self._set_learning_phase(train=train)

        steps = len(data_loader)
        result = {
            'loss_ce': torch.zeros(steps, device=self.local_rank)
        }

        with get_rich_pbar(transient=True, auto_refresh=False) as pg:

            if self.local_rank == 0:
                task = pg.add_task(f"[bold red] Training...", total=steps)

            y_true, y_pred = [], []
            for i, batch in enumerate(data_loader):
                with torch.cuda.amp.autocast(self.mixed_precision):
                    loss_ce, y, logit = self.train_step(batch)
                if train:
                    self.update(loss_ce)

                result['loss_ce'][i] = loss_ce.detach()

                if self.local_rank == 0:
                    desc = f"[bold green] Epoch {self.epoch} [{i + 1}/{steps}]: "

                    for k, v in result.items():
                        desc += f" {k} : {v[:i + 1].mean():.4f} |"
                    pg.update(task, advance=1., description=desc)
                    pg.refresh()

                # Save only labeled samples
                y = y.chunk(self.pretrained_config.num_slices)[0].long()
                y_true.append(y)

                num_classes = logit.shape[-1]
                logit = logit.reshape(self.pretrained_config.num_slices, -1, num_classes).mean(0)
                y_pred.append(logit)

        result = {k: v.mean().item() for k, v in result.items()}

        # enforce to float32: accuracy and macro f1 score
        y_true = torch.cat(y_true, dim=0)
        y_pred = torch.cat(y_pred, dim=0).to(torch.float32)

        clf_result = classification_result(y_true=y_true.cpu().numpy(),
                                           y_pred=y_pred.softmax(1).detach().cpu().numpy(),
                                           adjusted=adjusted)
        for k, v in clf_result.items():
            result[k] = v

        return result

    def train_step_multi(self, batch):

        x_mri = torch.concat(batch['mri']).float().to(self.local_rank)
        x_pet = torch.concat(batch['pet']).float().to(self.local_rank)
        demo = batch['demo'].float().repeat(self.pretrained_config.num_slices, 1).to(self.local_rank)
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
        z_demo = self.networks['encoder_demo'](demo)

        # classifier
        logit = self.networks['classifier_demo'](z_image, z_demo)

        loss_ce = self.loss_function(logit, y)

        return loss_ce, y, logit

    def train_step_single(self, batch):

        x_mri = torch.concat(batch['mri']).float().to(self.local_rank)
        demo = batch['demo'].float().repeat(self.pretrained_config.num_slices, 1).to(self.local_rank)
        y = batch['y'].long().repeat(self.pretrained_config.num_slices).to(self.local_rank)

        # forward through final multi
        h_mri = self.networks['projector_mri_s'](self.networks['extractor_mri_s'](x_mri))
        z_image = self.networks['encoder_general_s'](h_mri)

        # forward through demo
        z_demo = self.networks['encoder_demo'](demo)

        # classifier
        logit = self.networks['classifier_demo'](z_image * 2, z_demo)

        loss_ce = self.loss_function(logit, y)

        return loss_ce, y, logit

    def _set_learning_phase(self, train: bool = True):
        # Teacher is fixed to eval mode
        for name in self.networks.keys():
            if train:
                self.networks[name].train()
            else:
                self.networks[name].eval()

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

    @staticmethod
    def freeze_params(net: nn.Module):
        for p in net.parameters():
            p.requires_grad = False
