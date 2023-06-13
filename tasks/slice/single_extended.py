import os
import collections
import wandb
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils.metrics import classification_result
from utils.logging import make_epoch_description, get_rich_pbar
from utils.optimization import get_optimizer
from utils.optimization import get_cosine_scheduler
from datasets.samplers import ImbalancedDatasetSampler, StratifiedSampler


class SingleExtended(object):

    # trainable network in each stage. MRI-only model
    network_names = ['extractor', 'projector', 'encoder', 'classifier']

    def __init__(self, networks: dict, data_type: str):

        self.networks = networks
        self.scaler = None
        self.optimizer = None
        self.scheduler = None
        self.prepared = False

        assert data_type == 'mri'
        self.networks = {k: v for k, v in self.networks.items() if v is not None}
        self.network_names = list(self.networks.keys())

    def prepare(self,
                config: argparse.Namespace,
                loss_function_ce,
                local_rank: int = 0,
                **kwargs):

        # Set attributes
        self.config = config

        self.loss_function_ce = loss_function_ce

        self.checkpoint_dir = config.checkpoint_dir
        self.epochs = config.epochs
        self.batch_size = config.batch_size
        self.num_workers = config.num_workers
        self.distributed = config.distributed
        self.local_rank = local_rank

        self.mixed_precision = config.mixed_precision
        self.enable_wandb = config.enable_wandb

        if self.config.train_slices == 'random':
            self.test_num_slices = 3
        elif self.config.train_slices == 'fixed':
            # Use all slices for test
            self.test_num_slices = self.config.num_slices
        elif self.config.train_slices in ['sagittal', 'coronal', 'axial']:
            self.test_num_slices = 1
        else:
            raise ValueError

        if self.distributed:
            raise NotImplementedError
        else:
            _ = [v.to(self.local_rank) for k, v in self.networks.items()]

        # Optimization setting
        params = []
        for name in self.networks.keys():
            if self.config.different_lr:
                if name.startswith('encoder_') or name.startswith('decoder_'):
                    params = params + [{'params': self.networks[name].parameters(),
                                        'lr': self.config.learning_rate / 10}]
                else:
                    params = params + [{'params': self.networks[name].parameters(),
                                        'lr': self.config.learning_rate}]
            else:
                params = params + [{'params': self.networks[name].parameters(),
                                    'lr': self.config.learning_rate}]

        self.optimizer = get_optimizer(params=params, name=config.optimizer,
                                       lr=config.learning_rate, weight_decay=config.weight_decay)
        self.scheduler = get_cosine_scheduler(self.optimizer, epochs=self.epochs, warmup_steps=config.cosine_warmup,
                                              cycles=config.cosine_cycles, min_lr=config.cosine_min_lr)
        self.scaler = torch.cuda.amp.GradScaler() if config.mixed_precision else None

        # Ready to train
        self.prepared = True

    def run(self, datasets, save_every: int = 10, **kwargs):

        if not self.prepared:
            raise RuntimeError("Training not prepared.")

        # DataSet & DataLoader
        train_sampler = None
        if self.config.sampler_type == 'over':
            train_sampler = ImbalancedDatasetSampler(dataset=datasets['train'])
        elif self.config.sampler_type == 'stratified':
            train_sampler = StratifiedSampler(class_vector=datasets['train'].y, batch_size=self.batch_size)

        if train_sampler is not None:
            loaders = {
                'train': DataLoader(dataset=datasets['train'], batch_size=self.batch_size,
                                    sampler=train_sampler, drop_last=True),
                'validation': DataLoader(dataset=datasets['validation'], batch_size=self.batch_size, drop_last=False),
                'test': DataLoader(dataset=datasets['test'], batch_size=self.batch_size, drop_last=False)
            }
        else:
            loaders = {
                'train': DataLoader(dataset=datasets['train'], batch_size=self.batch_size, shuffle=True,
                                    sampler=train_sampler, drop_last=True),
                'validation': DataLoader(dataset=datasets['validation'], batch_size=self.batch_size, drop_last=False),
                'test': DataLoader(dataset=datasets['test'], batch_size=self.batch_size, drop_last=False)
            }

        # Logging
        logger = kwargs.get('logger', None)

        if self.enable_wandb:
            wandb.watch([v for k, v in self.networks.items()], log='all', log_freq=len(loaders['train']))

        # Find the best model by MRI-only test loss
        best_eval_loss = float('inf')
        best_epoch = 0

        for epoch in range(1, self.epochs + 1):

            self.epoch = epoch

            # Train and Test
            train_history = self.train(data_loader=loaders['train'], adjusted=False)
            validation_history = self.evaluate(data_loader=loaders['validation'], adjusted=False)
            test_history = self.evaluate(data_loader=loaders['test'], adjusted=False)

            # Logging
            epoch_history = collections.defaultdict(dict)
            for mode, history in zip(['train', 'validation', 'test'],
                                     [train_history, validation_history, test_history]):
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

            # Save best model checkpoint
            eval_loss = validation_history['loss_ce']
            if eval_loss <= best_eval_loss:
                best_eval_loss = eval_loss
                best_epoch = epoch
                if self.local_rank == 0:
                    ckpt = os.path.join(self.checkpoint_dir, f"ckpt.best.pth.tar")
                    self.save_checkpoint(ckpt, epoch=epoch)

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
        validation_history = self.evaluate(loaders['validation'], adjusted=True)
        test_history = self.evaluate(loaders['test'], adjusted=True)

        last_history = collections.defaultdict(dict)
        for mode, history in zip(['validation', 'test'], [validation_history, test_history]):
            for k, v in history.items():
                last_history[f'adjusted-last/{mode}/{k}'] = v
        if self.enable_wandb:
            wandb.log(last_history)

        # adjusted evaluation (best)
        ckpt = os.path.join(self.checkpoint_dir, f"ckpt.best.pth.tar")
        for k, v in self.networks.items():
            v.load_weights_from_checkpoint(path=ckpt, key=k)

        validation_history = self.evaluate(loaders['validation'], adjusted=True)
        test_history = self.evaluate(loaders['test'], adjusted=True)

        best_history = collections.defaultdict(dict)
        for mode, history in zip(['validation', 'test'], [validation_history, test_history]):
            for k, v in history.items():
                best_history[f'adjusted-best/{mode}/{k}'] = v
        if self.enable_wandb:
            wandb.log(best_history)

    def train(self, data_loader, adjusted=False):

        self._set_learning_phase(train=True)

        # Logging
        steps = len(data_loader)
        result = {'loss_ce': torch.zeros(steps, device=self.local_rank)}

        with get_rich_pbar(transient=True, auto_refresh=False) as pg:

            if self.local_rank == 0:
                task = pg.add_task(f"[bold red] Training...", total=steps)

            y_true, y_pred = [], []
            for i, batch in enumerate(data_loader):
                with torch.cuda.amp.autocast(self.mixed_precision):
                    loss_ce, y, logit = self.train_step(batch)
                self.update(loss_ce)

                # save monitoring values
                result['loss_ce'][i] = loss_ce.detach()

                if self.local_rank == 0:
                    desc = f"[bold green] Epoch {self.epoch} [{i+1}/{steps}]: "
                    for k, v in result.items():
                        desc += f" {k} : {v[:i+1].mean():.4f} |"
                    pg.update(task, advance=1., description=desc)
                    pg.refresh()

                # save only labeled samples
                y = y.chunk(self.config.num_slices)[0].long()
                y_true.append(y)

                num_classes = logit.shape[-1]
                logit = logit.reshape(self.config.num_slices, -1, num_classes).mean(0)
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

    def train_step(self, batch):

        # input data
        x_mri = torch.concat(batch['mri']).float().to(self.local_rank)
        y = batch['y'].long().repeat(self.config.num_slices).to(self.local_rank)

        h_mri = self.networks['projector'](self.networks['extractor'](x_mri))
        z_mri = self.networks['encoder'](h_mri)
        logit = self.networks['classifier'](z_mri)

        loss_ce = self.loss_function_ce(logit, y)
        loss_ce = loss_ce.mean()

        return loss_ce, y, logit

    @torch.no_grad()
    def evaluate(self, data_loader, adjusted=False):

        self._set_learning_phase(train=False)

        # Logging
        steps = len(data_loader)
        result = {'loss_ce': torch.zeros(steps, device=self.local_rank)}

        with get_rich_pbar(transient=True, auto_refresh=False) as pg:

            if self.local_rank == 0:
                task = pg.add_task(f"[bold red] Evaluating...", total=steps)

            y_true, y_pred = [], []
            for i, batch in enumerate(data_loader):
                with torch.cuda.amp.autocast(self.mixed_precision):
                    loss_ce, y, logit = self.train_step(batch)

                # save monitoring values
                result['loss_ce'][i] = loss_ce.detach()

                if self.local_rank == 0:
                    pg.update(task, advance=1.)
                    pg.refresh()

                y_true.append(y.chunk(self.test_num_slices)[0].long())
                num_classes = logit.shape[-1]
                logit = logit.reshape(self.config.num_slices, -1, num_classes).mean(0)
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

    @staticmethod
    def move_optimizer_states(optimizer: torch.optim.Optimizer, device: int = 0):
        for state in optimizer.state.values():  # dict; state of parameters
            for k, v in state.items():          # iterate over paramteters (k=name, v=tensor)
                if torch.is_tensor(v):          # If a tensor,
                    state[k] = v.to(device)     # configure appropriate device

    @staticmethod
    def freeze_params(net: nn.Module, freeze: bool):
        for p in net.parameters():
            p.requires_grad = not freeze

    def _set_learning_phase(self, train: bool = True):
        for _, network in self.networks.items():
            if train:
                network.train()
            else:
                network.eval()

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
