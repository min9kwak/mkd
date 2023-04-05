import os
import collections

import numpy as np
import wandb

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader

from utils.metrics import classification_result
from utils.logging import make_epoch_description, get_rich_pbar
from utils.optimization import get_optimizer
from utils.optimization import get_cosine_scheduler
from dataset.samplers import ImbalancedDatasetSampler


class Single(object):

    # trainable network in each stage
    network_names = ['encoder', 'projector', 'classifier']

    def __init__(self, networks: dict, data_type: str):

        self.networks = networks
        self.scaler = None
        self.optimizer = None
        self.scheduler = None
        self.prepared = False

        assert data_type in ['mri', 'pet']
        self.data_type = data_type

        if self.networks['projector'] is None:
            del self.networks['projector']
            self.network_names = ['encoder', 'projector']
            self.use_projector = False
        else:
            self.use_projector = True

    def prepare(self,
                checkpoint_dir,
                loss_function_ce,
                optimizer: str = 'adamw',
                learning_rate: float = 0.0001,
                weight_decay: float = 0.00001,
                cosine_warmup: int = 0,
                cosine_cycles: int = 1,
                cosine_min_lr: float = 0.0,
                epochs: int = 100,
                batch_size: int = 4,
                accumulate: int = 4,
                num_workers: int = 4,
                distributed: bool = False,
                local_rank: int = 0,
                mixed_precision: bool = True,
                enable_wandb: bool = True,
                **kwargs):

        # Set attributes
        self.checkpoint_dir = checkpoint_dir
        self.loss_function_ce = loss_function_ce
        self.epochs = epochs
        self.batch_size = batch_size
        self.accumulate = accumulate
        self.num_workers = num_workers
        self.distributed = distributed
        self.local_rank = local_rank
        self.mixed_precision = mixed_precision
        self.enable_wandb = enable_wandb

        if self.accumulate > 1:
            assert self.mixed_precision

        if distributed:
            raise NotImplementedError
        else:
            _ = [v.to(self.local_rank) for k, v in self.networks.items()]

        # Optimization setting
        if self.use_projector:
            params = [{'params': self.networks['encoder'].parameters()},
                      {'params': self.networks['projector'].parameters()},
                      {'params': self.networks['classifier'].parameters()}]
        else:
            params = [{'params': self.networks['encoder'].parameters()},
                      {'params': self.networks['classifier'].parameters()}]
        self.optimizer = get_optimizer(
            params=params,
            name=optimizer,
            lr=learning_rate,
            weight_decay=weight_decay
        )
        self.scheduler = get_cosine_scheduler(
            self.optimizer,
            epochs=self.epochs,
            warmup_steps=cosine_warmup,
            cycles=cosine_cycles,
            min_lr=cosine_min_lr,
            )
        self.scaler = torch.cuda.amp.GradScaler() if mixed_precision else None

        # Ready to train
        self.prepared = True

    def run(self, datasets, save_every: int = 10, **kwargs):

        if not self.prepared:
            raise RuntimeError("Training not prepared.")

        # DataSet & DataLoader
        train_sampler = ImbalancedDatasetSampler(dataset=datasets['train'])

        loaders = {
            'train': DataLoader(dataset=datasets['train'], batch_size=self.batch_size,
                                sampler=train_sampler, drop_last=True),
            'validation': DataLoader(dataset=datasets['validation'], batch_size=self.batch_size,
                                     drop_last=False),
            'test': DataLoader(dataset=datasets['test'], batch_size=self.batch_size,
                               drop_last=False)
        }

        # Logging
        logger = kwargs.get('logger', None)

        # Find the best model by MRI-only test loss
        best_eval_loss = float('inf')
        best_epoch = 0

        if self.enable_wandb:
            len_loader = len(loaders['train'])
            log_freq = (len_loader * self.batch_size // (self.batch_size * self.accumulate)) * self.accumulate
            wandb.watch([v for k, v in self.networks.items()], log='all', log_freq=log_freq)

        for epoch in range(1, self.epochs + 1):

            self.epoch = epoch

            # Train and Test
            train_history = self.train(train_loader=loaders['train'], adjusted=False)
            validation_history = self.evaluate(eval_loader=loaders['validation'], adjusted=False)
            test_history = self.evaluate(eval_loader=loaders['test'], adjusted=False)

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
            eval_loss = validation_history['cross_entropy']
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

        # adjusted evaluation
        test_adjusted_history = self.evaluate(eval_loader=loaders['test'], adjusted=True)

        last_history = collections.defaultdict(dict)
        for k, v in test_adjusted_history.items():
            last_history[f'adjusted/{k}'] = v

        if self.enable_wandb:
            wandb.log(last_history)

    def train(self, train_loader, adjusted=False):

        self._set_learning_phase(train=True)

        # Logging
        len_loader = len(train_loader)
        steps = (len_loader * self.batch_size // (self.batch_size * self.accumulate)) * self.accumulate

        result = {'cross_entropy': torch.zeros(steps, device=self.local_rank)}

        with get_rich_pbar(transient=True, auto_refresh=False) as pg:

            if self.local_rank == 0:
                task = pg.add_task(f"[bold red] Training...", total=steps)

            y_true, y_pred = [], []
            for i, batch in enumerate(train_loader):
                with torch.cuda.amp.autocast(self.mixed_precision):

                    # input data
                    x = batch[f'{self.data_type}'].float().to(self.local_rank)
                    y = batch['y'].long().to(self.local_rank)

                    # hidden representations
                    h = self.networks['encoder'](x)
                    if self.use_projector:
                        h = self.networks['projector'](h)
                    logits = self.networks['classifier'](h)

                    loss = self.loss_function_ce(logits, y)
                    loss = loss / self.accumulate

                # Accumulates scaled gradients
                if self.scaler is not None:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()

                # Update Optimizer
                if (i + 1) % self.accumulate == 0:
                    if self.scaler is not None:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()
                    self.optimizer.zero_grad()

                # save monitoring values
                result['cross_entropy'][i] = loss * self.accumulate

                if self.local_rank == 0:
                    desc = f"[bold green] Epoch {self.epoch} [{i+1}/{steps}]: "
                    for k, v in result.items():
                        desc += f" {k} : {v[:i+1].mean():.4f} |"
                    pg.update(task, advance=1., description=desc)
                    pg.refresh()

                y_true.append(y.long())
                y_pred.append(logits)

                # Ignore the remained batches
                if (i + 1) == steps:
                    break

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

    @torch.no_grad()
    def evaluate(self, eval_loader, adjusted=False):

        self._set_learning_phase(train=False)
        steps = len(eval_loader)
        result = {'cross_entropy': torch.zeros(steps, device=self.local_rank)}

        with get_rich_pbar(transient=True, auto_refresh=False) as pg:

            if self.local_rank == 0:
                task = pg.add_task(f"[bold red] Evaluating...", total=steps)

            y_true, y_pred = [], []
            for i, batch in enumerate(eval_loader):
                with torch.cuda.amp.autocast(self.mixed_precision):

                    # input data
                    x = batch[f'{self.data_type}'].float().to(self.local_rank)
                    y = batch['y'].long().to(self.local_rank)

                    # hidden representations
                    h = self.networks['encoder'](x)
                    if self.use_projector:
                        h = self.networks['projector'](h)
                    logits = self.networks['classifier'](h)
                    loss = self.loss_function_ce(logits, y)

                # save monitoring values
                result['cross_entropy'][i] = loss

                if self.local_rank == 0:
                    pg.update(task, advance=1.)
                    pg.refresh()

                y_true.append(y.long())
                y_pred.append(logits)

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
