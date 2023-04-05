import os
import collections
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


class IMFi(object):

    # trainable network in each stage
    network_names = ['mri_encoder', 'pet_encoder',
                     'mri_projector', 'pet_projector', 'common_projector',
                     'predictor', 'classifier']

    stage2network = {
        1: ['mri_encoder', 'pet_encoder', 'mri_projector', 'pet_projector', 'common_projector', 'classifier'],
        2: ['predictor'],
        3: ['mri_encoder']
    }

    def __init__(self,
                 networks: dict):

        self.networks = networks
        self.scaler = None
        self.optimizer = None
        self.scheduler = None
        self.prepared = False

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
                common_type: str = 'concat',
                num_workers: int = 4,
                distributed: bool = False,
                local_rank: int = 0,
                cosine_ratio: float = 0.5,
                mixed_precision: bool = True,
                enable_wandb: bool = True,
                **kwargs):

        # Set attributes
        self.checkpoint_dir = checkpoint_dir
        self.loss_function_ce = loss_function_ce
        self.epochs = epochs
        self.batch_size = batch_size
        self.accumulate = accumulate
        self.common_type = common_type
        self.num_workers = num_workers
        self.distributed = distributed
        self.local_rank = local_rank
        self.cosine_ratio = cosine_ratio
        self.mixed_precision = mixed_precision
        self.enable_wandb = enable_wandb

        if distributed:
            raise NotImplementedError
        else:
            _ = [v.to(self.local_rank) for k, v in self.networks.items()]

        # Optimization setting
        self.optimizer = get_optimizer(
            params=[
                {'params': self.networks['mri_encoder'].parameters()},
                {'params': self.networks['pet_encoder'].parameters()},
                {'params': self.networks['mri_projector'].parameters()},
                {'params': self.networks['pet_projector'].parameters()},
                {'params': self.networks['common_projector'].parameters()},
                {'params': self.networks['predictor'].parameters()},
                {'params': self.networks['classifier'].parameters()}
            ],
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
        mri_pet_complete_train_sampler = ImbalancedDatasetSampler(dataset=datasets['mri_pet_complete_train'])
        mri_incomplete_train_sampler = ImbalancedDatasetSampler(dataset=datasets['mri_incomplete_train'])

        loaders = {
            'mri_pet_complete_train': DataLoader(dataset=datasets['mri_pet_complete_train'],
                                                 batch_size=self.batch_size,
                                                 sampler=mri_pet_complete_train_sampler,
                                                 drop_last=True),
            'mri_incomplete_train': DataLoader(dataset=datasets['mri_incomplete_train'],
                                               batch_size=self.batch_size,
                                               sampler=mri_incomplete_train_sampler,
                                               drop_last=True),
            'mri_pet_complete_test': DataLoader(dataset=datasets['mri_pet_complete_test'],
                                                batch_size=self.batch_size,
                                                drop_last=False)
        }

        # Logging
        logger = kwargs.get('logger', None)

        # Find the best model by MRI-only test loss
        best_eval_loss = float('inf')
        best_epoch = 0

        if self.enable_wandb:
            len_loader = len(loaders['mri_pet_complete_train'])
            log_freq = (len_loader * self.batch_size // (self.batch_size * self.accumulate)) * self.accumulate
            wandb.watch([v for k, v in self.networks.items()], log='all', log_freq=log_freq)

        for epoch in range(1, self.epochs + 1):

            # Train
            train_1_history = self.train_stage_1(loaders, adjusted=False)
            train_2_history = self.train_stage_2(loaders)
            train_3_history = self.train_stage_3(loaders, adjusted=False)

            # Test
            test_complete_history = self.evaluate_complete(loaders, adjusted=False)
            test_incomplete_history = self.evaluate_incomplete(loaders, adjusted=False)

            # Logging
            epoch_history = collections.defaultdict(dict)
            for phase, history in zip(['phase1', 'phase2', 'phase3'],
                                      [train_1_history, train_2_history, train_3_history]):
                for k, v in history.items():
                    epoch_history[f'train/{phase}/{k}'] = v

            for phase, history in zip(['complete', 'incomplete'],
                                      [test_complete_history, test_incomplete_history]):
                for k, v in history.items():
                    epoch_history[f'test/{phase}/{k}'] = v

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
            eval_loss = test_incomplete_history['cross_entropy']
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
        test_complete_history = self.evaluate_complete(loaders, adjusted=True)
        test_incomplete_history = self.evaluate_incomplete(loaders, adjusted=True)

        last_history = collections.defaultdict(dict)
        for phase, history in zip(['complete', 'incomplete'],
                                  [test_complete_history, test_incomplete_history]):
            for k, v in history.items():
                last_history[f'adjusted/{phase}/{k}'] = v

        if self.enable_wandb:
            wandb.log(last_history)

    # TODO: Define forward paths 1~3
    def train_stage_1(self, loaders, adjusted=False):

        # Train models for complete modality
        # Set network
        self._set_learning_phase(train=True)
        for name in self.network_names:
            if name in self.stage2network[1]:
                self.freeze_params(net=self.networks[name], freeze=False)
            else:
                self.freeze_params(net=self.networks[name], freeze=True)

        # Logging
        len_loader = len(loaders['mri_pet_complete_train'])
        steps = (len_loader * self.batch_size // (self.batch_size * self.accumulate)) * self.accumulate

        result = {'total_loss': torch.zeros(steps, device=self.local_rank),
                  'cos_mri_common': torch.zeros(steps, device=self.local_rank),
                  'cos_pet_common': torch.zeros(steps, device=self.local_rank),
                  'cos_mri_pet': torch.zeros(steps, device=self.local_rank),
                  'cross_entropy': torch.zeros(steps, device=self.local_rank)}

        with get_rich_pbar(transient=True, auto_refresh=False) as pg:

            if self.local_rank == 0:
                task = pg.add_task(f"[bold red] Training Stage 1...", total=steps)

            y_true, y_pred = [], []
            for i, batch in enumerate(loaders['mri_pet_complete_train']):
                with torch.cuda.amp.autocast(self.mixed_precision):

                    # input data
                    x_mri = batch['mri'].float().to(self.local_rank)
                    x_pet = batch['pet'].float().to(self.local_rank)
                    y = batch['y'].long().to(self.local_rank)

                    # hidden representations
                    h_mri = self.networks['mri_encoder'](x_mri)
                    h_pet = self.networks['pet_encoder'](x_pet)
                    if self.common_type == 'concat':
                        h_common = torch.concat([h_mri, h_pet], dim=1)
                    elif self.common_type == 'add':
                        h_common = h_mri + h_pet

                    # TODO: config.z_normalize
                    z_mri = self.networks['mri_projector'](h_mri)
                    z_pet = self.networks['pet_projector'](h_pet)
                    z_common = self.networks['common_projector'](h_common)

                    # cosine loss
                    cos_mri_common = torch.mean((1 + F.cosine_similarity(z_mri, z_common)) / 2)
                    cos_pet_common = torch.mean((1 + F.cosine_similarity(z_pet, z_common)) / 2)
                    cos_mri_pet = torch.mean((1 + F.cosine_similarity(z_mri, z_pet)) / 2)

                    # cross entropy loss
                    z_final = z_mri + z_pet + z_common
                    logits = self.networks['classifier'](z_final)
                    loss_ce = self.loss_function_ce(logits, y)

                    # final loss
                    loss = loss_ce + \
                           cos_mri_common * self.cosine_ratio + \
                           cos_pet_common * self.cosine_ratio + \
                           cos_mri_pet * self.cosine_ratio
                    loss = loss / (1 + 3 * self.cosine_ratio)
                    loss = loss / self.accumulate

                # Accumulate scaled gradients
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
                result['total_loss'][i] = loss.item() * self.accumulate
                result['cos_mri_common'][i] = cos_mri_common.item()
                result['cos_pet_common'][i] = cos_pet_common.item()
                result['cos_mri_pet'][i] = cos_mri_pet.item()
                result['cross_entropy'][i] = loss_ce

                if self.local_rank == 0:
                    desc = f"[bold green] Stage 1 [{i+1}/{steps}]: "
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

    def train_stage_2(self, loaders):

        # Train predictor
        self._set_learning_phase(train=True)
        for name in self.network_names:
            if name in self.stage2network[2]:
                self.freeze_params(net=self.networks[name], freeze=False)
            else:
                self.freeze_params(net=self.networks[name], freeze=True)

        len_loader = len(loaders['mri_pet_complete_train'])
        steps = (len_loader * self.batch_size // (self.batch_size * self.accumulate)) * self.accumulate
        result = {'cos_final_pred': torch.zeros(steps, device=self.local_rank)}

        with get_rich_pbar(transient=True, auto_refresh=False) as pg:

            if self.local_rank == 0:
                task = pg.add_task(f"[bold red] Training Stage 2...", total=steps)

            for i, batch in enumerate(loaders['mri_pet_complete_train']):
                with torch.cuda.amp.autocast(self.mixed_precision):

                    # input data
                    x_mri = batch['mri'].float().to(self.local_rank)
                    x_pet = batch['pet'].float().to(self.local_rank)

                    # 1. z_final
                    # hidden representations
                    h_mri = self.networks['mri_encoder'](x_mri)
                    h_pet = self.networks['pet_encoder'](x_pet)
                    if self.common_type == 'concat':
                        h_common = torch.concat([h_mri, h_pet], dim=1)
                    elif self.common_type == 'add':
                        h_common = h_mri + h_pet

                    z_mri = self.networks['mri_projector'](h_mri)
                    z_pet = self.networks['pet_projector'](h_mri)
                    z_common = self.networks['common_projector'](h_common)

                    z_final = z_mri + z_pet + z_common

                    # 2. z_pred
                    z_pred = self.networks['predictor'](h_mri)

                    # cosine loss: to maximize the cosine similarity between two representations
                    loss = torch.mean((1 - F.cosine_similarity(z_final, z_pred)) / 2)
                    loss = loss / self.accumulate

                # Accumulate scaled gradients
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
                result['cos_final_pred'][i] = loss.item() * self.accumulate

                if self.local_rank == 0:
                    desc = f"[bold green] Stage 2 [{i + 1}/{steps}]: "
                    for k, v in result.items():
                        desc += f" {k} : {v[:i + 1].mean():.4f} |"
                    pg.update(task, advance=1., description=desc)
                    pg.refresh()

                # Ignore the remained batches
                if (i + 1) == steps:
                    break

        result = {k: v.mean().item() for k, v in result.items()}

        return result

    def train_stage_3(self, loaders, adjusted=False):

        # Train MRI encoder
        self._set_learning_phase(train=True)
        for name in self.network_names:
            if name in self.stage2network[3]:
                self.freeze_params(net=self.networks[name], freeze=False)
            else:
                self.freeze_params(net=self.networks[name], freeze=True)

        # Logging
        len_loader = len(loaders['mri_incomplete_train'])
        steps = (len_loader * self.batch_size // (self.batch_size * self.accumulate)) * self.accumulate

        result = {'cross_entropy': torch.zeros(steps, device=self.local_rank)}

        with get_rich_pbar(transient=True, auto_refresh=False) as pg:

            if self.local_rank == 0:
                task = pg.add_task(f"[bold red] Training Stage 3...", total=steps)

            y_true, y_pred = [], []
            for i, batch in enumerate(loaders['mri_incomplete_train']):
                with torch.cuda.amp.autocast(self.mixed_precision):

                    # input data
                    x_mri = batch['mri'].float().to(self.local_rank)
                    y = batch['y'].long().to(self.local_rank)

                    # 1. z_pred
                    h_mri = self.networks['mri_encoder'](x_mri)
                    z_pred = self.networks['predictor'](h_mri)

                    # 2. classifier
                    logits = self.networks['classifier'](z_pred)
                    loss = self.loss_function_ce(logits, y)
                    loss = loss / self.accumulate

                # Accumulate scaled gradients
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
                result['cross_entropy'][i] = loss.item() * self.accumulate

                if self.local_rank == 0:
                    desc = f"[bold green] Stage 3 [{i + 1}/{steps}]: "
                    for k, v in result.items():
                        desc += f" {k} : {v[:i + 1].mean():.4f} |"
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
    def evaluate_complete(self, loaders, adjusted=False):

        self._set_learning_phase(train=False)
        steps = len(loaders['mri_pet_complete_test'])
        result = {'total_loss': torch.zeros(steps, device=self.local_rank),
                  'cos_mri_common': torch.zeros(steps, device=self.local_rank),
                  'cos_pet_common': torch.zeros(steps, device=self.local_rank),
                  'cos_mri_pet': torch.zeros(steps, device=self.local_rank),
                  'cross_entropy': torch.zeros(steps, device=self.local_rank)}

        with get_rich_pbar(transient=True, auto_refresh=False) as pg:

            y_true, y_pred = [], []
            for i, batch in enumerate(loaders['mri_pet_complete_test']):
                with torch.cuda.amp.autocast(self.mixed_precision):

                    # input data
                    x_mri = batch['mri'].float().to(self.local_rank)
                    x_pet = batch['pet'].float().to(self.local_rank)
                    y = batch['y'].long().to(self.local_rank)

                    # hidden representations
                    h_mri = self.networks['mri_encoder'](x_mri)
                    h_pet = self.networks['pet_encoder'](x_pet)
                    if self.common_type == 'concat':
                        h_common = torch.concat([h_mri, h_pet], dim=1)
                    elif self.common_type == 'add':
                        h_common = h_mri + h_pet

                    # TODO: config.z_normalize
                    z_mri = self.networks['mri_projector'](h_mri)
                    z_pet = self.networks['pet_projector'](h_pet)
                    z_common = self.networks['common_projector'](h_common)

                    # cosine loss
                    cos_mri_common = torch.mean((1 + F.cosine_similarity(z_mri, z_common)) / 2)
                    cos_pet_common = torch.mean((1 + F.cosine_similarity(z_pet, z_common)) / 2)
                    cos_mri_pet = torch.mean((1 + F.cosine_similarity(z_mri, z_pet)) / 2)

                    # cross entropy loss
                    z_final = z_mri + z_pet + z_common
                    logits = self.networks['classifier'](z_final)
                    loss_ce = self.loss_function_ce(logits, y)

                    # final loss
                    loss = loss_ce + \
                           cos_mri_common * self.cosine_ratio + \
                           cos_pet_common * self.cosine_ratio + \
                           cos_mri_pet * self.cosine_ratio
                    loss = loss / (1 + 3 * self.cosine_ratio)
                    loss = loss

                # save monitoring values
                result['total_loss'][i] = loss.item()
                result['cos_mri_common'][i] = cos_mri_common.item()
                result['cos_pet_common'][i] = cos_pet_common.item()
                result['cos_mri_pet'][i] = cos_mri_pet
                result['cross_entropy'][i] = loss_ce

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

    @torch.no_grad()
    def evaluate_incomplete(self, loaders, adjusted=False):

        self._set_learning_phase(train=False)
        steps = len(loaders['mri_pet_complete_test'])
        result = {'cross_entropy': torch.zeros(steps, device=self.local_rank)}

        with get_rich_pbar(transient=True, auto_refresh=False) as pg:

            y_true, y_pred = [], []
            for i, batch in enumerate(loaders['mri_pet_complete_test']):
                with torch.cuda.amp.autocast(self.mixed_precision):

                    # input data
                    x_mri = batch['mri'].float().to(self.local_rank)
                    y = batch['y'].long().to(self.local_rank)

                    # 1. z_pred
                    h_mri = self.networks['mri_encoder'](x_mri)
                    z_pred = self.networks['predictor'](h_mri)

                    # 2. classifier
                    logits = self.networks['classifier'](z_pred)
                    loss = self.loss_function_ce(logits, y)

                # save monitoring values
                result['cross_entropy'][i] = loss.item()

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
