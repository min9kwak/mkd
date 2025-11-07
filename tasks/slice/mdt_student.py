import os
import collections
import wandb
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from utils.metrics import classification_result
from utils.logging import make_epoch_description, get_rich_pbar
from utils.optimization import get_optimizer
from utils.optimization import get_cosine_scheduler
from datasets.samplers import ImbalancedDatasetSampler, StratifiedSampler


class MDTStudent(object):
    """MDT-Student: Knowledge Distillation to MRI-only Model.
    
    Trains a student model using only MRI scans by distilling knowledge from 
    the pre-trained MDT teacher. The student learns to mimic the teacher's 
    common representations, enabling AD diagnosis without PET scans.
    
    Key advantage: Trained with more samples (both complete and MRI-only cases).
    """

    def __init__(self,
                 networks: dict):

        self.networks = networks
        self.scaler = None
        self.optimizer = None
        self.scheduler = None
        self.prepared = False

        self.networks = {k: v for k, v in self.networks.items() if v is not None}
        self.network_names = list(self.networks.keys())

    def prepare(self,
                config: argparse.Namespace,
                loss_function_ce,
                local_rank: int = 0,
                **kwargs):
        """Prepare model for knowledge distillation from frozen teacher to student."""

        # Store configuration and training settings
        self.config = config

        # Store loss function: classification and knowledge distillation
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

        # Teacher model is fixed to eval mode
        for name in self.networks.keys():
            self.networks[name].eval()

        # Optimization setting
        params = []
        for name in self.networks.keys():
            if name.endswith('_s'):
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

        # Freeze & Eval Teacher Model
        for name in self.networks.keys():
            if not name.endswith('_s'):
                self.freeze_params(self.networks[name], freeze=True)
        self._set_learning_phase(train=False)

        # Ready to train
        self.prepared = True

    def run(self, datasets, save_every: int = 20, **kwargs):

        if not self.prepared:
            raise RuntimeError("Training not prepared.")

        # DataSet & DataLoader
        train_sampler, train_mri_sampler = None, None
        if self.config.sampler_type == 'over':
            train_sampler = ImbalancedDatasetSampler(dataset=datasets['train'])
            train_mri_sampler = ImbalancedDatasetSampler(dataset=datasets['train_mri'])
        elif self.config.sampler_type == 'stratified':
            train_sampler = StratifiedSampler(class_vector=datasets['train'].y, batch_size=self.batch_size)
            train_mri_sampler = StratifiedSampler(class_vector=datasets['train_mri'].y, batch_size=self.batch_size)

        if train_sampler is not None:
            loaders = {
                'train': DataLoader(dataset=datasets['train'], batch_size=self.batch_size,
                                    sampler=train_sampler, drop_last=True),
                'train_mri': DataLoader(dataset=datasets['train_mri'], batch_size=self.batch_size,
                                        sampler=train_mri_sampler, drop_last=True),
                'validation': DataLoader(dataset=datasets['validation'], batch_size=self.batch_size, drop_last=False),
                'test': DataLoader(dataset=datasets['test'], batch_size=self.batch_size, drop_last=False)
            }
        else:
            loaders = {
                'train': DataLoader(dataset=datasets['train'], batch_size=self.batch_size, shuffle=True,
                                    sampler=train_sampler, drop_last=True),
                'train_mri': DataLoader(dataset=datasets['train_mri'], batch_size=self.batch_size, shuffle=True,
                                        sampler=train_mri_sampler, drop_last=True),
                'validation': DataLoader(dataset=datasets['validation'], batch_size=self.batch_size, drop_last=False),
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
            train_history = self.train(data_loader=loaders['train'], data_mri_loader=loaders['train_mri'],
                                       adjusted=False)
            validation_history = self.evaluate(loaders['validation'], adjusted=False)
            test_history = self.evaluate(loaders['test'], adjusted=False)

            # Logging
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
            eval_loss = validation_history['total_loss']
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

    def train(self, data_loader, data_mri_loader, adjusted=False):

        self._set_learning_phase(train=True)

        steps = min(len(data_loader), len(data_mri_loader))
        result = {'total_loss': torch.zeros(steps, device=self.local_rank),
                  'loss_ce': torch.zeros(steps, device=self.local_rank),
                  'loss_kd_repr': torch.zeros(steps, device=self.local_rank),
                  'loss_kd_clf': torch.zeros(steps, device=self.local_rank),}

        with get_rich_pbar(transient=True, auto_refresh=False) as pg:

            if self.local_rank == 0:
                task = pg.add_task(f"[bold red] Training...", total=steps)

            y_true, y_pred = [], []
            for i, (batch, batch_mri) in enumerate(zip(data_loader, data_mri_loader)):
                with torch.cuda.amp.autocast(self.mixed_precision):
                    loss, loss_ce, loss_kd_repr, loss_kd_clf, y, logit_s = self.train_step(batch, batch_mri)
                self.update(loss)

                # save monitoring values
                result['total_loss'][i] = loss.detach()
                result['loss_ce'][i] = loss_ce.detach()
                result['loss_kd_repr'][i] = loss_kd_repr.detach()
                result['loss_kd_clf'][i] = loss_kd_clf.detach()

                if self.local_rank == 0:
                    desc = f"[bold green] Epoch {self.epoch} [{i + 1}/{steps}]: "
                    for k, v in result.items():
                        desc += f" {k} : {v[:i + 1].mean():.4f} |"
                    pg.update(task, advance=1., description=desc)
                    pg.refresh()

                # Save only labeled samples
                labeled_index = (y != -1)
                y = y[labeled_index].chunk(self.config.num_slices)[0].long()
                y_true.append(y)

                num_classes = logit_s.shape[-1]
                logit_s = logit_s[labeled_index]
                logit_s = logit_s.reshape(self.config.num_slices, -1, num_classes).mean(0)
                y_pred.append(logit_s)

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

    def train_step(self, batch, batch_mri):

        # A. Complete Training. Some of them are unlabeled.
        # input data
        x_mri = torch.concat(batch['mri']).float().to(self.local_rank)
        x_pet = torch.concat(batch['pet']).float().to(self.local_rank)
        y = batch['y'].long().repeat(self.config.num_slices).to(self.local_rank)

        # 1. Teacher
        with torch.no_grad():
            # hidden representations - h
            h_mri = self.networks['projector_mri'](self.networks['extractor_mri'](x_mri))
            h_pet = self.networks['projector_pet'](self.networks['extractor_pet'](x_pet))

            # separated representations - z
            z_mri_general = self.networks['encoder_general'](h_mri)
            z_pet_general = self.networks['encoder_general'](h_pet)

            if self.config.use_specific_t:
                z_mri = self.networks['encoder_mri'](h_mri)
                z = z_mri_general + z_pet_general + z_mri
            else:
                z = z_mri_general + z_pet_general

            # classification
            logit = self.networks['classifier'](z)

        # 2. Student
        h_mri_s = self.networks['projector_mri_s'](self.networks['extractor_mri_s'](x_mri))
        z_mri_general_s = self.networks['encoder_general_s'](h_mri_s)
        if self.config.use_specific:
            z_mri_s = self.networks['encoder_mri_s'](h_mri_s)
            logit_s = self.networks['classifier_s'](z_mri_general_s * 2 + z_mri_s)
        else:
            logit_s = self.networks['classifier_s'](z_mri_general_s * 2)

        # 4. Knowledge Distillation
        # general representation
        cos_general = torch.einsum('nc,nc->n', [z_mri_general, z_mri_general_s])
        loss_kd_repr_general = (1 - cos_general) / (2 * self.config.temperature ** 2)
        loss_kd_repr_general = loss_kd_repr_general.mean()

        # specific representation
        if self.config.use_specific and self.config.use_specific_t:
            cos_mri = torch.einsum('nc,nc->n', [z_mri, z_mri_s])
            loss_kd_repr_mri = (1 - cos_mri) / (2 * self.config.temperature ** 2)
            loss_kd_repr_mri = loss_kd_repr_mri.mean()
            loss_kd_repr = (loss_kd_repr_general + loss_kd_repr_mri) / 2
        else:
            loss_kd_repr = loss_kd_repr_general

        # classification
        loss_kd_clf = F.kl_div(F.log_softmax(logit_s / self.config.temperature, dim=1),
                               F.softmax(logit / self.config.temperature, dim=1),
                               reduction='none')
        loss_kd_clf = (loss_kd_clf[y != -1]).sum() / ((y != -1).sum() + 1e-6)

        # B. Incomplete Training. Some of them are unlabeled.
        # input data
        x_mri_in = torch.concat(batch_mri['mri']).float().to(self.local_rank)
        y_in = batch_mri['y'].long().repeat(self.config.num_slices).to(self.local_rank)

        # 1. Student
        h_mri_in = self.networks['projector_mri_s'](self.networks['extractor_mri_s'](x_mri_in))
        z_mri_general_in = self.networks['encoder_general_s'](h_mri_in)

        if self.config.use_specific:
            z_mri_in = self.networks['encoder_mri_s'](h_mri_in)
            z = z_mri_general_in * 2 + z_mri_in
        else:
            z = z_mri_general_in * 2
        logit_in = self.networks['classifier_s'](z)

        # C. Loss Aggregation
        logit_total = torch.concat([logit_s, logit_in])
        y_total = torch.concat([y, y_in])

        loss_ce = self.loss_function_ce(logit_total, y_total)
        loss_ce = loss_ce / ((y != -1).sum() + (y_in != -1).sum() + 1e-6)

        loss = self.config.alpha_ce * loss_ce + \
               self.config.alpha_kd_repr * loss_kd_repr + \
               self.config.alpha_kd_clf * loss_kd_clf

        return loss, loss_ce, loss_kd_repr, loss_kd_clf, y_total, logit_total

    @torch.no_grad()
    def evaluate(self, data_loader, adjusted=False):

        self._set_learning_phase(train=False)

        steps = len(data_loader)
        result = {'total_loss': torch.zeros(steps, device=self.local_rank),
                  'loss_ce': torch.zeros(steps, device=self.local_rank),
                  'loss_kd_repr': torch.zeros(steps, device=self.local_rank),
                  'loss_kd_clf': torch.zeros(steps, device=self.local_rank),}

        # 1. Training Teacher Model
        with get_rich_pbar(transient=True, auto_refresh=False) as pg:
            if self.local_rank == 0:
                task = pg.add_task(f"[bold red] Evaluating...", total=steps)

            y_true, y_pred = [], []
            for i, batch in enumerate(data_loader):
                with torch.cuda.amp.autocast(False):
                    loss, loss_ce, loss_kd_repr, loss_kd_clf, y, logit_s = self.train_step(batch, batch)

                # save monitoring values
                result['total_loss'][i] = loss.detach()
                result['loss_ce'][i] = loss_ce.detach()
                result['loss_kd_repr'][i] = loss_kd_repr.detach()
                result['loss_kd_clf'][i] = loss_kd_clf.detach()

                if self.local_rank == 0:
                    pg.update(task, advance=1.)
                    pg.refresh()

                y_true.append(y.chunk(self.test_num_slices)[0].long())
                num_classes = logit_s.shape[-1]
                logit_s = logit_s.reshape(self.config.num_slices, -1, num_classes).mean(0)
                y_pred.append(logit_s)

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
            for k, v in state.items():  # iterate over paramteters (k=name, v=tensor)
                if torch.is_tensor(v):  # If a tensor,
                    state[k] = v.to(device)  # configure appropriate device

    @staticmethod
    def freeze_params(net: nn.Module, freeze: bool):
        for p in net.parameters():
            p.requires_grad = not freeze

    def _set_learning_phase(self, train: bool = True):
        # Teacher is fixed to eval mode
        for name in self.networks.keys():
            if name.endswith('_s'):
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
