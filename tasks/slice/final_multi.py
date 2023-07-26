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


class FinalMulti(object):

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
                loss_function_sim,
                loss_function_diff,
                loss_function_recon,
                swap: bool,
                local_rank: int = 0,
                **kwargs):

        # Set attributes
        self.config = config

        # CE / CMD / DIFF / MSE / KD (repr)
        self.loss_function_ce = loss_function_ce
        self.loss_function_sim = loss_function_sim
        self.loss_function_diff = loss_function_diff
        self.loss_function_recon = loss_function_recon

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

        # Student model is fixed to eval mode
        for name in self.networks.keys():
            if name.endswith('_s'):
                self.networks[name].eval()

        # Optimization setting
        params = []
        for name in self.networks.keys():
            if not name.endswith('_s'):
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

        # Freeze & Eval Student Model
        for name in self.networks.keys():
            if name.endswith('_s'):
                self.freeze_params(self.networks[name], freeze=True)
        self._set_learning_phase(train=False)

        # Ready to train
        self.prepared = True

    def run(self, datasets, save_every: int = 20, **kwargs):

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

        # Find the best model by total loss
        best_eval_loss = float('inf')
        best_epoch = 0

        for epoch in range(1, self.epochs + 1):

            self.epoch = epoch

            # Train and Test
            epoch_history = collections.defaultdict(dict)
            train_history = self.train(loaders['train'], adjusted=False)
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

    def train(self, data_loader, adjusted=False):

        self._set_learning_phase(train=True)

        steps = len(data_loader)
        result = {'total_loss': torch.zeros(steps, device=self.local_rank),
                  'loss_ce': torch.zeros(steps, device=self.local_rank),
                  'loss_sim': torch.zeros(steps, device=self.local_rank),
                  'loss_diff_specific': torch.zeros(steps, device=self.local_rank),
                  'loss_diff_mri': torch.zeros(steps, device=self.local_rank),
                  'loss_diff_pet': torch.zeros(steps, device=self.local_rank),
                  'loss_recon_mri': torch.zeros(steps, device=self.local_rank),
                  'loss_recon_pet': torch.zeros(steps, device=self.local_rank),
                  'loss_kd_repr': torch.zeros(steps, device=self.local_rank)}

        with get_rich_pbar(transient=True, auto_refresh=False) as pg:

            if self.local_rank == 0:
                task = pg.add_task(f"[bold red] Training...", total=steps)

            y_true, y_pred = [], []
            for i, batch in enumerate(data_loader):
                with torch.cuda.amp.autocast(self.mixed_precision):
                    loss, loss_ce, loss_sim, loss_diff_specific, loss_diff_mri, loss_diff_pet, \
                    loss_recon_mri, loss_recon_pet, loss_kd_repr, y, logit = \
                        self.train_step(batch)
                self.update(loss)

                # save monitoring values
                result['total_loss'][i] = loss.detach()
                result['loss_ce'][i] = loss_ce.detach()
                result['loss_sim'][i] = loss_sim.detach()
                result['loss_diff_specific'][i] = loss_diff_specific.detach()
                result['loss_diff_mri'][i] = loss_diff_mri.detach()
                result['loss_diff_pet'][i] = loss_diff_pet.detach()
                result['loss_recon_mri'][i] = loss_recon_mri.detach()
                result['loss_recon_pet'][i] = loss_recon_pet.detach()
                result['loss_kd_repr'][i] = loss_kd_repr.detach()

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

                num_classes = logit.shape[-1]
                logit = logit[labeled_index]
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

        # TODO: utilize large dataset for feature-level KD

        # input data
        x_mri = torch.concat(batch['mri']).float().to(self.local_rank)
        x_pet = torch.concat(batch['pet']).float().to(self.local_rank)
        y = batch['y'].long().repeat(self.config.num_slices).to(self.local_rank)

        # 1. Student
        with torch.no_grad():
            # hidden representations - h
            h_mri_s = self.networks['projector_mri_s'](self.networks['extractor_mri_s'](x_mri))

        # 2. Final Multi
        # hidden representations - h
        h_mri = self.networks['projector_mri'](self.networks['extractor_mri'](x_mri))
        h_pet = self.networks['projector_pet'](self.networks['extractor_pet'](x_pet))

        # separated representations - z
        z_mri_general = self.networks['encoder_general'](h_mri)
        z_pet_general = self.networks['encoder_general'](h_pet)
        z_mri = self.networks['encoder_mri'](h_mri)
        z_pet = self.networks['encoder_pet'](h_pet)

        # difference
        loss_diff_specific = self.loss_function_diff(z_mri, z_pet)
        loss_diff_mri = self.loss_function_diff(z_mri, z_mri_general)
        loss_diff_pet = self.loss_function_diff(z_pet, z_pet_general)
        loss_diff = loss_diff_specific + loss_diff_mri + loss_diff_pet

        # similarity
        loss_sim = self.loss_function_sim(z_mri_general, z_pet_general)

        # reconstruction - h
        h_mri_recon = self.networks['decoder_mri'](z_mri_general + z_mri)
        h_pet_recon = self.networks['decoder_pet'](z_pet_general + z_pet)

        loss_recon_mri = self.loss_function_recon(h_mri_recon, h_mri)
        loss_recon_pet = self.loss_function_recon(h_pet_recon, h_pet)
        loss_recon = loss_recon_mri + loss_recon_pet

        # classification
        if self.config.use_specific_final:
            logit = self.networks['classifier'](z_mri_general + z_pet_general + z_mri + z_pet)
        else:
            logit = self.networks['classifier'](z_mri_general + z_pet_general)

        loss_ce = self.loss_function_ce(logit, y)
        loss_ce = loss_ce / ((y != -1).sum() + 1e-6)

        # knowledge distillation
        h_mri_s_norm = F.normalize(h_mri_s, p=2, dim=1)
        h_mri_norm = F.normalize(h_mri, p=2, dim=1)

        cos = torch.einsum('nc,nc->n', [h_mri_s_norm, h_mri_norm])
        loss_kd_repr = (1 - cos) / (2 * self.config.temperature ** 2)
        loss_kd_repr = loss_kd_repr.mean()

        # 3. Loss Aggregation
        loss = self.config.alpha_ce * loss_ce + \
               self.config.alpha_sim * loss_sim + \
               self.config.alpha_diff * loss_diff + \
               self.config.alpha_recon * loss_recon + \
               self.config.alpha_kd_repr * loss_kd_repr

        return loss, loss_ce, loss_sim, loss_diff_specific, loss_diff_mri, loss_diff_pet, \
               loss_recon_mri, loss_recon_pet, loss_kd_repr, y, logit

    @torch.no_grad()
    def evaluate(self, data_loader, adjusted=False):

        self._set_learning_phase(train=False)

        steps = len(data_loader)
        result = {'total_loss': torch.zeros(steps, device=self.local_rank),
                  'loss_ce': torch.zeros(steps, device=self.local_rank),
                  'loss_sim': torch.zeros(steps, device=self.local_rank),
                  'loss_diff_specific': torch.zeros(steps, device=self.local_rank),
                  'loss_diff_mri': torch.zeros(steps, device=self.local_rank),
                  'loss_diff_pet': torch.zeros(steps, device=self.local_rank),
                  'loss_recon_mri': torch.zeros(steps, device=self.local_rank),
                  'loss_recon_pet': torch.zeros(steps, device=self.local_rank),
                  'loss_kd_repr': torch.zeros(steps, device=self.local_rank),}

        # 1. Training Teacher Model
        with get_rich_pbar(transient=True, auto_refresh=False) as pg:
            if self.local_rank == 0:
                task = pg.add_task(f"[bold red] Evaluating...", total=steps)

            y_true, y_pred = [], []
            for i, batch in enumerate(data_loader):
                with torch.cuda.amp.autocast(False):
                    loss, loss_ce, loss_sim, loss_diff_specific, loss_diff_mri, loss_diff_pet, \
                    loss_recon_mri, loss_recon_pet, loss_kd_repr, y, logit = \
                        self.train_step(batch)

                # save monitoring values
                result['total_loss'][i] = loss.detach()
                result['loss_ce'][i] = loss_ce.detach()
                result['loss_sim'][i] = loss_sim.detach()
                result['loss_diff_specific'][i] = loss_diff_specific.detach()
                result['loss_diff_mri'][i] = loss_diff_mri.detach()
                result['loss_diff_pet'][i] = loss_diff_pet.detach()
                result['loss_recon_mri'][i] = loss_recon_mri.detach()
                result['loss_recon_pet'][i] = loss_recon_pet.detach()
                result['loss_kd_repr'][i] = loss_kd_repr.detach()

                if self.local_rank == 0:
                    pg.update(task, advance=1.)
                    pg.refresh()

                y_true.append(y.chunk(self.test_num_slices)[0].long())
                num_classes = logit.shape[-1]
                logit = logit.reshape(self.test_num_slices, -1, num_classes).mean(0)
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
            if not name.endswith('_s'):
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
