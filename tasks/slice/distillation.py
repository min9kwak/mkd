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
from datasets.samplers import ImbalancedDatasetSampler


class Distillation(object):

    network_names = ['encoder_t_pet', 'encoder_t_mri', 'classifier_t',
                     'encoder_s_mri', 'classifier_s']

    def __init__(self,
                 networks: dict):

        self.networks = networks
        self.scaler = None
        self.optimizer_t = None
        self.optimizer_s = None
        self.scheduler_t = None
        self.scheduler_s = None
        self.prepared = False

    def prepare(self,
                config: object,
                loss_function_ce,
                local_rank: int = 0,
                **kwargs):

        # Set attributes
        self.config = config

        self.checkpoint_dir = config.checkpoint_dir
        self.loss_function_ce = loss_function_ce
        self.epochs = config.epochs
        self.batch_size = config.batch_size
        self.num_workers = config.num_workers
        self.distributed = config.distributed
        self.local_rank = local_rank
        assert config.add_type in ['concat', 'add']
        self.add_type = config.add_type
        self.warmup = config.warmup

        self.temperature = config.temperature
        assert config.feature_kd in ['cos', 'mse']
        self.feature_kd = config.feature_kd
        self.alpha_t2s = config.alpha_t2s
        self.alpha_s2t = config.alpha_s2t

        self.mixed_precision = config.mixed_precision
        self.enable_wandb = config.enable_wandb

        if self.config.train_slices == 'random':
            self.test_num_slices = 3
        elif self.config.train_slices == 'fixed':
            self.test_num_slices = 3
        elif self.config.train_slices in ['sagittal', 'coronal', 'axial']:
            self.test_num_slices = 1
        else:
            raise ValueError

        if self.distributed:
            raise NotImplementedError
        else:
            _ = [v.to(self.local_rank) for k, v in self.networks.items()]

        # Optimization setting
        self.optimizer_t = get_optimizer(params=[{'params': self.networks['encoder_t_pet'].parameters()},
                                                 {'params': self.networks['encoder_t_mri'].parameters()},
                                                 {'params': self.networks['classifier_t'].parameters()}],
                                         name=config.optimizer,
                                         lr=config.learning_rate,
                                         weight_decay=config.weight_decay)
        self.scheduler_t = get_cosine_scheduler(self.optimizer_t, epochs=self.epochs, warmup_steps=config.cosine_warmup,
                                                cycles=config.cosine_cycles, min_lr=config.cosine_min_lr)

        self.optimizer_s = get_optimizer(params=[{'params': self.networks['encoder_s_mri'].parameters()},
                                                 {'params': self.networks['classifier_s'].parameters()}],
                                         name=config.optimizer,
                                         lr=config.learning_rate,
                                         weight_decay=config.weight_decay)
        self.scheduler_s = get_cosine_scheduler(self.optimizer_s, epochs=self.epochs, warmup_steps=config.cosine_warmup,
                                                cycles=config.cosine_cycles, min_lr=config.cosine_min_lr)

        self.scaler = torch.cuda.amp.GradScaler() if config.mixed_precision else None

        # Ready to train
        self.prepared = True

    def run(self, datasets, save_every: int = 10, **kwargs):

        if not self.prepared:
            raise RuntimeError("Training not prepared.")

        # DataSet & DataLoader
        train_sampler_t = ImbalancedDatasetSampler(dataset=datasets['train_t'])
        train_sampler_s = ImbalancedDatasetSampler(dataset=datasets['train_s'])

        loaders = {
            'train_t': DataLoader(dataset=datasets['train_t'], batch_size=self.batch_size,
                                  sampler=train_sampler_t, drop_last=True),
            'train_s': DataLoader(dataset=datasets['train_s'], batch_size=self.batch_size,
                                  sampler=train_sampler_s, drop_last=True),
            'validation': DataLoader(dataset=datasets['validation'], batch_size=self.batch_size, drop_last=False),
            'test': DataLoader(dataset=datasets['test'], batch_size=self.batch_size, drop_last=False)
        }

        # Logging
        logger = kwargs.get('logger', None)

        # Find the best model by MRI-only test loss
        best_eval_loss = float('inf')
        best_epoch = 0

        for epoch in range(1, self.epochs + 1):

            self.epoch = epoch

            # Train and Test
            epoch_history = collections.defaultdict(dict)

            if epoch <= self.warmup:
                train_history_t = self.train_teacher(loaders['train_t'], adjusted=False)
                train_history_s = self.train_student(loaders['train_s'], adjusted=False)
                for mode, history in zip(['train-teacher', 'train-student'], [train_history_t, train_history_s]):
                    for k, v in history.items():
                        epoch_history[f'{mode}/{k}'] = v

            else:
                train_history_t = self.train_teacher(loaders['train_t'], adjusted=False)
                train_history_kd_t2s = self.train_kd_teacher_to_student(loaders['train_t'])
                train_history_s = self.train_student(loaders['train_s'], adjusted=False)
                train_history_kd_s2t = self.train_kd_student_to_teacher(loaders['train_s'])
                for mode, history in zip(
                        ['train-teacher', 'train-student', 'train-t2s', 'train-s2t'],
                        [train_history_t, train_history_s, train_history_kd_t2s, train_history_kd_s2t]
                ):
                    for k, v in history.items():
                        epoch_history[f'{mode}/{k}'] = v
            validation_history = self.evaluate(loaders['validation'], adjusted=False)
            test_history = self.evaluate(loaders['test'], adjusted=False)

            # Logging
            for mode, history in zip(['validation', 'test'], [validation_history, test_history]):
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
                if self.scheduler_t is not None:
                    wandb.log({'lr_t': self.scheduler_t.get_last_lr()[0]}, commit=False)
                    wandb.log({'lr_s': self.scheduler_s.get_last_lr()[0]}, commit=False)
                else:
                    wandb.log({'lr_t': self.optimizer_t.param_groups[0]['lr']}, commit=False)
                    wandb.log({'lr_s': self.optimizer_s.param_groups[0]['lr']}, commit=False)
                wandb.log(epoch_history)

            # Save best model checkpoint
            eval_loss = test_history['cross_entropy_t']
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
            if self.scheduler_t is not None:
                self.scheduler_t.step()
            if self.scheduler_s is not None:
                self.scheduler_s.step()

        # Save final model checkpoint
        ckpt = os.path.join(self.checkpoint_dir, f"ckpt.last.pth.tar")
        self.save_checkpoint(ckpt, epoch=epoch)

        # adjusted evaluation (last)
        validation_history = self.evaluate(loaders['validation'], adjusted=True)
        test_history = self.evaluate(loaders['test'], adjusted=True)

        last_history = collections.defaultdict(dict)
        for k, v in validation_history.items():
            last_history[f'adjusted-last/validation/{k}'] = v
        for k, v in test_history.items():
            last_history[f'adjusted-last/test/{k}'] = v
        if self.enable_wandb:
            wandb.log(last_history)

        # adjusted evaluation (best)
        ckpt = os.path.join(self.checkpoint_dir, f"ckpt.best.pth.tar")
        for k, v in self.networks.items():
            v.load_weights_from_checkpoint(path=ckpt, key=k)

        validation_history = self.evaluate(loaders['validation'], adjusted=True)
        test_history = self.evaluate(loaders['test'], adjusted=True)

        best_history = collections.defaultdict(dict)
        for k, v in validation_history.items():
            best_history[f'adjusted-best/validation/{k}'] = v
        for k, v in test_history.items():
            best_history[f'adjusted-best/test/{k}'] = v
        if self.enable_wandb:
            wandb.log(best_history)

    def train_teacher(self, data_loader_t, adjusted=False):

        self._set_learning_phase(network_type='teacher', train=True)

        steps = len(data_loader_t)
        result = {'cross_entropy_t': torch.zeros(steps, device=self.local_rank)}

        with get_rich_pbar(transient=True, auto_refresh=False) as pg:

            if self.local_rank == 0:
                task = pg.add_task(f"[bold red] Training Teacher...", total=steps)

            y_true, y_pred = [], []
            for i, batch in enumerate(data_loader_t):
                with torch.cuda.amp.autocast(self.mixed_precision):
                    # input data
                    x_pet = torch.concat(batch['pet']).float().to(self.local_rank)
                    x_mri = torch.concat(batch['mri']).float().to(self.local_rank)
                    y = batch['y'].long().repeat(self.config.num_slices).to(self.local_rank)

                    # hidden representations
                    h_pet = self.networks['encoder_t_pet'](x_pet)
                    h_mri = self.networks['encoder_t_mri'](x_mri)
                    if self.add_type == 'concat':
                        h_common = torch.concat([h_pet, h_mri], dim=1)
                    else:
                        h_common = h_pet + h_mri
                    logits = self.networks['classifier_t'](h_common)
                    loss_ce = self.loss_function_ce(logits, y)

                if self.scaler is not None:
                    self.scaler.scale(loss_ce).backward()
                    self.scaler.step(self.optimizer_t)
                    self.scaler.update()
                else:
                    loss_ce.backward()
                    self.optimizer_t.step()
                self.optimizer_t.zero_grad()

                # save monitoring values
                result['cross_entropy_t'][i] = loss_ce.detach()

                if self.local_rank == 0:
                    desc = f"[bold green] Epoch {self.epoch} [{i+1}/{steps}]: "
                    for k, v in result.items():
                        desc += f" {k} : {v[:i+1].mean():.4f} |"
                    pg.update(task, advance=1., description=desc)
                    pg.refresh()

                y_true.append(y.chunk(self.config.num_slices)[0].long())
                num_classes = logits.shape[-1]
                logits = logits.reshape(self.config.num_slices, -1, num_classes).mean(0)
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

    def train_student(self, data_loader_s, adjusted=False):

        self._set_learning_phase(network_type='student', train=True)

        steps = len(data_loader_s)
        result = {'cross_entropy_s': torch.zeros(steps, device=self.local_rank)}

        with get_rich_pbar(transient=True, auto_refresh=False) as pg:

            if self.local_rank == 0:
                task = pg.add_task(f"[bold red] Training Student...", total=steps)

            y_true, y_pred = [], []
            for i, batch in enumerate(data_loader_s):
                with torch.cuda.amp.autocast(self.mixed_precision):
                    # input data
                    x_mri = torch.concat(batch['mri']).float().to(self.local_rank)
                    y = batch['y'].long().repeat(self.config.num_slices).to(self.local_rank)

                    # hidden representations
                    h = self.networks['encoder_s_mri'](x_mri)
                    logits = self.networks['classifier_s'](h)
                    loss_ce = self.loss_function_ce(logits, y)

                if self.scaler is not None:
                    self.scaler.scale(loss_ce).backward()
                    self.scaler.step(self.optimizer_s)
                    self.scaler.update()
                else:
                    loss_ce.backward()
                    self.optimizer_s.step()
                self.optimizer_s.zero_grad()

                # save monitoring values
                result['cross_entropy_s'][i] = loss_ce.detach()

                if self.local_rank == 0:
                    desc = f"[bold green] Epoch {self.epoch} [{i+1}/{steps}]: "
                    for k, v in result.items():
                        desc += f" {k} : {v[:i+1].mean():.4f} |"
                    pg.update(task, advance=1., description=desc)
                    pg.refresh()

                y_true.append(y.chunk(self.config.num_slices)[0].long())
                num_classes = logits.shape[-1]
                logits = logits.reshape(self.config.num_slices, -1, num_classes).mean(0)
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

    def train_kd_teacher_to_student(self, data_loader_t):

        self._set_learning_phase(network_type='teacher', train=False)
        self._set_learning_phase(network_type='student', train=True)

        steps = len(data_loader_t)
        result = {'kd_t2s': torch.zeros(steps, device=self.local_rank)}

        with get_rich_pbar(transient=True, auto_refresh=False) as pg:

            if self.local_rank == 0:
                task = pg.add_task(f"[bold red] KD Teacher -> Student...", total=steps)

            for i, batch in enumerate(data_loader_t):
                with torch.cuda.amp.autocast(self.mixed_precision):

                    # input data
                    x_pet = torch.concat(batch['pet']).float().to(self.local_rank)
                    x_mri = torch.concat(batch['mri']).float().to(self.local_rank)

                    # Teacher Logits
                    with torch.no_grad():
                        # hidden representations
                        h_t_pet = self.networks['encoder_t_pet'](x_pet)
                        h_t_mri = self.networks['encoder_t_mri'](x_mri)
                        if self.add_type == 'concat':
                            h_common = torch.concat([h_t_pet, h_t_mri], dim=1)
                        else:
                            h_common = h_t_pet + h_t_mri
                        logits_t = self.networks['classifier_t'](h_common)

                    # Student Logits
                    h_s_mri = self.networks['encoder_s_mri'](x_mri)
                    logits_s = self.networks['classifier_s'](h_s_mri)

                    # KL-div
                    kd_loss = F.kl_div(F.log_softmax(logits_s / self.temperature, dim=1),
                                       F.softmax(logits_t / self.temperature, dim=1),
                                       reduction='batchmean')
                    kd_loss = kd_loss * self.alpha_t2s

                if self.scaler is not None:
                    self.scaler.scale(kd_loss).backward()
                    self.scaler.step(self.optimizer_s)
                    self.scaler.update()
                else:
                    kd_loss.backward()
                    self.optimizer_s.step()
                self.optimizer_s.zero_grad()

                # save monitoring values
                result['kd_t2s'][i] = kd_loss.detach() / self.alpha_t2s

                if self.local_rank == 0:
                    desc = f"[bold green] Epoch {self.epoch} [{i + 1}/{steps}]: "
                    for k, v in result.items():
                        desc += f" {k} : {v[:i + 1].mean():.4f} |"
                    pg.update(task, advance=1., description=desc)
                    pg.refresh()

        result = {k: v.mean().item() for k, v in result.items()}

        return result

    def train_kd_student_to_teacher(self, data_loader_s):

        self._set_learning_phase(network_type='teacher', train=True)
        self._set_learning_phase(network_type='student', train=False)

        steps = len(data_loader_s)
        result = {'kd_s2t': torch.zeros(steps, device=self.local_rank)}

        with get_rich_pbar(transient=True, auto_refresh=False) as pg:

            if self.local_rank == 0:
                task = pg.add_task(f"[bold red] KD Student -> Teacher...", total=steps)

            for i, batch in enumerate(data_loader_s):
                with torch.cuda.amp.autocast(self.mixed_precision):

                    # input data
                    x_mri = torch.concat(batch['mri']).float().to(self.local_rank)

                    # Student Features
                    with torch.no_grad():
                        h_s_mri = self.networks['encoder_s_mri'](x_mri)
                        h_s_mri = F.adaptive_avg_pool2d(h_s_mri, 1)
                        h_s_mri = h_s_mri.view(h_s_mri.size(0), -1)

                    # Teacher Features
                    h_t_mri = self.networks['encoder_t_mri'](x_mri)
                    h_t_mri = F.adaptive_avg_pool2d(h_t_mri, 1)
                    h_t_mri = h_t_mri.view(h_t_mri.size(0), -1)

                    # MSE Loss
                    if self.feature_kd == 'mse':
                        kd_loss = F.mse_loss(h_t_mri, h_s_mri, reduction="mean")
                    elif self.feature_kd == 'cos':
                        # Cosine Loss
                        h_s_mri = F.normalize(h_s_mri, p=2, dim=1)
                        h_t_mri = F.normalize(h_t_mri, p=2, dim=1)

                        dist = F.cosine_similarity(h_s_mri, h_t_mri, dim=1)
                        kd_loss = (1 - dist) / (2 * self.temperature ** 2)
                        kd_loss = kd_loss.mean()

                    kd_loss = kd_loss * self.alpha_s2t

                if self.scaler is not None:
                    self.scaler.scale(kd_loss).backward()
                    self.scaler.step(self.optimizer_t)
                    self.scaler.update()
                else:
                    kd_loss.backward()
                    self.optimizer_t.step()
                self.optimizer_t.zero_grad()

                # save monitoring values
                result['kd_s2t'][i] = kd_loss.detach() / self.alpha_s2t

                if self.local_rank == 0:
                    desc = f"[bold green] Epoch {self.epoch} [{i + 1}/{steps}]: "
                    for k, v in result.items():
                        desc += f" {k} : {v[:i + 1].mean():.4f} |"
                    pg.update(task, advance=1., description=desc)
                    pg.refresh()

        result = {k: v.mean().item() for k, v in result.items()}

        return result

    @torch.no_grad()
    def evaluate(self, data_loder, adjusted=False):

        self._set_learning_phase(network_type='teacher', train=False)
        self._set_learning_phase(network_type='student', train=False)

        steps = len(data_loder)
        result = {'cross_entropy_t': torch.zeros(steps, device=self.local_rank)}

        # 1. Training Teacher Model
        with get_rich_pbar(transient=True, auto_refresh=False) as pg:
            if self.local_rank == 0:
                task = pg.add_task(f"[bold red] Evaluating...", total=steps)
            y_true, y_pred = [], []
            for i, batch in enumerate(data_loder):
                with torch.cuda.amp.autocast(self.mixed_precision):
                    # input data
                    x_pet = torch.concat(batch['pet']).float().to(self.local_rank)
                    x_mri = torch.concat(batch['mri']).float().to(self.local_rank)
                    y = batch['y'].long().repeat(self.test_num_slices).to(self.local_rank)

                    # hidden representations
                    h_pet = self.networks['encoder_t_pet'](x_pet)
                    h_mri = self.networks['encoder_t_mri'](x_mri)
                    if self.add_type == 'concat':
                        h_common = torch.concat([h_pet, h_mri], dim=1)
                    else:
                        h_common = h_pet + h_mri
                    logits = self.networks['classifier_t'](h_common)
                    loss_ce = self.loss_function_ce(logits, y)

                # save monitoring values
                result['cross_entropy_t'][i] = loss_ce.detach()

                if self.local_rank == 0:
                    pg.update(task, advance=1.)
                    pg.refresh()

                y_true.append(y.chunk(self.test_num_slices)[0].long())
                num_classes = logits.shape[-1]
                logits = logits.reshape(self.test_num_slices, -1, num_classes).mean(0)
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

    def _set_learning_phase(self, network_type: str, train: bool = True):
        assert network_type in ['teacher', 'student']
        if network_type == 'teacher':
            suffix = '_t'
        else:
            suffix = '_s'
        for name, network in self.networks.items():
            if suffix in name:
                if train:
                    network.train()
                else:
                    network.eval()

    def save_checkpoint(self, path: str, **kwargs):
        ckpt = {k: v.state_dict() for k, v in self.networks.items()}
        if kwargs:
            ckpt.update(kwargs)
        torch.save(ckpt, path)
