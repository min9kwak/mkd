import argparse
import collections
import copy
import os
import pickle
import tqdm
import wandb

from itertools import cycle

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from utils.metrics import classification_result
from utils.simulation import build_networks, build_short_networks, build_simple_networks
from utils.logging import get_rich_pbar
from utils.optimization import get_optimizer, get_cosine_scheduler


class Simulator:

    def __init__(self):

        self.networks = None
        self.networks_single = None
        self.networks_teacher = None
        self.networks_kd = None
        self.networks_multi = None
        self.optimizer = None
        self.scheduler = None
        self.train_mode = None
        self.train_params = None
        self.use_specific_final = None
        self.prepared = False

    def prepare(self,
                config: argparse.Namespace,
                loss_function_ce,
                loss_function_sim,
                loss_function_diff,
                loss_function_recon,
                save_log,
                local_rank: int = 0,
                **kwargs):

        self.config = config

        self.checkpoint_dir = config.checkpoint_dir
        self.batch_size = config.batch_size
        self.short = config.short

        self.loss_function_ce = loss_function_ce
        self.loss_function_sim = loss_function_sim
        self.loss_function_diff = loss_function_diff
        self.loss_function_recon = loss_function_recon

        self.save_log = save_log
        self.logs = collections.defaultdict(dict)

        self.local_rank = local_rank

        self.prepared = True

    def run(self, datasets, **kwargs):

        if not self.prepared:
            raise RuntimeError("Training is not prepared.")

        # dataset & dataloader
        loaders = {
            'train_complete': DataLoader(dataset=datasets['train_complete'], batch_size=self.batch_size,
                                         shuffle=True, drop_last=True),
            'train_incomplete': DataLoader(dataset=datasets['train_incomplete'], batch_size=self.batch_size,
                                           shuffle=True, drop_last=True),
            'train_total': DataLoader(dataset=datasets['train_total'], batch_size=self.batch_size,
                                      shuffle=True, drop_last=True),
            'test': DataLoader(dataset=datasets['test'], batch_size=self.batch_size,
                               shuffle=False, drop_last=False)
        }
        # 1. Student from scratch
        if 1 in self.config.train_level:
            self.train_mode = 'single'
            self.train_params = copy.deepcopy(self.config.train_params[self.train_mode])
            self.create_network_optimizer(train_mode=self.train_mode, train_params=self.train_params)

            # train
            with get_rich_pbar(transient=True, auto_refresh=False) as pg:
                task = pg.add_task(f"[bold red] Training Single...")
                for epoch in range(1, self.train_params['epochs'] + 1):
                    self.epoch = epoch
                    train_history = self.train_single(loaders['train_total'], train=True, adjusted=False)
                    with torch.no_grad():
                        test_history = self.train_single(loaders['test'], train=False, adjusted=False)

                    epoch_history = self.make_epoch_history(train_history=train_history,
                                                            test_history=test_history,
                                                            adjusted=False)
                    if self.config.enable_wandb:
                        wandb.log({'epoch_single': epoch}, commit=False)
                        if self.scheduler is not None:
                            wandb.log({'lr_single': self.scheduler.get_last_lr()[0]}, commit=False)
                        else:
                            wandb.log({'lr_single': self.optimizer.param_groups[0]['lr']}, commit=False)
                        wandb.log(epoch_history)

                    if self.save_log:
                        self.logs[self.train_mode][epoch] = epoch_history
                    desc = f"[bold green] Training Single... Epoch {self.epoch} / {self.train_params['epochs']}"
                    pg.update(task, advance=1.0, description=desc)
                    pg.refresh()

                    # update learning rate
                    if self.scheduler is not None:
                        self.scheduler.step()
            del pg

            # last & adjusted
            with torch.no_grad():
                adjusted_history = self.train_single(loaders['test'], train=False, adjusted=True)
            adjusted_history = self.make_epoch_history(train_history=None,
                                                       test_history=adjusted_history,
                                                       adjusted=True)
            if self.config.enable_wandb:
                wandb.log(adjusted_history)
            if self.save_log:
                self.logs[self.train_mode]['adjusted'] = adjusted_history

            # save network
            self.networks_single = self.networks

        # 2. General Teacher
        if 2 in self.config.train_level:
            self.train_mode = 'teacher'
            self.train_params = copy.deepcopy(self.config.train_params[self.train_mode])
            self.create_network_optimizer(train_mode=self.train_mode, train_params=self.train_params)

            # train
            with get_rich_pbar(transient=True, auto_refresh=False) as pg:
                task = pg.add_task(f"[bold red] Training Teacher...")
                for epoch in range(1, self.train_params['epochs'] + 1):
                    self.epoch = epoch
                    train_history = self.train_teacher(loaders['train_complete'], train=True, adjusted=False)
                    with torch.no_grad():
                        test_history = self.train_teacher(loaders['test'], train=False, adjusted=False)

                    epoch_history = self.make_epoch_history(train_history=train_history,
                                                            test_history=test_history,
                                                            adjusted=False)

                    if self.config.enable_wandb:
                        wandb.log({'epoch_teacher': epoch}, commit=False)
                        if self.scheduler is not None:
                            wandb.log({'lr_teacher': self.scheduler.get_last_lr()[0]}, commit=False)
                        else:
                            wandb.log({'lr_teacher': self.optimizer.param_groups[0]['lr']}, commit=False)
                        wandb.log(epoch_history)

                    if self.save_log:
                        self.logs[self.train_mode][epoch] = epoch_history

                    desc = f"[bold green] Training Teacher... Epoch {self.epoch} / {self.train_params['epochs']}"
                    pg.update(task, advance=1.0, description=desc)
                    pg.refresh()

                    # update learning rate
                    if self.scheduler is not None:
                        self.scheduler.step()
            del pg

            # last & adjusted
            with torch.no_grad():
                adjusted_history = self.train_teacher(loaders['test'], train=False, adjusted=True)
            adjusted_history = self.make_epoch_history(train_history=None,
                                                       test_history=adjusted_history,
                                                       adjusted=True)
            if self.config.enable_wandb:
                wandb.log(adjusted_history)
            if self.save_log:
                self.logs[self.train_mode]['adjusted'] = adjusted_history

            # save network
            self.networks_teacher = copy.deepcopy(self.networks)

        # 3. Knowledge Distillation
        if 3 in self.config.train_level:
            self.train_mode = 'kd'
            self.train_params = copy.deepcopy(self.config.train_params[self.train_mode])
            self.create_network_optimizer(train_mode=self.train_mode, train_params=self.train_params)

            # train
            with get_rich_pbar(transient=True, auto_refresh=False) as pg:
                task = pg.add_task(f"[bold red] Training KD...")
                for epoch in tqdm.tqdm(range(1, self.train_params['epochs'] + 1),
                                       total=max(len(loaders['train_complete']), len(loaders['train_incomplete'])),
                                       desc='Training KD'):
                    self.epoch = epoch
                    train_history = self.train_kd(loaders['train_complete'],
                                                  loaders['train_incomplete'],
                                                  train=True,
                                                  adjusted=False)
                    with torch.no_grad():
                        test_history = self.train_kd(loaders['test'], loaders['test'], train=False, adjusted=False)

                    epoch_history = self.make_epoch_history(train_history=train_history,
                                                            test_history=test_history,
                                                            adjusted=False)
                    if self.config.enable_wandb:
                        wandb.log({'epoch_kd': epoch}, commit=False)
                        if self.scheduler is not None:
                            wandb.log({'lr_kd': self.scheduler.get_last_lr()[0]}, commit=False)
                        else:
                            wandb.log({'lr_kd': self.optimizer.param_groups[0]['lr']}, commit=False)
                        wandb.log(epoch_history)

                    if self.save_log:
                        self.logs[self.train_mode][epoch] = epoch_history
                    desc = f"[bold green] Training KD... Epoch {self.epoch} / {self.train_params['epochs']}"
                    pg.update(task, advance=1.0, description=desc)
                    pg.refresh()

                    # update learning rate
                    if self.scheduler is not None:
                        self.scheduler.step()
            del pg

            # last & adjusted
            with torch.no_grad():
                adjusted_history = self.train_kd(loaders['test'], loaders['test'], train=False, adjusted=True)
            adjusted_history = self.make_epoch_history(train_history=None,
                                                       test_history=adjusted_history,
                                                       adjusted=True)
            if self.config.enable_wandb:
                wandb.log(adjusted_history)
            if self.save_log:
                self.logs[self.train_mode]['adjusted'] = adjusted_history

            # save network
            self.networks_kd = self.networks

        # 4. Final Multi-Modal
        if 4 in self.config.train_level:

            for use_specific_final in [True, False]:
                self.use_specific_final = use_specific_final
                self.train_mode = 'final'
                self.train_params = copy.deepcopy(self.config.train_params[self.train_mode])
                self.create_network_optimizer(train_mode=self.train_mode, train_params=self.train_params)

                # train
                with get_rich_pbar(transient=True, auto_refresh=False) as pg:
                    task = pg.add_task(f"[bold red] Training Final ({self.use_specific_final})...")
                    for epoch in tqdm.tqdm(range(1, self.train_params['epochs'] + 1), total=len(loaders['train_complete']),
                                           desc=f'Training Final ({self.use_specific_final})'):
                        self.epoch = epoch
                        train_history = self.train_final(loaders['train_complete'], train=True, adjusted=False)
                        with torch.no_grad():
                            test_history = self.train_final(loaders['test'], train=False, adjusted=False)

                        epoch_history = self.make_epoch_history(train_history=train_history,
                                                                test_history=test_history,
                                                                adjusted=False)
                        if self.config.enable_wandb:
                            wandb.log({f'epoch_final_{use_specific_final}': epoch}, commit=False)
                            if self.scheduler is not None:
                                wandb.log({f'lr_final_{use_specific_final}': self.scheduler.get_last_lr()[0]}, commit=False)
                            else:
                                wandb.log({f'lr_final_{use_specific_final}': self.optimizer.param_groups[0]['lr']}, commit=False)
                            wandb.log(epoch_history)

                        if self.save_log:
                            self.logs[self.train_mode][epoch] = epoch_history
                        desc = f"[bold green] Training Final ({self.use_specific_final})... " \
                               f"Epoch {self.epoch} / {self.train_params['epochs']}"
                        pg.update(task, advance=1.0, description=desc)
                        pg.refresh()

                        # update learning rate
                        if self.scheduler is not None:
                            self.scheduler.step()
                del pg

                # last & adjusted
                with torch.no_grad():
                    adjusted_history = self.train_final(loaders['test'], train=False, adjusted=True)
                adjusted_history = self.make_epoch_history(train_history=None,
                                                           test_history=adjusted_history,
                                                           adjusted=True)
                if self.config.enable_wandb:
                    wandb.log(adjusted_history)
                if self.save_log:
                    self.logs[f'{self.train_mode}-{use_specific_final}']['adjusted'] = adjusted_history

        # 5. Multi-Modal
        if 5 in self.config.train_level:
            self.train_mode = 'multi'
            self.train_params = copy.deepcopy(self.config.train_params[self.train_mode])
            self.create_network_optimizer(train_mode=self.train_mode, train_params=self.train_params)

            # train
            with get_rich_pbar(transient=True, auto_refresh=False) as pg:
                task = pg.add_task(f"[bold red] Training Multi...")
                for epoch in tqdm.tqdm(range(1, self.train_params['epochs'] + 1), total=len(loaders['train_complete']),
                                       desc='Training Final'):
                    self.epoch = epoch
                    train_history = self.train_multi(loaders['train_complete'], train=True, adjusted=False)
                    with torch.no_grad():
                        test_history = self.train_multi(loaders['test'], train=False, adjusted=False)

                    epoch_history = self.make_epoch_history(train_history=train_history,
                                                            test_history=test_history,
                                                            adjusted=False)
                    if self.config.enable_wandb:
                        wandb.log({'epoch_multi': epoch}, commit=False)
                        if self.scheduler is not None:
                            wandb.log({'lr_multi': self.scheduler.get_last_lr()[0]}, commit=False)
                        else:
                            wandb.log({'lr_multi': self.optimizer.param_groups[0]['lr']}, commit=False)
                        wandb.log(epoch_history)

                    if self.save_log:
                        self.logs[self.train_mode][epoch] = epoch_history
                    desc = f"[bold green] Training Multi... Epoch {self.epoch} / {self.train_params['epochs']}"
                    pg.update(task, advance=1.0, description=desc)
                    pg.refresh()

                    # update learning rate
                    if self.scheduler is not None:
                        self.scheduler.step()
            del pg

            # last & adjusted
            with torch.no_grad():
                adjusted_history = self.train_multi(loaders['test'], train=False, adjusted=True)
            adjusted_history = self.make_epoch_history(train_history=None,
                                                       test_history=adjusted_history,
                                                       adjusted=True)
            if self.config.enable_wandb:
                wandb.log(adjusted_history)
            if self.save_log:
                self.logs[self.train_mode]['adjusted'] = adjusted_history

        # save results
        if self.save_log:
            with open(os.path.join(self.config.checkpoint_dir, 'logs.pkl'), 'wb') as fb:
                pickle.dump(self.logs, fb)

    def train_single(self, data_loader, train=True, adjusted=False):

        self._set_learning_phase(train=train, train_mode='single')
        steps = len(data_loader)
        metric_names = ['total_loss', 'loss_ce']
        result = {k: torch.zeros(steps, device=self.local_rank) for k in metric_names}

        y_true, y_pred = [], []
        for i, batch in enumerate(data_loader):
            loss, loss_ce, y, logit = self.train_single_step(batch)
            if train:
                self.update(loss)
            result['total_loss'] = loss.detach()
            result['loss_ce'] = loss_ce.detach()

            y_true.append(y)
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

    def train_single_step(self, batch):
        x1 = batch['x1'].float().to(self.local_rank)
        y = batch['y'].long().to(self.local_rank)

        if self.short:
            z1 = self.networks['extractor_1_s'](x1)
        else:
            h1 = self.networks['extractor_1_s'](x1)
            z1 = self.networks['encoder_general_s'](h1)
        logit = self.networks['classifier_s'](z1 * 2)

        loss_ce = self.loss_function_ce(logit, y)
        loss = self.config.alpha_ce * loss_ce

        return loss, loss_ce, y, logit

    def train_teacher(self, data_loader, train=True, adjusted=False):

        self._set_learning_phase(train=train, train_mode='teacher')
        steps = len(data_loader)
        metric_names = ['total_loss', 'loss_ce', 'loss_sim',
                        'loss_diff_specific', 'loss_diff_1', 'loss_diff_2',
                        'loss_recon_1', 'loss_recon_2']
        result = {k: torch.zeros(steps, device=self.local_rank) for k in metric_names}

        y_true, y_pred = [], []
        for i, batch in enumerate(data_loader):
            loss, loss_ce, loss_sim, loss_diff_specific, loss_diff_1, loss_diff_2, \
            loss_recon_1, loss_recon_2, y, logit = self.train_teacher_step(batch)
            if train:
                self.update(loss)

            # save
            result['total_loss'][i] = loss.detach()
            result['loss_ce'][i] = loss_ce.detach()
            result['loss_sim'][i] = loss_sim.detach()
            result['loss_diff_specific'][i] = loss_diff_specific.detach()
            result['loss_diff_1'][i] = loss_diff_1.detach()
            result['loss_diff_2'][i] = loss_diff_2.detach()
            result['loss_recon_1'][i] = loss_recon_1.detach()
            result['loss_recon_2'][i] = loss_recon_2.detach()

            y_true.append(y)
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

    def train_teacher_step(self, batch):
        # input data
        x1 = batch['x1'].float().to(self.local_rank)
        x2 = batch['x2'].float().to(self.local_rank)
        y = batch['y'].long().to(self.local_rank)

        # representation h and z
        h1 = self.networks['extractor_1'](x1)
        h2 = self.networks['extractor_2'](x2)

        z1_general = self.networks['encoder_general'](h1)
        z2_general = self.networks['encoder_general'](h2)
        z1 = self.networks['encoder_1'](h1)
        z2 = self.networks['encoder_2'](h2)

        # reconstruction
        h1_recon = self.networks['decoder_1'](z1_general + z1)
        h2_recon = self.networks['decoder_2'](z2_general + z2)

        # classification
        logit = self.networks['classifier'](z1_general + z2_general)

        # Losses
        # difference
        loss_diff_specific = self.loss_function_diff(z1, z2)
        loss_diff_1 = self.loss_function_diff(z1, z1_general)
        loss_diff_2 = self.loss_function_diff(z2, z2_general)
        loss_diff = loss_diff_specific + loss_diff_1 + loss_diff_2

        # similarity
        loss_sim = self.loss_function_sim(z1_general, z2_general)

        # reconstruction
        loss_recon_1 = self.loss_function_recon(h1_recon, h1)
        loss_recon_2 = self.loss_function_recon(h2_recon, h2)
        loss_recon = loss_recon_1 + loss_recon_2

        # cross-entropy
        loss_ce = self.loss_function_ce(logit, y)

        loss = self.config.alpha_ce * loss_ce + \
               self.config.alpha_sim * loss_sim + \
               self.config.alpha_diff * loss_diff + \
               self.config.alpha_recon * loss_recon

        return loss, loss_ce, loss_sim, loss_diff_specific, loss_diff_1, loss_diff_2,\
               loss_recon_1, loss_recon_2, y, logit

    def train_kd(self, data_complete_loader, data_incomplete_loader, train=True, adjusted=False):

        self._set_learning_phase(train=train, train_mode='kd')
        steps = max(len(data_complete_loader), len(data_incomplete_loader))
        metric_names = ['total_loss', 'loss_ce', 'loss_kd']
        result = {k: torch.zeros(steps, device=self.local_rank) for k in metric_names}

        # use cycle to extend short loader to be equal to long loader
        if len(data_complete_loader) >= len(data_incomplete_loader):
            complete_is_long = True
            long_loader, short_loader = data_complete_loader, data_incomplete_loader
        else:
            complete_is_long = False
            long_loader, short_loader = data_incomplete_loader, data_complete_loader
        short_loader_cycle = cycle(short_loader)

        y_true, y_pred = [], []

        for i, (batch_l, batch_s) in enumerate(zip(long_loader, short_loader_cycle)):
            if complete_is_long:
                batch_c, batch_ic = batch_l, batch_s
            else:
                batch_c, batch_ic = batch_s, batch_l
            loss, loss_ce, loss_kd, y, logit = self.train_kd_step(batch_c, batch_ic)
            if train:
                self.update(loss)

            result['total_loss'][i] = loss.detach()
            result['loss_ce'][i] = loss_ce.detach()
            result['loss_kd'] = loss_kd.detach()

            y_true.append(y)
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

    def train_kd_step(self, batch, batch_in):

        # A. Complete Training
        x1 = batch['x1'].float().to(self.local_rank)
        x2 = batch['x2'].float().to(self.local_rank)
        y = batch['y'].long().to(self.local_rank)

        # A1. Teacher
        with torch.no_grad():
            h1 = self.networks['extractor_1'](x1)
            h2 = self.networks['extractor_2'](x2)

            z1_general = self.networks['encoder_general'](h1)
            z2_general = self.networks['encoder_general'](h2)

            logit = self.networks['classifier'](z1_general + z2_general)

        # A2. Student
        h1_s = self.networks['extractor_1_s'](x1)
        z1_general_s = self.networks['encoder_general_s'](h1_s)
        logit_s = self.networks['classifier_s'](z1_general_s * 2)

        # A3. Knowledge Distillation
        # classification
        loss_kd_clf = F.kl_div(F.log_softmax(logit_s / self.config.temperature, dim=1),
                               F.softmax(logit / self.config.temperature, dim=1),
                               reduction='batchmean')

        # B. Incomplete Training
        x1_in = batch_in['x1'].float().to(self.local_rank)
        y_in = batch_in['y'].long().to(self.local_rank)

        h1_s_in = self.networks['extractor_1_s'](x1_in)
        z1_general_s_in = self.networks['encoder_general_s'](h1_s_in)
        logit_s_in = self.networks['classifier_s'](z1_general_s_in * 2)

        # C. Loss
        logit_total = torch.concat([logit_s, logit_s_in])
        y_total = torch.concat([y, y_in])
        loss_ce = self.loss_function_ce(logit_total, y_total)

        loss = self.config.alpha_ce * loss_ce + self.config.alpha_kd_clf * loss_kd_clf

        return loss, loss_ce, loss_kd_clf, y_total, logit_total

    # def train_final_with_ul
    def train_final_all(self, data_complete_loader, data_incomplete_loader, train=True, adjusted=False):

        self._set_learning_phase(train=train, train_mode='final')
        steps = max(len(data_complete_loader), len(data_incomplete_loader))
        metric_names = ['total_loss', 'loss_ce', 'loss_sim',
                        'loss_diff_specific', 'loss_diff_1', 'loss_diff_2',
                        'loss_recon_1', 'loss_recon_2', 'loss_kd']
        result = {k: torch.zeros(steps, device=self.local_rank) for k in metric_names}

        # use cycle to extend short loader to be equal to long loader
        if len(data_complete_loader) >= len(data_incomplete_loader):
            complete_is_long = True
            long_loader, short_loader = data_complete_loader, data_incomplete_loader
        else:
            complete_is_long = False
            long_loader, short_loader = data_incomplete_loader, data_complete_loader
        short_loader_cycle = cycle(short_loader)

        y_true, y_pred = [], []

        for i, (batch_l, batch_s) in enumerate(zip(long_loader, short_loader_cycle)):
            if complete_is_long:
                batch_c, batch_ic = batch_l, batch_s
            else:
                batch_c, batch_ic = batch_s, batch_l
            loss, loss_ce, loss_sim, loss_diff_specific, loss_diff_1, loss_diff_2, \
            loss_recon_1, loss_recon_2, loss_kd, y, logit = self.train_final_all_step(batch_c, batch_ic)
            if train:
                self.update(loss)

            # save
            result['total_loss'][i] = loss.detach()
            result['loss_ce'][i] = loss_ce.detach()
            result['loss_sim'][i] = loss_sim.detach()
            result['loss_diff_specific'][i] = loss_diff_specific.detach()
            result['loss_diff_1'][i] = loss_diff_1.detach()
            result['loss_diff_2'][i] = loss_diff_2.detach()
            result['loss_recon_1'][i] = loss_recon_1.detach()
            result['loss_recon_2'][i] = loss_recon_2.detach()
            result['loss_kd'][i] = loss_kd.detach()

            y_true.append(y)
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

    def train_final_all_step(self, batch, batch_in):

        # complete batch
        # input data
        x1 = batch['x1'].float().to(self.local_rank)
        x2 = batch['x2'].float().to(self.local_rank)
        y = batch['y'].long().to(self.local_rank)

        # 1. Student
        with torch.no_grad():
            h1_s = self.networks['extractor_1_s'](x1)

        # 2. Final Multi
        # representation h and z
        h1 = self.networks['extractor_1'](x1)
        h2 = self.networks['extractor_2'](x2)

        z1_general = self.networks['encoder_general'](h1)
        z2_general = self.networks['encoder_general'](h2)
        z1 = self.networks['encoder_1'](h1)
        z2 = self.networks['encoder_2'](h2)

        # reconstruction
        h1_recon = self.networks['decoder_1'](z1_general + z1)
        h2_recon = self.networks['decoder_2'](z2_general + z2)

        # classification
        if self.use_specific_final:
            logit = self.networks['classifier'](z1_general + z2_general + z1 + z2)
        else:
            logit = self.networks['classifier'](z1_general + z2_general)

        # incomplete batch
        x1_in = batch_in['x1'].float().to(self.local_rank)
        with torch.no_grad():
            h1_s_in = self.networks['extractor_1_s'](x1)

        ########## TODO

        # Losses
        # difference
        loss_diff_specific = self.loss_function_diff(z1, z2)
        loss_diff_1 = self.loss_function_diff(z1, z1_general)
        loss_diff_2 = self.loss_function_diff(z2, z2_general)
        loss_diff = loss_diff_specific + loss_diff_1 + loss_diff_2

        # similarity
        loss_sim = self.loss_function_sim(z1_general, z2_general)

        # reconstruction
        loss_recon_1 = self.loss_function_recon(h1_recon, h1)
        loss_recon_2 = self.loss_function_recon(h2_recon, h2)
        loss_recon = loss_recon_1 + loss_recon_2

        # cross-entropy
        loss_ce = self.loss_function_ce(logit, y)

        # knowledge distillation
        h1_s_norm = F.normalize(h1_s, p=2, dim=1)
        h1_norm = F.normalize(h1, p=2, dim=1)

        cos = torch.einsum('nc,nc->n', [h1_s_norm, h1_norm])
        loss_kd = (1 - cos) / (2 * self.config.temperature ** 2)
        loss_kd = loss_kd.mean()

        loss = self.config.alpha_ce * loss_ce + \
               self.config.alpha_sim * loss_sim + \
               self.config.alpha_diff * loss_diff + \
               self.config.alpha_recon * loss_recon + \
               self.config.alpha_kd_repr * loss_kd

        return loss, loss_ce, loss_sim, loss_diff_specific, loss_diff_1, loss_diff_2, \
               loss_recon_1, loss_recon_2, loss_kd, y, logit

    def train_final(self, data_loader, train=True, adjusted=False):

        self._set_learning_phase(train=train, train_mode='final')
        steps = len(data_loader)
        metric_names = ['total_loss', 'loss_ce', 'loss_sim',
                        'loss_diff_specific', 'loss_diff_1', 'loss_diff_2',
                        'loss_recon_1', 'loss_recon_2', 'loss_kd']
        result = {k: torch.zeros(steps, device=self.local_rank) for k in metric_names}

        y_true, y_pred = [], []
        for i, batch in enumerate(data_loader):
            loss, loss_ce, loss_sim, loss_diff_specific, loss_diff_1, loss_diff_2, \
            loss_recon_1, loss_recon_2, loss_kd, y, logit = self.train_final_step(batch)
            if train:
                self.update(loss)

            # save
            result['total_loss'][i] = loss.detach()
            result['loss_ce'][i] = loss_ce.detach()
            result['loss_sim'][i] = loss_sim.detach()
            result['loss_diff_specific'][i] = loss_diff_specific.detach()
            result['loss_diff_1'][i] = loss_diff_1.detach()
            result['loss_diff_2'][i] = loss_diff_2.detach()
            result['loss_recon_1'][i] = loss_recon_1.detach()
            result['loss_recon_2'][i] = loss_recon_2.detach()
            result['loss_kd'][i] = loss_kd.detach()

            y_true.append(y)
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

    def train_final_step(self, batch):

        # input data
        x1 = batch['x1'].float().to(self.local_rank)
        x2 = batch['x2'].float().to(self.local_rank)
        y = batch['y'].long().to(self.local_rank)

        # 1. Student
        with torch.no_grad():
            h1_s = self.networks['extractor_1_s'](x1)

        # 2. Final Multi
        # representation h and z
        h1 = self.networks['extractor_1'](x1)
        h2 = self.networks['extractor_2'](x2)

        z1_general = self.networks['encoder_general'](h1)
        z2_general = self.networks['encoder_general'](h2)
        z1 = self.networks['encoder_1'](h1)
        z2 = self.networks['encoder_2'](h2)

        # reconstruction
        h1_recon = self.networks['decoder_1'](z1_general + z1)
        h2_recon = self.networks['decoder_2'](z2_general + z2)

        # classification
        if self.use_specific_final:
            logit = self.networks['classifier'](z1_general + z2_general + z1 + z2)
        else:
            logit = self.networks['classifier'](z1_general + z2_general)

        # Losses
        # difference
        loss_diff_specific = self.loss_function_diff(z1, z2)
        loss_diff_1 = self.loss_function_diff(z1, z1_general)
        loss_diff_2 = self.loss_function_diff(z2, z2_general)
        loss_diff = loss_diff_specific + loss_diff_1 + loss_diff_2

        # similarity
        loss_sim = self.loss_function_sim(z1_general, z2_general)

        # reconstruction
        loss_recon_1 = self.loss_function_recon(h1_recon, h1)
        loss_recon_2 = self.loss_function_recon(h2_recon, h2)
        loss_recon = loss_recon_1 + loss_recon_2

        # cross-entropy
        loss_ce = self.loss_function_ce(logit, y)

        # knowledge distillation
        h1_s_norm = F.normalize(h1_s, p=2, dim=1)
        h1_norm = F.normalize(h1, p=2, dim=1)

        cos = torch.einsum('nc,nc->n', [h1_s_norm, h1_norm])
        loss_kd = (1 - cos) / (2 * self.config.temperature ** 2)
        loss_kd = loss_kd.mean()

        loss = self.config.alpha_ce * loss_ce + \
               self.config.alpha_sim * loss_sim + \
               self.config.alpha_diff * loss_diff + \
               self.config.alpha_recon * loss_recon + \
               self.config.alpha_kd_repr * loss_kd

        return loss, loss_ce, loss_sim, loss_diff_specific, loss_diff_1, loss_diff_2, \
               loss_recon_1, loss_recon_2, loss_kd, y, logit

    def train_multi(self, data_loader, train=True, adjusted=False):

        self._set_learning_phase(train=train, train_mode='multi')
        steps = len(data_loader)
        metric_names = ['total_loss', 'loss_ce']
        result = {k: torch.zeros(steps, device=self.local_rank) for k in metric_names}

        y_true, y_pred = [], []
        for i, batch in enumerate(data_loader):
            loss, loss_ce, y, logit = self.train_multi_step(batch)
            if train:
                self.update(loss)

            # save
            result['total_loss'][i] = loss.detach()
            result['loss_ce'][i] = loss_ce.detach()

            y_true.append(y)
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

    def train_multi_step(self, batch):
        # input data
        x1 = batch['x1'].float().to(self.local_rank)
        x2 = batch['x2'].float().to(self.local_rank)
        y = batch['y'].long().to(self.local_rank)

        if self.short:
            z1 = self.networks['extractor_1'](x1)
            z2 = self.networks['extractor_2'](x2)
        else:
            # representation h and z
            h1 = self.networks['extractor_1'](x1)
            h2 = self.networks['extractor_2'](x2)

            z1 = self.networks['encoder_1'](h1)
            z2 = self.networks['encoder_2'](h2)

        # classification
        logit = self.networks['classifier'](z1 + z2)

        # Losses
        # cross-entropy
        loss_ce = self.loss_function_ce(logit, y)

        loss = self.config.alpha_ce * loss_ce

        return loss, loss_ce, y, logit

    def update(self, loss):
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

    def create_network_optimizer(self, train_mode, train_params: dict):

        assert train_mode in ['single', 'teacher', 'kd', 'final', 'multi']
        epochs = train_params['epochs']
        learning_rate = train_params['learning_rate']
        weight_decay = train_params['weight_decay']

        # build networks and bring target pre-trained weights
        if self.short and train_mode in ['single', 'multi']:
            networks = build_short_networks(config=self.config)
        else:
            if self.config.simple:
                networks = build_simple_networks(config=self.config)
            else:
                networks = build_networks(config=self.config)
            networks_student = {
                'extractor_1_s': copy.deepcopy(networks['extractor_1']),
                'encoder_general_s': copy.deepcopy(networks['encoder_general']),
                'classifier_s': copy.deepcopy(networks['classifier'])
            }

        params = []
        if train_mode == 'single':
            if self.short:
                self.networks = {'extractor_1_s': copy.deepcopy(networks['extractor_1']),
                                 'classifier_s': copy.deepcopy(networks['classifier'])}
            else:
                # use only student network
                self.networks = copy.deepcopy(networks_student)
            for name in self.networks.keys():
                params = params + [{'params': self.networks[name].parameters(), 'lr': learning_rate}]

        elif train_mode == 'teacher':
            # use only general_teacher network
            self.networks = copy.deepcopy(networks)
            for name in self.networks.keys():
                params = params + [{'params': self.networks[name].parameters(), 'lr': learning_rate}]

        elif train_mode == 'kd':
            # bring all pre-trained weights from networks_teacher
            for k in networks.keys():
                networks[k].load_state_dict(self.networks_teacher[k].state_dict())

            # include student network that uses pre-trained weights from networks_teacher
            for k, v in networks_student.items():
                v.load_state_dict(self.networks_teacher[k.replace('_s', '')].state_dict())
                networks[k] = v

            self.networks = copy.deepcopy(networks)

            for name in self.networks.keys():
                if name.endswith('_s'):
                    params = params + [{'params': self.networks[name].parameters(), 'lr': learning_rate}]

        elif train_mode == 'final':
            # bring extractor_1_s weights from networks_kd to both extractor_1 and extractor_1_s
            networks_student['extractor_1_s'].load_state_dict(self.networks_kd['extractor_1_s'].state_dict())
            # networks['extractor_1'].load_state_dict(self.networks_kd['extractor_1_s'].state_dict())

            for k, v in networks_student.items():
                networks[k] = v

            self.networks = copy.deepcopy(networks)

            for name in self.networks.keys():
                if not name.endswith('_s'):
                    params = params + [{'params': self.networks[name].parameters(), 'lr': learning_rate}]

        elif train_mode == 'multi':
            if self.short:
                self.networks = copy.deepcopy(networks)
            else:
                self.networks = copy.deepcopy(networks)
                del self.networks['encoder_general']
            for name in self.networks.keys():
                params = params + [{'params': self.networks[name].parameters(), 'lr': learning_rate}]

        else:
            raise ValueError

        _ = [v.to(self.local_rank) for k, v in self.networks.items()]

        self.optimizer = get_optimizer(params=params,
                                       name=self.config.optimizer,
                                       lr=learning_rate,
                                       weight_decay=weight_decay)
        self.scheduler = get_cosine_scheduler(self.optimizer,
                                              epochs=epochs,
                                              warmup_steps=self.config.cosine_warmup,
                                              cycles=self.config.cosine_cycles,
                                              min_lr=self.config.cosine_min_lr)

        if self.short:
            del networks
        else:
            del networks, networks_student

    def make_epoch_history(self, train_history: dict = None, test_history: dict = None, adjusted: bool = False):
        epoch_history = collections.defaultdict(dict)
        train_mode = self.train_mode
        if self.train_mode == 'final':
            train_mode = f'{train_mode}-{self.use_specific_final}'
        if not adjusted:
            if train_history is not None:
                for k, v in train_history.items():
                    epoch_history[f'{train_mode}-train/{k}'] = v
            if test_history is not None:
                for k, v in test_history.items():
                    epoch_history[f'{train_mode}-test/{k}'] = v
        else:
            assert test_history is not None
            for k, v in test_history.items():
                epoch_history[f'{train_mode}-adjusted/{k}'] = v
        return epoch_history

    @staticmethod
    def freeze_params(net: nn.Module, freeze: bool):
        for p in net.parameters():
            p.requires_grad = not freeze

    def _set_learning_phase(self, train: bool = True, train_mode: str = 'teacher'):
        if train_mode == 'single':
            for name, network in self.networks.items():
                if train:
                    network.train()
                else:
                    network.eval()
        elif train_mode == 'teacher':
            for name, network in self.networks.items():
                if train:
                    network.train()
                else:
                    network.eval()
        elif train_mode == 'kd':
            for name in self.networks.keys():
                if name.endswith('_s'):
                    if train:
                        self.networks[name].train()
                    else:
                        self.networks[name].eval()
                else:
                    self.networks[name].eval()
        elif train_mode == 'final':
            for name in self.networks.keys():
                if not name.endswith('_s'):
                    if train:
                        self.networks[name].train()
                    else:
                        self.networks[name].eval()
                else:
                    self.networks[name].eval()
        elif train_mode == 'multi':
            for name, network in self.networks.items():
                if train:
                    network.train()
                else:
                    network.eval()
        else:
            raise ValueError
