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
from utils.optimization import get_optimizer, get_cosine_scheduler


class Simulator:

    def __init__(self, networks: dict):

        self.networks = networks
        self.optimizer = None
        self.scheduler = None
        self.train_mode = None
        self.train_params = None
        self.prepared = False

    def prepare(self,
                config: argparse.Namespace,
                loss_function_ce,
                loss_function_sim,
                loss_function_diff,
                loss_function_recon,
                local_rank: int = 0,
                **kwargs):

        self.config = config

        self.checkpoint_dir = config.checkpoint_dir
        self.batch_size = config.batch_size

        self.loss_function_ce = loss_function_ce
        self.loss_function_sim = loss_function_sim
        self.loss_function_diff = loss_function_diff
        self.loss_function_recon = loss_function_recon

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
            'test': DataLoader(dataset=datasets['test'], batch_size=self.batch_size,
                               shuffle=False, drop_last=False)
        }

        logger = kwargs.get('logger', None)

        # 1. General Teacher
        self.train_mode = 'teacher'
        self.train_params = self.config.train_params[self.train_mode]
        self.set_optimizer(train_mode=self.train_mode, train_params=self.train_params)

        for epoch in range(1, self.train_params['epochs'] + 1):
            epoch_history = collections.defaultdict(dict)
            train_history = self.train_teacher(loaders['train_complete'], train=True, adjusted=False)
            with torch.no_grad():
                test_history = self.train_teacher(loaders['test'], train=False, adjusted=False)

        # 2. Knowledge Distillation
        self.train_mode = 'kd'
        self.train_params = self.config.train_params[self.train_mode]
        self.set_optimizer(train_mode=self.train_mode, train_params=self.train_params)

        for epoch in range(1, self.train_params['epochs'] + 1):
            epoch_history = collections.defaultdict(dict)
            train_history = self.train_kd(loaders['train_complete'],
                                          loaders['train_incomplete'],
                                          train=True,
                                          adjusted=False)
            with torch.no_grad():
                pass

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
        steps = min(len(data_complete_loader), len(data_incomplete_loader))
        metric_names = ['total_loss', 'loss_ce', 'loss_kd']
        result = {k: torch.zeros(steps, device=self.local_rank) for k in metric_names}

        y_true, y_pred = [], []
        for i, (batch_c, batch_ic) in enumerate(zip(data_complete_loader, data_incomplete_loader)):
            loss, loss_ce, loss_kd, y, logit = self.train_kd_step(batch_c, batch_ic)

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
                               reduction='mean')

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

    def update(self, loss):
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

    def set_optimizer(self, train_mode, train_params: dict):

        assert train_mode in ['teacher', 'kd', 'final']
        epochs = train_params['epochs']
        learning_rate = train_params['learning_rate']
        weight_decay = train_params['weight_deacy']

        params = []

        if train_mode == 'teacher':
            for name in self.networks.keys():
                params = params + [{'params': self.networks[name].parameters(), 'lr': learning_rate}]
        elif train_mode == 'kd':
            for name in self.networks.keys():
                if name.endswith('_s'):
                    params = params + [{'params': self.networks[name].parameters(), 'lr': learning_rate}]
        elif train_mode == 'final':
            for name in self.networks.keys():
                if not name.endswith('_s'):
                    params = params + [{'params': self.networks[name].parameters(), 'lr': learning_rate}]
        else:
            raise ValueError

        self.optimizer = get_optimizer(params=params,
                                       name=self.config.optimizer,
                                       lr=learning_rate,
                                       weight_decay=weight_decay)
        self.scheduler = get_cosine_scheduler(self.optimizer,
                                              epochs=epochs,
                                              warmup_steps=self.config.cosine_warmup,
                                              cycles=self.config.cosine_cycles,
                                              min_lr=self.config.cosine_min_lr)

    @staticmethod
    def freeze_params(net: nn.Module, freeze: bool):
        for p in net.parameters():
            p.requires_grad = not freeze

    def _set_learning_phase(self, train: bool = True, train_mode: str = 'teacher'):
        if train_mode == 'teacher':
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
        else:
            raise ValueError
