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


class Swap(object):

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
                incomplete_ratio: int = 1,
                add_type: str = 'concat',
                alpha: float = 1.0,
                swap: bool = True,
                num_workers: int = 4,
                distributed: bool = False,
                local_rank: int = 0,
                mixed_precision: bool = True,
                enable_wandb: bool = True,
                **kwargs):

        # TODO: ramp_up
        # Assertion
        assert add_type in ['concat', 'add']
        assert incomplete_ratio == 1

        # Set attributes
        self.checkpoint_dir = checkpoint_dir
        self.loss_function_ce = loss_function_ce
        self.epochs = epochs
        self.batch_size = batch_size
        self.accumulate = accumulate
        self.incomplete_ratio = incomplete_ratio
        self.add_type = add_type
        self.alpha = alpha
        self.swap = swap
        self.num_workers = num_workers
        self.distributed = distributed
        self.local_rank = local_rank
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
                {'params': self.networks['mri_common_projector'].parameters()},
                {'params': self.networks['pet_common_projector'].parameters()},
                {'params': self.networks['predictor'].parameters()},
                {'params': self.networks['mri_classifier'].parameters()},
                {'params': self.networks['pet_classifier'].parameters()},
                {'params': self.networks['common_classifier'].parameters()}
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
            'mri_pet_complete_validation': DataLoader(dataset=datasets['mri_pet_complete_validation'],
                                                batch_size=self.batch_size,
                                                drop_last=False),
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

            self.epoch = epoch

            # Train, Validation, and Test
            train_history, train_clf_result = self.train(
                complete_loader=loaders['mri_pet_complete_train'],
                incomplete_loader=loaders['mri_incomplete_train'],
                adjusted=False
            )
            validation_history, validation_clf_result = self.evaluate(
                eval_loader=loaders['mri_pet_complete_validation'],
                adjusted=False
            )
            test_history, test_clf_result = self.evaluate(
                eval_loader=loaders['mri_pet_complete_test'],
                adjusted=False
            )

            train_history = {'train-loss/' + k: v for k, v in train_history.items()}
            validation_history = {'validation-loss/' + k: v for k, v in validation_history.items()}
            test_history = {'test-loss/' + k: v for k, v in test_history.items()}

            train_clf_result = {f'train-{k1}-{k2}/{k3}': v3
                                for k1, v1 in train_clf_result.items()
                                for k2, v2 in v1.items()
                                for k3, v3 in v2.items()}
            validation_clf_result = {f'validation-{k1}-{k2}/{k3}': v3
                                     for k1, v1 in validation_clf_result.items()
                                     for k2, v2 in v1.items()
                                     for k3, v3 in v2.items()}
            test_clf_result = {f'test-{k1}-{k2}/{k3}': v3
                               for k1, v1 in test_clf_result.items()
                               for k2, v2 in v1.items()
                               for k3, v3 in v2.items()}

            if self.enable_wandb:
                wandb.log({'epoch': epoch}, commit=False)
                if self.scheduler is not None:
                    wandb.log({'lr': self.scheduler.get_last_lr()[0]}, commit=False)
                else:
                    wandb.log({'lr': self.optimizer.param_groups[0]['lr']}, commit=False)

                wandb.log(train_history, commit=False)
                wandb.log(validation_history, commit=False)
                wandb.log(test_history, commit=False)

                wandb.log(train_clf_result, commit=False)
                wandb.log(validation_clf_result, commit=False)
                wandb.log(test_clf_result, commit=True)

            # # Logging
            # epoch_history = collections.defaultdict(dict)
            # for phase, history in zip(['phase1'], [train_history]):
            #     for k, v in history.items():
            #         epoch_history[f'train/{phase}/{k}'] = v
            #
            # for phase, history in zip(['complete'], [test_history]):
            #     for k, v in history.items():
            #         epoch_history[f'test/{phase}/{k}'] = v
            #
            # # log = make_epoch_description(
            #     history=epoch_history,
            #     current=epoch,
            #     total=self.epochs,
            #     best=best_epoch,
            # )
            # if logger is not None:
            #     logger.info(log)

            # if self.enable_wandb:
            #     wandb.log({'epoch': epoch}, commit=False)
            #     if self.scheduler is not None:
            #         wandb.log({'lr': self.scheduler.get_last_lr()[0]}, commit=False)
            #     else:
            #         wandb.log({'lr': self.optimizer.param_groups[0]['lr']}, commit=False)
            #     wandb.log(epoch_history)

            # Save best model checkpoint
            eval_loss = validation_history['validation-loss/loss']
            if eval_loss <= best_eval_loss:
                best_eval_loss = eval_loss
                best_epoch = epoch
                if self.local_rank == 0:
                    ckpt = os.path.join(self.checkpoint_dir, f"ckpt.best.pth.tar")
                    self.save_checkpoint(ckpt, epoch=best_epoch)

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
        _, adjusted_clf_result = self.evaluate(
            eval_loader=loaders['mri_pet_complete_test'],
            adjusted=True
        )
        adjusted_clf_result = {f'adjusted-{k1}-{k2}/{k3}': v3
                               for k1, v1 in adjusted_clf_result.items()
                               for k2, v2 in v1.items()
                               for k3, v3 in v2.items()}
        if self.enable_wandb:
            wandb.log(adjusted_clf_result)

    def complete_step(self, batch) -> dict:

        # input
        x_mri = batch['mri'].float().to(self.local_rank)
        x_pet = batch['pet'].float().to(self.local_rank)
        y = batch['y'].long().to(self.local_rank)

        # encoder
        h_mri = self.networks['mri_encoder'](x_mri)
        h_pet = self.networks['pet_encoder'](x_pet)

        # projector
        z_mri = self.networks['mri_projector'](h_mri)
        z_mri_common = self.networks['mri_common_projector'](h_mri)

        z_pet = self.networks['pet_projector'](h_pet)
        z_pet_common = self.networks['pet_common_projector'](h_pet)

        # predictor
        z_pet_common_pred = self.networks['predictor'](z_mri_common)

        # add z
        if self.add_type == 'concat':
            if self.swap:
                z_m = torch.concat([z_mri, z_pet_common], dim=1)
                z_p = torch.concat([z_pet, z_mri_common], dim=1)
            else:
                z_m = torch.concat([z_mri, z_mri_common], dim=1)
                z_p = torch.concat([z_pet, z_pet_common], dim=1)
            z_c = torch.concat([z_mri_common, z_pet_common], dim=1)
        else:
            if self.swap:
                z_m = z_mri + z_pet_common
                z_p = z_pet + z_mri_common
            else:
                z_m = z_mri + z_mri_common
                z_p = z_pet + z_pet_common
            z_c = z_mri_common + z_pet_common

        # classifier
        logit_m = self.networks['mri_classifier'](z_m)
        logit_p = self.networks['pet_classifier'](z_p)
        logit_c = self.networks['common_classifier'](z_c)

        # losses
        ce_m = self.loss_function_ce(logit_m, y)
        ce_p = self.loss_function_ce(logit_p, y)
        ce_c = self.loss_function_ce(logit_c, y)

        mse = F.mse_loss(z_pet_common_pred, z_pet_common, reduction='sum')

        out = {'y_true': y.long(),
               'y_pred': {'mri': logit_m, 'pet': logit_p, 'common': logit_c},
               'ce': {'mri': ce_m, 'pet': ce_p, 'common': ce_c},
               'mse': mse}

        return out

    def incomplete_step(self, batch) -> dict:

        # input
        x_mri = batch['mri'].float().to(self.local_rank)
        y = batch['y'].long().to(self.local_rank)

        # encoder
        h_mri = self.networks['mri_encoder'](x_mri)

        # projector
        z_mri = self.networks['mri_projector'](h_mri)
        z_mri_common = self.networks['mri_common_projector'](h_mri)

        # predictor with stopping gradient
        with torch.no_grad():
            z_pet_common_pred = self.networks['predictor'](z_mri_common)

        # add z
        if self.add_type == 'concat':
            if self.swap:
                z_m = torch.concat([z_mri, z_pet_common_pred], dim=1)
            else:
                z_m = torch.concat([z_mri, z_mri_common], dim=1)
            z_c = torch.concat([z_mri_common, z_pet_common_pred], dim=1)
        else:
            if self.swap:
                z_m = z_mri + z_pet_common_pred
            else:
                z_m = z_mri + z_mri_common
            z_c = z_mri_common + z_pet_common_pred

        # classifier
        logit_m = self.networks['mri_classifier'](z_m)
        logit_c = self.networks['common_classifier'](z_c)

        # losses
        ce_m = self.loss_function_ce(logit_m, y)
        ce_c = self.loss_function_ce(logit_c, y)

        out = {'y_true': y.long(),
               'y_pred': {'mri': logit_m, 'common': logit_c},
               'ce': {'mri': ce_m, 'common': ce_c}}

        return out

    def train(self, complete_loader, incomplete_loader, adjusted=False):

        # Set network
        self._set_learning_phase(train=True)

        # Logging
        len_loader = len(complete_loader)
        steps = (len_loader * self.batch_size // (self.batch_size * self.accumulate)) * self.accumulate

        history = {'loss': torch.zeros(steps, device=self.local_rank),
                   'ce_mri': torch.zeros(steps, device=self.local_rank),
                   'ce_pet': torch.zeros(steps, device=self.local_rank),
                   'ce_common': torch.zeros(steps, device=self.local_rank),
                   'mse': torch.zeros(steps, device=self.local_rank)}

        with get_rich_pbar(transient=True, auto_refresh=False) as pg:

            if self.local_rank == 0:
                task = pg.add_task(f"[bold red] Training...", total=steps)

            y_true = {
                'complete': [],
                'incomplete': []
            }
            y_pred = {
                'complete': {'mri': [], 'pet': [], 'common': []},
                'incomplete': {'mri': [], 'common': []}
            }

            incomplete_iter = incomplete_loader.__iter__()

            for i, batch_complete in enumerate(complete_loader):
                with torch.cuda.amp.autocast(self.mixed_precision):

                    # 1. Train
                    # complete
                    result_complete = self.complete_step(batch=batch_complete)
                    del batch_complete

                    # incomplete
                    try:
                        batch_incomplete = next(incomplete_iter)
                    except StopIteration:
                        incomplete_iter = incomplete_loader.__iter__()
                        batch_incomplete = next(incomplete_iter)
                    result_incomplete = self.incomplete_step(batch=batch_incomplete)
                    del batch_incomplete

                    # sum and divide by valid number of observations
                    n_complete, n_incomplete = len(result_complete['y_true']), len(result_incomplete['y_true'])

                    # loss
                    ce_mri = result_complete['ce']['mri'].sum() + result_incomplete['ce']['mri'].sum()
                    ce_pet = result_complete['ce']['pet'].sum()
                    ce_common = result_complete['ce']['common'].sum() + result_incomplete['ce']['common']
                    mse = result_complete['mse']

                    ce_mri = ce_mri / (n_complete + n_incomplete)
                    ce_pet = ce_pet / n_complete
                    ce_common = ce_common / (n_complete + n_incomplete)
                    mse = mse / n_complete

                    loss = (ce_mri + ce_pet + ce_common) + self.alpha * mse
                    loss = loss / (3 + self.alpha)
                    loss = loss / self.accumulate

                # Accumulate scaled gradients
                if self.scaler is not None:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()

                # Update Optimizer
                if (i + 1) % self.accumulate == 0:
                    if self.scaler is not None:

                        # self.scaler.unscale_(self.optimizer)
                        # for name, net in self.networks.items():
                        #     torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=10)

                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()
                    self.optimizer.zero_grad()

                # save monitoring values
                history['loss'][i] = loss.item() * self.accumulate
                history['ce_mri'][i] = ce_mri.item()
                history['ce_pet'][i] = ce_pet.item()
                history['ce_common'][i] = ce_common.item()
                history['mse'][i] = mse

                if self.local_rank == 0:
                    desc = f"[bold green] Epoch {self.epoch} [{i+1}/{steps}]: "
                    for k, v in history.items():
                        desc += f" {k} : {v[:i+1].mean():.4f} |"
                    pg.update(task, advance=1., description=desc)
                    pg.refresh()

                # save logits and y_true for evaluation
                y_true['complete'].append(result_complete['y_true'])
                y_true['incomplete'].append(result_incomplete['y_true'])

                y_pred['complete']['mri'].append(result_complete['y_pred']['mri'])
                y_pred['complete']['pet'].append(result_complete['y_pred']['pet'])
                y_pred['complete']['common'].append(result_complete['y_pred']['common'])
                y_pred['incomplete']['mri'].append(result_incomplete['y_pred']['mri'])
                y_pred['incomplete']['common'].append(result_incomplete['y_pred']['common'])

                # Ignore the remained batches
                if (i + 1) == steps:
                    break

        history = {k: v.mean().item() for k, v in history.items()}

        # enforce to float32: accuracy and macro f1 score
        y_true['complete'] = torch.cat(y_true['complete'], dim=0)
        y_true['incomplete'] = torch.cat(y_true['incomplete'], dim=0)

        y_pred['complete']['mri'] = torch.cat(y_pred['complete']['mri'], dim=0).to(torch.float32)
        y_pred['complete']['pet'] = torch.cat(y_pred['complete']['pet'], dim=0).to(torch.float32)
        y_pred['complete']['common'] = torch.cat(y_pred['complete']['common'], dim=0).to(torch.float32)
        y_pred['incomplete']['mri'] = torch.cat(y_pred['incomplete']['mri'], dim=0).to(torch.float32)
        y_pred['incomplete']['common'] = torch.cat(y_pred['incomplete']['common'], dim=0).to(torch.float32)

        # classification result
        clf_result = {}
        clf_result['complete'] = {}
        clf_result['incomplete'] = {}

        for data_type in ['mri', 'pet', 'common']:
            clf_result_ = classification_result(
                y_true=y_true['complete'].cpu().numpy(),
                y_pred=y_pred['complete'][data_type].softmax(1).detach().cpu().numpy(),
                adjusted=adjusted
            )
            clf_result['complete'][data_type] = clf_result_

        for data_type in ['mri', 'common']:
            clf_result_ = classification_result(
                y_true=y_true['incomplete'].cpu().numpy(),
                y_pred=y_pred['incomplete'][data_type].softmax(1).detach().cpu().numpy(),
                adjusted=adjusted
            )
            clf_result['incomplete'][data_type] = clf_result_

        return history, clf_result

    @torch.no_grad()
    def evaluate(self, eval_loader, adjusted=False):

        # Set network
        self._set_learning_phase(train=False)

        # Logging
        steps = len(eval_loader)
        history = {'loss': torch.zeros(steps, device=self.local_rank),
                   'ce_mri': torch.zeros(steps, device=self.local_rank),
                   'ce_pet': torch.zeros(steps, device=self.local_rank),
                   'ce_common': torch.zeros(steps, device=self.local_rank),
                   'mse': torch.zeros(steps, device=self.local_rank)}

        with get_rich_pbar(transient=True, auto_refresh=False) as pg:

            if self.local_rank == 0:
                task = pg.add_task(f"[bold red] Evaluating...", total=steps)

            y_true = {
                'complete': [],
                'incomplete': []
            }
            y_pred = {
                'complete': {'mri': [], 'pet': [], 'common': []},
                'incomplete': {'mri': [], 'common': []}
            }

            for i, batch in enumerate(eval_loader):

                # complete & incomplete step
                with torch.cuda.amp.autocast(self.mixed_precision):

                    result_complete = self.complete_step(batch=batch)
                    result_incomplete = self.incomplete_step(batch=batch)
                    del batch

                    # sum and divide by valid number of observations
                    n_complete, n_incomplete = len(result_complete['y_true']), len(result_incomplete['y_true'])

                    # loss
                    ce_mri = result_complete['ce']['mri'].sum() + result_incomplete['ce']['mri'].sum()
                    ce_pet = result_complete['ce']['pet'].sum()
                    ce_common = result_complete['ce']['common'].sum() + result_incomplete['ce']['common']
                    mse = result_complete['mse']

                    ce_mri = ce_mri / (n_complete + n_incomplete)
                    ce_pet = ce_pet / n_complete
                    ce_common = ce_common / (n_complete + n_incomplete)
                    mse = mse / n_complete

                    loss = (ce_mri + ce_pet + ce_common) + self.alpha * mse
                    loss = loss / (3 + self.alpha)

                # save monitoring values
                history['loss'][i] = loss.item()
                history['ce_mri'][i] = ce_mri.item()
                history['ce_pet'][i] = ce_pet.item()
                history['ce_common'][i] = ce_common.item()
                history['mse'][i] = mse

                if self.local_rank == 0:
                    pg.update(task, advance=1.)
                    pg.refresh()

                # save logits and y_true for evaluation
                y_true['complete'].append(result_complete['y_true'])
                y_true['incomplete'].append(result_incomplete['y_true'])

                y_pred['complete']['mri'].append(result_complete['y_pred']['mri'])
                y_pred['complete']['pet'].append(result_complete['y_pred']['pet'])
                y_pred['complete']['common'].append(result_complete['y_pred']['common'])
                y_pred['incomplete']['mri'].append(result_incomplete['y_pred']['mri'])
                y_pred['incomplete']['common'].append(result_incomplete['y_pred']['common'])

        history = {k: v.mean().item() for k, v in history.items()}

        # enforce to float32: accuracy and macro f1 score
        y_true['complete'] = torch.cat(y_true['complete'], dim=0)
        y_true['incomplete'] = torch.cat(y_true['incomplete'], dim=0)

        y_pred['complete']['mri'] = torch.cat(y_pred['complete']['mri'], dim=0).to(torch.float32)
        y_pred['complete']['pet'] = torch.cat(y_pred['complete']['pet'], dim=0).to(torch.float32)
        y_pred['complete']['common'] = torch.cat(y_pred['complete']['common'], dim=0).to(torch.float32)
        y_pred['incomplete']['mri'] = torch.cat(y_pred['incomplete']['mri'], dim=0).to(torch.float32)
        y_pred['incomplete']['common'] = torch.cat(y_pred['incomplete']['common'], dim=0).to(torch.float32)

        # classification result
        clf_result = {}
        clf_result['complete'] = {}
        clf_result['incomplete'] = {}

        for data_type in ['mri', 'pet', 'common']:
            clf_result_ = classification_result(
                y_true=y_true['complete'].cpu().numpy(),
                y_pred=y_pred['complete'][data_type].softmax(1).detach().cpu().numpy(),
                adjusted=adjusted
            )
            clf_result['complete'][data_type] = clf_result_

        for data_type in ['mri', 'common']:
            clf_result_ = classification_result(
                y_true=y_true['incomplete'].cpu().numpy(),
                y_pred=y_pred['incomplete'][data_type].softmax(1).detach().cpu().numpy(),
                adjusted=adjusted
            )
            clf_result['incomplete'][data_type] = clf_result_

        return history, clf_result

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


if __name__ == '__main__':

    import torch
    res = {}
    out1 = {'true': torch.tensor([1, 2, 3]),
            'ce': {'mri': torch.tensor([0.1, 0.2, 0.3]),
                   'pet': torch.tensor([0.05, 0.07, 0.01])}}


    out2 = {'true': torch.tensor([1, 2, 5]),
            'ce': {'mri': torch.tensor([0.11, 0.22, 0.33]),
                   'pet': torch.tensor([0.05, 0.07, 0.15])}}

    out3 = {'true': torch.tensor([3, 3, 8]),
            'ce': {'mri': torch.tensor([0.21, 0.26, 0.31]),
                   'pet': torch.tensor([0.15, 0.17, 0.18])}}

    for j, out in enumerate([out1, out2, out3]):
        # update result dictionary (maximum nesting depth = 2)
        if j == 0:
            res.update(out)
        else:
            for key1, value1 in out.items():
                if isinstance(value1, dict):
                    for key2, value2 in value1.items():
                        assert not isinstance(value2, dict)
                        res[key1][key2] = torch.concat([res[key1][key2], value2], dim=0)
                else:
                    res[key1] = torch.concat([res[key1], value1], dim=0)

    ## for multiple iterations
    # result_incomplete = {}
    # for j in range(self.incomplete_ratio):
    #
    #     try:
    #         batch_incomplete = next(incomplete_iter)
    #     except StopIteration:
    #         incomplete_iter = incomplete_loader.__iter__()
    #         batch_incomplete = next(incomplete_iter)
    #
    #     result_incomplete_ = self.incomplete_step(batch=batch_incomplete)
    #
    #     # update result dictionary (maximum nesting depth = 2)
    #     if j == 0:
    #         result_incomplete.update(result_incomplete_)
    #     else:
    #         for key1, value1 in result_incomplete_.items():
    #             if isinstance(value1, dict):
    #                 for key2, value2 in value1.items():
    #                     assert not isinstance(value2, dict)
    #                     result_incomplete_[key1][key2] = \
    #                         torch.concat([result_incomplete[key1][key2], value2], dim=0)
    #             else:
    #                 result_incomplete[key1] = torch.concat([result_incomplete[key1], value1], dim=0)
