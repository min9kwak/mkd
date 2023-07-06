from easydict import EasyDict as edict
from utils.simulation import generate_data, build_general_teacher, MultiModalDataset
from torch.utils.data import DataLoader
from utils.optimization import get_optimizer
from utils.optimization import get_cosine_scheduler
from utils.metrics import classification_result

import torch
import torch.nn as nn
from utils.loss import SimCosineLoss, DiffCosineLoss

# configs
config = edict()

# data
config.n_train = 1000
config.missing_rate = 0.3
if config.missing_rate == -1:
    config.missing_rate = None
config.n_test = 1000
config.random_state = 2021
config.n_runs = 10

config.x1_dim = 100
config.x2_dim = 100
config.xs1_dim = 25
config.xs2_dim = 25
config.alpha = config.xs1_dim / (config.xs1_dim + config.xs2_dim)
config.gamma = 1 - config.alpha

# network
config.hidden = 25

# training
config.alpha_ce = 1.0
config.alpha_sim = 10.0
config.alpha_diff = 5.0
config.alpha_recon = 0.1

# optimizer
config.batch_size = 16
config.epochs = 100

config.optimizer = 'sgd'
config.lr = 0.1
config.weight_deacy = 0.0001

config.cosine_warmup = 0
config.cosine_cycles = 1
config.cosine_min_lr = 0.0

# environment
config.local_rank = 0

# run(seed, 'alg1', config.xs1_dim, config.xs2_dim, [1, 1], config.n_runs)

# Generate Gaussian Multi-Modal Data

data = generate_data(n_train=config.n_train, n_test=config.n_test, x1_dim=config.x1_dim, x2_dim=config.x2_dim,
                     x1_common_dim=config.xs1_dim, x2_common_dim=config.xs2_dim, y_dummy=500,
                     missing_rate=config.missing_rate, random_state=config.random_state)

dataset_complete = MultiModalDataset(x1=data['x1_train_complete'],
                                     x2=data['x2_train_complete'],
                                     y=data['y_train_complete'])

# 1. Train Teacher
loader_complete = DataLoader(dataset_complete, batch_size=config.batch_size, shuffle=True, drop_last=True)

teacher = build_general_teacher(config=config)
_ = [v.to(config.local_rank) for v in teacher.values()]

params = []
for v in teacher.values():
    params = params + [{'params': v.parameters()}]
optimizer = get_optimizer(params=params, name=config.optimizer, lr=config.lr, weight_decay=config.weight_deacy)
scheduler = get_cosine_scheduler(optimizer=optimizer, epochs=config.epochs, warmup_steps=config.cosine_warmup,
                                 cycles=config.cosine_cycles, min_lr=config.cosine_min_lr)

loss_function_ce = nn.CrossEntropyLoss(reduction='mean')
loss_function_sim = SimCosineLoss()
loss_function_diff = DiffCosineLoss()
loss_function_recon = nn.MSELoss(reduction='mean')


def train_step(batch):
    x1, x2 = batch['x1'].float().to(config.local_rank), batch['x2'].float().to(config.local_rank)
    y = batch['y'].long().to(config.local_rank)

    h1 = teacher['extractor_1'](x1)
    h2 = teacher['extractor_2'](x2)

    z1_general = teacher['encoder_general'](h1)
    z2_general = teacher['encoder_general'](h2)
    z1 = teacher['encoder_1'](h1)
    z2 = teacher['encoder_2'](h2)

    # difference
    loss_diff_specific = loss_function_diff(z1, z2)
    loss_diff_1 = loss_function_diff(z1, z1_general)
    loss_diff_2 = loss_function_diff(z2, z2_general)
    loss_diff = loss_diff_specific + loss_diff_1 + loss_diff_2

    # similarity
    loss_sim = loss_function_sim(z1_general, z2_general)

    # reconstruction
    h1_recon = teacher['decoder_1'](z1_general + z1)
    h2_recon = teacher['decoder_2'](z2_general + z2)

    loss_recon_1 = loss_function_recon(h1_recon, h1)
    loss_recon_2 = loss_function_recon(h2_recon, h2)
    loss_recon = loss_recon_1 + loss_recon_2

    # classification
    logit = teacher['classifier'](z1_general + z2_general)
    loss_ce = loss_function_ce(logit, y)

    # sum
    loss = config.alpha_ce * loss_ce + config.alpha_sim * loss_sim + \
           config.alpha_diff * loss_diff + config.alpha_recon * loss_recon

    return loss, loss_ce, loss_sim, loss_diff_specific, loss_diff_1, loss_diff_2, \
           loss_recon_1, loss_recon_2, y, logit


for epcoh in range(1, config.epochs + 1):

    # train
    for v in teacher.values():
        v.train()

    steps = len(loader_complete)
    result = {'total_loss': torch.zeros(steps, device=config.local_rank),
              'loss_ce': torch.zeros(steps, device=config.local_rank),
              'loss_sim': torch.zeros(steps, device=config.local_rank),
              'loss_diff_specific': torch.zeros(steps, device=config.local_rank),
              'loss_diff_mri': torch.zeros(steps, device=config.local_rank),
              'loss_diff_pet': torch.zeros(steps, device=config.local_rank),
              'loss_recon_mri': torch.zeros(steps, device=config.local_rank),
              'loss_recon_pet': torch.zeros(steps, device=config.local_rank), }

    y_true, y_pred = [], []
    for i, batch in enumerate(loader_complete):
        loss, loss_ce, loss_sim, loss_diff_specific, loss_diff_1, loss_diff_2, loss_recon_1, loss_recon_2, y, logit = \
            train_step(batch=batch)
        loss.backward()
        optimizer.step()

        # save monitoring values
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
                                       adjusted=False)

    for k, v in clf_result.items():
        result[k] = v

    scheduler.step()
