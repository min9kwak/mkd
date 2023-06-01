import torch
import torch.nn as nn
import torch.nn.functional as F


class DiffFroLoss(nn.Module):

    def __init__(self):
        super(DiffFroLoss, self).__init__()

    def forward(self, input1, input2):
        batch_size = input1.size(0)
        input1 = input1.view(batch_size, -1)
        input2 = input2.view(batch_size, -1)

        # Zero mean
        input1_l2_norm = torch.norm(input1, p=2, dim=1, keepdim=True).detach()
        input1_l2 = input1.div(input1_l2_norm.expand_as(input1) + 1e-6)

        input2_l2_norm = torch.norm(input2, p=2, dim=1, keepdim=True).detach()
        input2_l2 = input2.div(input2_l2_norm.expand_as(input2) + 1e-6)

        diff_loss = (input1_l2.t().mm(input2_l2)).mean()

        return diff_loss


class SimCosineLoss(nn.Module):
    def __init__(self):
        super(SimCosineLoss, self).__init__()

    def forward(self, x1: torch.Tensor, x2: torch.Tensor):
        # x1 and x2 are L2-normalized
        loss = torch.einsum('nc,nc->n', [x1, x2])
        loss = 1 - loss
        return loss.mean()


class DiffCosineLoss(nn.Module):
    def __init__(self):
        super(DiffCosineLoss, self).__init__()

    def forward(self, x1: torch.Tensor, x2: torch.Tensor):
        # x1 and x2 are L2-normalized
        loss = torch.einsum('nc,nc->n', [x1, x2])
        loss = 1 + loss
        return loss.mean()


class SimL2Loss(nn.Module):
    def __init__(self):
        super(SimL2Loss, self).__init__()

    def forward(self, x1: torch.Tensor, x2: torch.Tensor):
        # x1 and x2 are L2-normalized
        loss = torch.norm(x1 - x2, p=2, dim=1)
        return loss.mean()


class DiffMSELoss(nn.Module):
    def __init__(self):
        super(DiffMSELoss, self).__init__()

    def forward(self, x1: torch.Tensor, x2: torch.Tensor):
        # x1 and x2 are L2-normalized
        loss = F.mse_loss(x1, x2, reduction='mean')
        return -loss


class SimCMDLoss(nn.Module):
    """
    Adapted from https://github.com/wzell/cmd/blob/master/models/domain_regularizer.py
    """
    def __init__(self, n_moments: int = 5):
        super(SimCMDLoss, self).__init__()
        self.n_moments = n_moments

    def forward(self, x1: torch.Tensor, x2: torch.Tensor):
        scms = self.matchnorm(x1, x2)
        for i in range(self.n_moments - 1):
            scms += self.scm(x1, x2, i + 2)
        return scms

    def matchnorm(self, x1, x2):
        power = torch.pow(x1-x2, 2)
        summed = torch.sum(power)
        sqrt = summed ** 0.5
        return sqrt

    def scm(self, sx1, sx2, k):
        ss1 = torch.mean(torch.pow(sx1, k), 0)
        ss2 = torch.mean(torch.pow(sx2, k), 0)
        return self.matchnorm(ss1, ss2)
