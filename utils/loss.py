import torch
import torch.nn as nn
import torch.nn.functional as F


class DiffLoss(nn.Module):

    def __init__(self):
        super(DiffLoss, self).__init__()

    def forward(self, input1: torch.Tensor, input2: torch.Tensor):
        batch_size = input1.size(0)
        input1 = input1.view(batch_size, -1)
        input2 = input2.view(batch_size, -1)

        # Zero mean
        input1_mean = torch.mean(input1, dim=0, keepdims=True)
        input2_mean = torch.mean(input2, dim=0, keepdims=True)
        input1 = input1 - input1_mean
        input2 = input2 - input2_mean

        input1_l2_norm = torch.norm(input1, p=2, dim=1, keepdim=True).detach()
        input1_l2 = input1.div(input1_l2_norm.expand_as(input1) + 1e-6)

        input2_l2_norm = torch.norm(input2, p=2, dim=1, keepdim=True).detach()
        input2_l2 = input2.div(input2_l2_norm.expand_as(input2) + 1e-6)

        diff_loss = torch.mean((input1_l2.t().mm(input2_l2)).pow(2))

        return diff_loss


class CosineLoss(nn.Module):
    def __init__(self):
        super(CosineLoss, self).__init__()

    def forward(self, x1: torch.Tensor, x2: torch.Tensor):
        # x1 and x2 are L2-normalized
        cos = F.cosine_similarity(x1, x2, dim=1)
        loss = (1 - cos) / 2
        return loss.mean()


class CMDLoss(nn.Module):
    """
    Adapted from https://github.com/wzell/cmd/blob/master/models/domain_regularizer.py
    """
    def __init__(self, n_moments: int = 5):
        super(CMDLoss, self).__init__()
        self.n_moments = n_moments

    def forward(self, x1: torch.Tensor, x2: torch.Tensor):
        mx1 = torch.mean(x1, 0)
        mx2 = torch.mean(x2, 0)
        sx1 = x1 - mx1
        sx2 = x2 - mx2
        dm = self.matchnorm(mx1, mx2)
        scms = dm
        for i in range(self.n_moments - 1):
            scms += self.scm(sx1, sx2, i + 2)
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


if __name__ == '__main__':

    z_1 = torch.rand(size=(10, 2))
    z_2 = torch.rand(size=(10, 2))

    criterion_cmd = CMDLoss(n_moments=5)
    criterion_cosine = CosineLoss()
    criterion_diff = DiffLoss()

    loss_cmd = criterion_cmd(z_1, z_2)
    loss_cosine = criterion_cosine(z_1, z_2)
    loss_diff = criterion_diff(z_1, z_2)
