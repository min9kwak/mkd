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


class L2Loss(nn.Module):
    def __init__(self):
        super(L2Loss, self).__init__()

    def forward(self, x1: torch.Tensor, x2: torch.Tensor):
        # x1 and x2 are L2-normalized
        loss = torch.norm(u_norm - v_norm, p=2, dim=1)
        return loss.mean()


class MSELoss(nn.Module):
    def __init__(self):
        super(MSELoss, self).__init__()

    def forward(self, x1: torch.Tensor, x2: torch.Tensor):
        # x1 and x2 are L2-normalized
        loss = torch.norm(u_norm - v_norm, p=2, dim=1)
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

    z1 = torch.rand(size=(16, 64))
    z2 = torch.rand(size=(16, 64))

    z1 = F.normalize(z1, p=2, dim=1)
    z2 = F.normalize(z2, p=2, dim=1)

    # F.cosine_similarity
    loss_cosine = F.cosine_similarity(z1, z2, dim=1)
    loss_cosine.mean()

    # einsum
    loss_einsum = torch.einsum('nc,nc->n', [z1, z2])


    #
    import torch

    # Random tensors for demonstration purposes
    u = torch.randn(size=(10, 64))
    v = torch.randn(size=(10, 64))

    # Normalize the vectors
    u_norm = F.normalize(u, p=2, dim=1)
    v_norm = F.normalize(v, p=2, dim=1)

    # Compute cosine similarity
    cos_sim = F.cosine_similarity(u_norm, v_norm)

    # Compute normalized L2 loss
    l2_loss = torch.norm(u_norm - v_norm, p=2, dim=1)

    # Compute MSE loss
    mse = F.mse_loss(u_norm, v_norm)

    # To show the relationship
    cos_sim_from_mse = 1 - mse

    print(f"Cosine Similarity: {cos_sim.mean()}")
    print(f"MSE loss: {mse}")
    print(f"Cosine Similarity computed from MSE loss: {cos_sim_from_mse}")

    # To show the relationship
    cos_sim_from_l2 = 1 - 0.5 * (l2_loss ** 2)
    l2_from_cos_sim = torch.sqrt((1 - cos_sim) * 2)

    print(f"Cosine Similarity: {cos_sim}")
    print(f"Normalized L2 loss: {l2_loss}")
    print(f"Cosine Similarity computed from L2 loss: {cos_sim_from_l2}")
    print(f"L2 loss computed from Cosine Similarity: {l2_from_cos_sim}")





    ###
    import torch
    from torch.nn.functional import cosine_similarity, normalize, mse_loss

    # Random tensors for demonstration purposes
    u = torch.randn(100, 64)
    v = torch.randn(100, 64)

    # Normalize the vectors
    u_norm = normalize(u, p=2, dim=1)
    v_norm = normalize(v, p=2, dim=1)

    # Compute cosine similarity
    cos_sim = cosine_similarity(u_norm, v_norm)

    # Compute MSE loss
    mse = mse_loss(u_norm, v_norm, reduction='none').mean(1)

    # Compute mean values
    mean_cos_sim = cos_sim.mean()
    mean_mse = mse.mean()

    # To show the relationship
    cos_sim_from_mean_mse = 1 - mean_mse

    print(f"Mean Cosine Similarity: {mean_cos_sim}")
    print(f"Mean MSE loss: {mean_mse}")
    print(f"Cosine Similarity computed from mean MSE loss: {cos_sim_from_mean_mse}")
