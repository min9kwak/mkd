import torch
import torch.nn as nn
import torch.nn.functional as F


class SemiBatchNorm3d(nn.BatchNorm3d):
    def __init__(self, channels):
        super().__init__(channels)
        self.update_batch_stats = True

    def forward(self, x):
        if self.update_batch_stats:
            return super().forward(x)
        else:
            return nn.functional.batch_norm(
                x, None, None, self.weight, self.bias, True, self.momentum, self.eps
            )


class PiLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y, backbone, classifier, mask):
        update_batch_stats(backbone, False)
        y_pred = classifier(backbone(x))
        update_batch_stats(backbone, True)
        loss = F.mse_loss(y_pred.softmax(1), y.softmax(1).detach(), reduction="none").mean(1)
        loss = loss * mask
        loss = loss.mean()
        return loss


class PseduoLabelLoss(nn.Module):
    def __init__(self, threshold, num_classes):
        super().__init__()
        self.th = threshold
        self.num_classes = num_classes

    def forward(self, x, y, backbone, classifier, mask):
        y_pred = y.softmax(1)
        onehot_label = self.__make_one_hot(y_pred.max(1)[1]).float()
        gt_mask = (y_pred > self.th).float()
        gt_mask = gt_mask.max(1)[0]  # reduce_any
        lt_mask = 1 - gt_mask  # logical not
        p_target = gt_mask[:, None] * self.num_classes * onehot_label + lt_mask[:, None] * y_pred
        update_batch_stats(backbone, False)
        output = classifier(backbone(x))
        update_batch_stats(backbone, True)
        loss = (-(p_target.detach() * F.log_softmax(output, 1)).sum(1) * mask).mean()
        return loss

    def __make_one_hot(self, y):
        return torch.eye(self.num_classes)[y].to(y.device)


def update_batch_stats(module, flag):
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm3d):
            child.update_batch_stats = flag
    for n, ch in module.named_children():
        update_batch_stats(ch, flag)


def replace_bn(module, name):
    for attr_str in dir(module):
        target_attr = getattr(module, attr_str)
        if type(target_attr) == torch.nn.BatchNorm3d:
            print('replaced: ', name, attr_str)
            setattr(module, attr_str, SemiBatchNorm3d(target_attr.num_features))
    for n, ch in module.named_children():
        replace_bn(ch, n)
