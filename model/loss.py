import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb


def cross_entropy(output, target, weight=None, is_train=True):
    if is_train:
        return F.cross_entropy(output, target, weight=weight, reduction='mean')
    else:
        target = target.float()
        if weight is not None:
            weighted_target = target * weight.unsqueeze(0)
        else:
            weighted_target = target
        output = torch.diagonal(F.log_softmax(output, 2), offset=0, dim1=1, dim2=2)
        return -torch.sum(output * weighted_target) / torch.sum(weighted_target)

