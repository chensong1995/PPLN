import torch
import torch.nn as nn

import pdb

class WeightedL1Loss(nn.Module):
    def __init__(self, multiplier=5):
        super(WeightedL1Loss, self).__init__()
        self.criterion = torch.nn.L1Loss(reduction='none')
        self.multiplier = multiplier

    def forward(self, pred, gt, weight=None):
        loss = self.criterion(pred, gt)
        if weight is None:
            weight = gt
        loss = loss * torch.exp(self.multiplier * torch.abs(weight))
        loss = torch.mean(loss)
        return loss

class KLDLoss(nn.Module):
    def __init__(self):
        super(KLDLoss, self).__init__()

    def forward(self, mean, log_var):
        loss = - 0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
        return loss
