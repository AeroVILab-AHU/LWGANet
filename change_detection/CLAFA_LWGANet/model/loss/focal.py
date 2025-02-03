"""
https://github.com/clcarwin/focal_loss_pytorch/blob/master/focalloss.py
"""

import torch
import torch.nn.functional as F
import torch.utils.data
import torch.nn as nn
from torch.autograd import Variable
from einops import rearrange


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=4.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        if isinstance(alpha, (float, int)):
            self.alpha = torch.as_tensor([alpha, 1 - alpha])
        if isinstance(alpha, list):
            self.alpha = torch.as_tensor(alpha)

    def forward(self, input, target):
        N, C, H, W = input.size()
        assert C == 2
        # input = input.view(N, C, -1)
        # input = input.transpose(1, 2)
        # input = input.contiguous().view(-1, C)
        input = rearrange(input, 'b c h w -> (b h w) c')
        # input = input.contiguous().view(-1)

        target = target.view(-1, 1)
        logpt = F.log_softmax(input, dim=1)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * Variable(at)
        loss = -1 * (1-pt)**self.gamma * logpt

        return loss.mean()


