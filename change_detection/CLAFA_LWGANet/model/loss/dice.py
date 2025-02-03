import torch
import torch.nn.functional as F
import torch.utils.data
import torch.nn as nn
from kornia.losses import dice_loss

class DICELoss(nn.Module):
    def __init__(self):
        super(DICELoss, self).__init__()

    def forward(self, input, target):
        target = target.squeeze(1)
        loss = dice_loss(input, target)

        return loss

### version 2
"""
class DICELoss(nn.Module):
    def __init__(self, eps=1e-5):
        super(DICELoss, self).__init__()
        self.eps = eps

    def to_one_hot(self, target):
        N, C, H, W = target.size()
        assert C == 1
        target = torch.zeros(N, 2, H, W).to(target.device).scatter_(1, target, 1)
        return target

    def forward(self, input, target):
        N, C, _, _ = input.size()
        input = F.softmax(input, dim=1)

        #target = self.to_one_hot(target)
        target = torch.eye(2)[target.squeeze(1)]
        target = target.permute(0, 3, 1, 2).type_as(input)

        dims = tuple(range(1, target.ndimension()))
        inter = torch.sum(input * target, dims)
        cardinality = torch.sum(input + target, dims)
        loss = ((2. * inter) / (cardinality + self.eps)).mean()

        return 1 - loss
"""
