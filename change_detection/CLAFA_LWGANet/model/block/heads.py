"""
https://github.com/swz30/MIRNet
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .convs import GatedConv2d, ContextGatedConv2d


class GatedResidualUp(nn.Module):
    def __init__(self, in_channels, up_mode='conv', gate_mode='gated'):
        super(GatedResidualUp, self).__init__()
        if up_mode == 'conv':
            self.residual_up = nn.Sequential(nn.ConvTranspose2d(in_channels, in_channels, 3, stride=2, padding=1,
                                                            output_padding=1, bias=False),
                                         nn.BatchNorm2d(in_channels),
                                         nn.ReLU(True))
        elif up_mode == 'bilinear':
            self.residual_up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

        if gate_mode == 'gated':
            self.gate = GatedConv2d(in_channels, in_channels // 2)
        elif gate_mode == 'context_gated':
            self.gate = ContextGatedConv2d(in_channels, in_channels // 2)
        self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                                nn.Conv2d(in_channels, in_channels // 2, 1, stride=1, padding=0, bias=True)
                                )
        self.relu = nn.ReLU(True)

    def forward(self, x):
        residual = self.residual_up(x)
        residual = self.gate(residual)
        up = self.up(x)
        out = self.relu(up + residual)
        return out


class GatedResidualUpHead(nn.Module):
    def __init__(self, in_channels=128, num_classes=1, dropout_rate=0.15):
        super(GatedResidualUpHead, self).__init__()

        self.up = nn.Sequential(GatedResidualUp(in_channels),
                                GatedResidualUp(in_channels // 2))
        self.smooth = nn.Sequential(nn.Conv2d(in_channels // 4, in_channels // 4, kernel_size=3, stride=1, padding=1),
                                    nn.ReLU(True),
                                    nn.Dropout2d(dropout_rate))
        self.final = nn.Conv2d(in_channels // 4, num_classes, 1)

    def forward(self, x):
        x = self.up(x)
        x = self.smooth(x)
        x = self.final(x)

        return x


class FCNHead(nn.Module):
    def __init__(self, in_channels, num_classes, num_convs=1, dropout_rate=0.15):
        self.num_convs = num_convs
        super(FCNHead, self).__init__()
        inter_channels = in_channels // 4

        convs = []
        convs.append(nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                   nn.BatchNorm2d(inter_channels),
                                   nn.ReLU(True)))
        for i in range(num_convs - 1):
            convs.append(nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                       nn.BatchNorm2d(inter_channels),
                                       nn.ReLU(True)))
        self.convs = nn.Sequential(*convs)
        self.final = nn.Conv2d(inter_channels, num_classes, 1)

    def forward(self, x):
        out = self.convs(x)
        out = self.final(out)

        return out

