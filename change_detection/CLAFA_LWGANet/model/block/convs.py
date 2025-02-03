"""
https://github.com/iduta/pyconv
https://github.com/XudongLinthu/context-gated-convolution
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ConvBnRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1):
        super(ConvBnRelu, self).__init__()
        self.block = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                                             padding=padding, dilation=dilation, bias=False),
                                   nn.BatchNorm2d(out_channels),
                                   nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.block(x)
        return x


class DsBnRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1):
        super(DsBnRelu, self).__init__()
        self.kernel_size = kernel_size
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding,
                                   dilation, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(True)

    def forward(self, x):
        if self.kernel_size != 1:
            x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class PyConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, pyconv_kernels=[1, 3, 5, 7], pyconv_groups=[1, 2, 4, 8], bias=False):
        super(PyConv2d, self).__init__()

        pyconv_levels = []
        for pyconv_kernel, pyconv_group in zip(pyconv_kernels, pyconv_groups):
            pyconv_levels.append(nn.Conv2d(in_channels, out_channels // 2, kernel_size=pyconv_kernel,
                                           padding=pyconv_kernel // 2, groups=pyconv_group, bias=bias))
        self.pyconv_levels = nn.Sequential(*pyconv_levels)
        self.to_out = nn.Sequential(nn.Conv2d((out_channels // 2) * len(pyconv_kernels), out_channels, 1, bias=False),
                                    nn.BatchNorm2d(out_channels),
                                    nn.ReLU(True))
        self.relu = nn.ReLU(True)

    def forward(self, x):
        out = []
        for level in self.pyconv_levels:
            out.append(self.relu(level(x)))
        out = torch.cat(out, dim=1)
        out = self.to_out(out)

        return out


class GatedConv2d(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True):
        super(GatedConv2d, self).__init__()
        self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.mask_conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.sigmoid = torch.nn.Sigmoid()

    def gated(self, mask):
        return self.sigmoid(mask)

    def forward(self, input):
        x = self.conv2d(input)
        mask = self.mask_conv2d(input)
        x = x * self.gated(mask)

        return x


class ContextGatedConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=True):
        super(ContextGatedConv2d, self).__init__(in_channels, out_channels, kernel_size, stride,
                                                 padding, dilation, groups, bias)
        # for convolutional layers with a kernel size of 1, just use traditional convolution
        if kernel_size == 1:
            self.ind = True
        else:
            self.ind = False
            self.oc = out_channels
            self.ks = kernel_size

            # the target spatial size of the pooling layer
            ws = kernel_size
            self.avg_pool = nn.AdaptiveAvgPool2d((ws, ws))

            # the dimension of the latent repsentation
            self.num_lat = int((kernel_size * kernel_size) / 2 + 1)

            # the context encoding module
            self.ce = nn.Linear(ws * ws, self.num_lat, False)
            self.ce_bn = nn.BatchNorm1d(in_channels)
            self.ci_bn2 = nn.BatchNorm1d(in_channels)

            # activation function is relu
            self.act = nn.ReLU(inplace=True)

            # the number of groups in the channel interacting module
            if in_channels // 16:
                self.g = 16
            else:
                self.g = in_channels
            # the channel interacting module
            self.ci = nn.Linear(self.g, out_channels // (in_channels // self.g), bias=False)
            self.ci_bn = nn.BatchNorm1d(out_channels)

            # the gate decoding module
            self.gd = nn.Linear(self.num_lat, kernel_size * kernel_size, False)
            self.gd2 = nn.Linear(self.num_lat, kernel_size * kernel_size, False)

            # used to prrepare the input feature map to patches
            self.unfold = nn.Unfold(kernel_size, dilation, padding, stride)

            # sigmoid function
            self.sig = nn.Sigmoid()

    def forward(self, x):
        # for convolutional layers with a kernel size of 1, just use traditional convolution
        if self.ind:
            return F.conv2d(x, self.weight, self.bias, self.stride,
                            self.padding, self.dilation, self.groups)
        else:
            b, c, h, w = x.size()
            weight = self.weight
            # allocate glbal information
            gl = self.avg_pool(x).view(b, c, -1)
            # context-encoding module
            out = self.ce(gl)
            # use different bn for the following two branches
            ce2 = out
            out = self.ce_bn(out)
            out = self.act(out)
            # gate decoding branch 1
            out = self.gd(out)
            # channel interacting module
            if self.g > 3:
                # grouped linear
                oc = self.ci(self.act(self.ci_bn2(ce2).view(b, c // self.g, self.g, -1).transpose(2, 3))).transpose(2, 3).contiguous()
            else:
                # linear layer for resnet.conv1
                oc = self.ci(self.act(self.ci_bn2(ce2).transpose(2, 1))).transpose(2, 1).contiguous()
            oc = oc.view(b, self.oc, -1)
            oc = self.ci_bn(oc)
            oc = self.act(oc)
            # gate decoding branch 2
            oc = self.gd2(oc)
            # produce gate
            out = self.sig(out.view(b, 1, c, self.ks, self.ks) + oc.view(b, self.oc, 1, self.ks, self.ks))
            # unfolding input feature map to patches
            x_un = self.unfold(x)
            b, _, l = x_un.size()
            # gating
            out = (out * weight.unsqueeze(0)).view(b, self.oc, -1)
            # currently only handle square input and output
            return torch.matmul(out, x_un).view(b, self.oc, int(np.sqrt(l)), int(np.sqrt(l)))


