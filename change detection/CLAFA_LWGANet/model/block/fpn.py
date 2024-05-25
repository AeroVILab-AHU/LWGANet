import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from .dcnv2 import DCNv2
from .convs import GatedConv2d, ContextGatedConv2d


class GenerateGamma(nn.Module):
    def __init__(self, channels=128, mode='SE'):
        super(GenerateGamma, self).__init__()
        self.mode = mode
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(nn.Conv2d(channels, channels // 4, 1, bias=False),
                                nn.ReLU(True),
                                nn.Conv2d(channels // 4, channels, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        if self.mode == 'SE':
            return self.sigmoid(avg_out)
        elif self.mode == 'CBAM':
            max_out = self.fc(self.max_pool(x))
            out = avg_out + max_out
            return self.sigmoid(out)
        else:
            raise NotImplementedError


class GenerateBeta(nn.Module):
    def __init__(self, channels=128, mode='conv'):
        super(GenerateBeta, self).__init__()
        self.stem = nn.Sequential(nn.Conv2d(channels, channels, 3, padding=1, bias=True), nn.ReLU(True))
        if mode == 'conv':
            self.conv = nn.Conv2d(channels, channels, 3, padding=1, bias=True)
        elif mode == 'gatedconv':
            self.conv = GatedConv2d(channels, channels, 3, padding=1, bias=True)
        elif mode == 'contextgatedconv':
            self.conv = ContextGatedConv2d(channels, channels, 3, padding=1, bias=True)
        else:
            raise NotImplementedError

    def forward(self, x):
        x = self.stem(x)
        return self.conv(x)


### MoFPN
class FPN(nn.Module):
    def __init__(self, in_channels, out_channels=128, deform_groups=4, gamma_mode='SE', beta_mode='contextgatedconv'):
        super(FPN, self).__init__()

        self.p2 = DCNv2(in_channels=in_channels[0], out_channels=out_channels,
                        kernel_size=3, padding=1, deform_groups=deform_groups)
        self.p3 = DCNv2(in_channels=in_channels[1], out_channels=out_channels,
                        kernel_size=3, padding=1, deform_groups=deform_groups)
        self.p4 = DCNv2(in_channels=in_channels[2], out_channels=out_channels,
                        kernel_size=3, padding=1, deform_groups=deform_groups)
        self.p5 = DCNv2(in_channels=in_channels[3], out_channels=out_channels,
                        kernel_size=3, padding=1, deform_groups=deform_groups)

        self.p5_bn = nn.BatchNorm2d(out_channels, affine=True)
        self.p4_bn = nn.BatchNorm2d(out_channels, affine=False)
        self.p3_bn = nn.BatchNorm2d(out_channels, affine=False)
        self.p2_bn = nn.BatchNorm2d(out_channels, affine=False)
        self.activation = nn.ReLU(True)

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.p4_Gamma = GenerateGamma(out_channels, mode=gamma_mode)
        self.p4_beta = GenerateBeta(out_channels, mode=beta_mode)
        self.p3_Gamma = GenerateGamma(out_channels, mode=gamma_mode)
        self.p3_beta = GenerateBeta(out_channels, mode=beta_mode)
        self.p2_Gamma = GenerateGamma(out_channels, mode=gamma_mode)
        self.p2_beta = GenerateBeta(out_channels, mode=beta_mode)

        self.p5_smooth = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.p4_smooth = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.p3_smooth = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.p2_smooth = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
    def forward(self, input):
        c2, c3, c4, c5 = input

        p5 = self.activation(self.p5_bn(self.p5(c5)))
        p5_up = F.interpolate(p5, size=c4.shape[-2:], mode='bilinear', align_corners=False)
        p4 = self.p4_bn(self.p4(c4))
        p4_gamma, p4_beta = self.p4_Gamma(p5_up), self.p4_beta(p5_up)
        p4 = self.activation(p4 * (1 + p4_gamma) + p4_beta)
        p4_up = F.interpolate(p4, size=c3.shape[-2:], mode='bilinear', align_corners=False)
        p3 = self.p3_bn(self.p3(c3))
        p3_gamma, p3_beta = self.p3_Gamma(p4_up), self.p3_beta(p4_up)
        p3 = self.activation(p3 * (1 + p3_gamma) + p3_beta)
        p3_up = F.interpolate(p3, size=c2.shape[-2:], mode='bilinear', align_corners=False)
        p2 = self.p2_bn(self.p2(c2))
        p2_gamma, p2_beta = self.p2_Gamma(p3_up), self.p2_beta(p3_up)
        p2 = self.activation(p2 * (1 + p2_gamma) + p2_beta)

        p5 = self.p5_smooth(p5)
        p4 = self.p4_smooth(p4)
        p3 = self.p3_smooth(p3)
        p2 = self.p2_smooth(p2)

        return p2, p3, p4, p5


