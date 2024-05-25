import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from .backbone.mobilenetv2 import mobilenet_v2
from .block.fpn import FPN
from .block.vertical import VerticalFusion
from .block.convs import ConvBnRelu, DsBnRelu
from .util import init_method
from .block.heads import FCNHead, GatedResidualUpHead
from .backbone.lwganet import LWGANet_L2_1442_e96_k11_ReLU, LWGANet_L0_1242_e32_k11_GELU


def get_backbone(backbone_name, dropout_rate):
    if backbone_name == 'mobilenetv2':
        backbone = mobilenet_v2(pretrained=True, progress=True)
        backbone.channels = [16, 24, 32, 96, 320]
    elif backbone_name == 'resnet18d':
        backbone = timm.create_model('resnet18d', pretrained=True, features_only=True)
        backbone.channels = [64, 64, 128, 256, 512]
    elif backbone_name == 'lwganet_l2':
        backbone = LWGANet_L2_1442_e96_k11_ReLU(pretrained=True, drop_path_rate=dropout_rate)
        backbone.channels = [96, 192, 384, 768]
    elif backbone_name == 'lwganet_l0':
        backbone = LWGANet_L0_1242_e32_k11_GELU(pretrained=True, drop_path_rate=dropout_rate)
        backbone.channels = [32, 64, 128, 256]
    else:
        raise NotImplementedError("BACKBONE [%s] is not implemented!\n" % backbone_name)
    return backbone


def get_fpn(fpn_name, in_channels, out_channels, deform_groups=4, gamma_mode='SE', beta_mode='contextgatedconv'):
    if fpn_name == 'fpn':
        fpn = FPN(in_channels, out_channels, deform_groups, gamma_mode, beta_mode)
    else:
        raise NotImplementedError("FPN [%s] is not implemented!\n" % fpn_name)
    return fpn


class Detector(nn.Module):
    def __init__(self, backbone_name='mobilenetv2', fpn_name='fpn', fpn_channels=128,
                 deform_groups=4, gamma_mode='SE', beta_mode='contextgatedconv',
                 num_heads=1, num_points=8, kernel_layers=1, dropout_rate=0.1, init_type='kaiming_normal'):
        super().__init__()
        self.backbone = get_backbone(backbone_name, dropout_rate)
        self.backbone_name = backbone_name
        self.fpn = get_fpn(fpn_name, in_channels=self.backbone.channels[-4:], out_channels=fpn_channels,
                           deform_groups=deform_groups, gamma_mode=gamma_mode, beta_mode=beta_mode)
        self.p5_to_p4 = VerticalFusion(fpn_channels, num_heads=num_heads, num_points=4,
                                                    kernel_layers=kernel_layers)
        self.p4_to_p3 = VerticalFusion(fpn_channels, num_heads=num_heads, num_points=8,
                                                    kernel_layers=kernel_layers)
        self.p3_to_p2 = VerticalFusion(fpn_channels, num_heads=num_heads, num_points=16,
                                                    kernel_layers=kernel_layers)

        self.p5_head = nn.Conv2d(fpn_channels, 2, 1)
        self.p4_head = nn.Conv2d(fpn_channels, 2, 1)
        self.p3_head = nn.Conv2d(fpn_channels, 2, 1)
        self.p2_head = nn.Conv2d(fpn_channels, 2, 1)
        self.project = nn.Sequential(nn.Conv2d(fpn_channels*4, fpn_channels, 1, bias=False),
                                     nn.BatchNorm2d(fpn_channels),
                                     nn.ReLU(True)
                                     )
        self.head = GatedResidualUpHead(fpn_channels, 2, dropout_rate=dropout_rate)
        # init_method(self.fpn, self.p5_to_p4, self.p4_to_p3, self.p3_to_p2, self.p5_head, self.p4_head,
        #             self.p3_head, self.p2_head, init_type=init_type)

    def forward(self, x1, x2):
        ### Extract backbone features
        if self.backbone_name == 'lwganet_l0' or self.backbone_name == 'lwganet_l2':
            t1_c2, t1_c3, t1_c4, t1_c5 = self.backbone.forward(x1)
            t2_c2, t2_c3, t2_c4, t2_c5 = self.backbone.forward(x2)
        else:
            t1_c1, t1_c2, t1_c3, t1_c4, t1_c5 = self.backbone.forward(x1)
            t2_c1, t2_c2, t2_c3, t2_c4, t2_c5 = self.backbone.forward(x2)
        t1_p2, t1_p3, t1_p4, t1_p5 = self.fpn([t1_c2, t1_c3, t1_c4, t1_c5])
        t2_p2, t2_p3, t2_p4, t2_p5 = self.fpn([t2_c2, t2_c3, t2_c4, t2_c5])

        diff_p2 = torch.abs(t1_p2 - t2_p2)
        diff_p3 = torch.abs(t1_p3 - t2_p3)
        diff_p4 = torch.abs(t1_p4 - t2_p4)
        diff_p5 = torch.abs(t1_p5 - t2_p5)

        fea_p5 = diff_p5
        pred_p5 = self.p5_head(fea_p5)
        fea_p4 = self.p5_to_p4(fea_p5, diff_p4)
        pred_p4 = self.p4_head(fea_p4)
        fea_p3 = self.p4_to_p3(fea_p4, diff_p3)
        pred_p3 = self.p3_head(fea_p3)
        fea_p2 = self.p3_to_p2(fea_p3, diff_p2)
        pred_p2 = self.p2_head(fea_p2)
        pred = self.head(fea_p2)

        pred_p2 = F.interpolate(pred_p2, size=(256, 256), mode='bilinear', align_corners=False)
        pred_p3 = F.interpolate(pred_p3, size=(256, 256), mode='bilinear', align_corners=False)
        pred_p4 = F.interpolate(pred_p4, size=(256, 256), mode='bilinear', align_corners=False)
        pred_p5 = F.interpolate(pred_p5, size=(256, 256), mode='bilinear', align_corners=False)

        return pred, pred_p2, pred_p3, pred_p4, pred_p5



