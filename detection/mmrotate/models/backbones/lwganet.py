# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import torch
import torch.nn as nn
from timm.models.layers import DropPath, trunc_normal_
from typing import List
from torch import Tensor
import os
import copy
import antialiased_cnns
import torch.nn.functional as F
from typing import Tuple
from mmcv.cnn import build_norm_layer
from ..builder import ROTATED_BACKBONES


try:
    from mmdet.utils import get_root_logger
    from mmcv.runner import _load_checkpoint
    has_mmdet = True
except ImportError:
    print("If for detection, please install mmdetection first")
    has_mmdet = False

class DRFD(nn.Module):
    def __init__(self, dim, act_layer):
        super().__init__()
        self.dim = dim
        self.outdim = dim * 2
        self.conv = nn.Conv2d(dim, dim*2, kernel_size=3, stride=1, padding=1, groups=dim)
        self.conv_c = nn.Conv2d(dim*2, dim*2, kernel_size=3, stride=2, padding=1, groups=dim*2)
        self.act_c = act_layer()
        self.batch_norm_c = nn.BatchNorm2d(dim*2)
        self.max_m = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.batch_norm_m = nn.BatchNorm2d(dim*2)
        self.fusion = nn.Conv2d(dim*4, self.outdim, kernel_size=1, stride=1)

    def forward(self, x):  # x = [B, C, H, W]

        # Gconv
        x = self.conv(x)  # x = [B, 2C, H, W]

        # 最大池化下采样
        max = self.batch_norm_m(self.max_m(x))                  # m = [B, 2C, H/2, W/2]
        # 卷积下采样
        conv = self.batch_norm_c(self.act_c(self.conv_c(x)))    # c = [B, 2C, H/2, W/2]

        # 拼接
        x = torch.cat([conv, max], dim=1)                       # x = [B, 2C+2C, H/2, W/2]  -->  [B, 4C, H/2, W/2]
        x = self.fusion(x)                                      # x = [B, 4C, H/2, W/2]     -->  [B, 2C, H/2, W/2]

        return x


class MRA(nn.Module):
    def __init__(self, channel, att_kernel):
        super().__init__()
        att_padding = att_kernel // 2
        self.gate_fn = nn.Sigmoid()
        self.channel = channel
        self.max_m1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.max_m2 = antialiased_cnns.BlurPool(channel, stride=3)
        self.H_att1 = nn.Conv2d(channel, channel, (att_kernel, 3), 1, (att_padding, 1), groups=channel, bias=False)
        self.V_att1 = nn.Conv2d(channel, channel, (3, att_kernel), 1, (1, att_padding), groups=channel, bias=False)
        self.H_att2 = nn.Conv2d(channel, channel, (att_kernel, 3), 1, (att_padding, 1), groups=channel, bias=False)
        self.V_att2 = nn.Conv2d(channel, channel, (3, att_kernel), 1, (1, att_padding), groups=channel, bias=False)
        self.batchnorm = nn.BatchNorm2d(channel)

    def forward(self, x):
        _, _, h, w = x.shape
        x_tem = self.max_m1(x)
        x_tem = self.max_m2(x_tem)
        x_h1 = self.H_att1(x_tem)
        x_w1 = self.V_att1(x_tem)
        if h == w:
            x_h2 = self.inv_h_transform(self.H_att2(self.h_transform(x_tem)))
            x_w2 = self.inv_v_transform(self.V_att2(self.v_transform(x_tem)))
            att = self.batchnorm(x_h1 + x_w1 + x_h2 + x_w2)
        else:
            att = self.batchnorm(x_h1 + x_w1)

        out = x[:, :self.channel, :, :] * F.interpolate(self.gate_fn(att),
                                                        size=(x.shape[-2], x.shape[-1]),
                                                        mode='nearest')
        return out

    def h_transform(self, x):
        shape = x.size()
        x = torch.nn.functional.pad(x, (0, shape[-1]))
        x = x.reshape(shape[0], shape[1], -1)[..., :-shape[-1]]
        x = x.reshape(shape[0], shape[1], shape[2], 2*shape[3]-1)
        return x

    def inv_h_transform(self, x):
        shape = x.size()
        x = x.reshape(shape[0], shape[1], -1).contiguous()
        x = torch.nn.functional.pad(x, (0, shape[-2]))
        x = x.reshape(shape[0], shape[1], shape[-2], 2*shape[-2])
        x = x[..., 0: shape[-2]]
        return x

    def v_transform(self, x):
        x = x.permute(0, 1, 3, 2)
        shape = x.size()
        x = torch.nn.functional.pad(x, (0, shape[-1]))
        x = x.reshape(shape[0], shape[1], -1)[..., :-shape[-1]]
        x = x.reshape(shape[0], shape[1], shape[2], 2*shape[3]-1)
        return x.permute(0, 1, 3, 2)

    def inv_v_transform(self, x):
        x = x.permute(0, 1, 3, 2)
        shape = x.size()
        x = x.reshape(shape[0], shape[1], -1)
        x = torch.nn.functional.pad(x, (0, shape[-2]))
        x = x.reshape(shape[0], shape[1], shape[-2], 2*shape[-2])
        x = x[..., 0: shape[-2]]
        return x.permute(0, 1, 3, 2)


class D_GA(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.norm = nn.BatchNorm2d(dim)
        self.attn = GA(dim)
        self.downpool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.uppool = nn.MaxUnpool2d((2, 2), 2, padding=0)

    def forward(self, x):
        x_, idx = self.downpool(x)
        x = self.norm(self.attn(x_))
        x = self.uppool(x, indices=idx)

        return x


class L_Conv(nn.Module):
    def __init__(self, dim, act_layer):
        super().__init__()
        self.downpool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.uppool = nn.MaxUnpool2d((2, 2), 2, padding=0)
        self.proj_1 = nn.Conv2d(dim, dim, 1)
        self.activation = act_layer()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv_spatial = nn.Conv2d(dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)
        self.conv1 = nn.Conv2d(dim, dim // 2, 1)
        self.conv2 = nn.Conv2d(dim, dim // 2, 1)
        self.conv_squeeze = nn.Conv2d(2, 2, 7, padding=3)
        self.conv = nn.Conv2d(dim // 2, dim, 1)
        self.proj_2 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        x_, idx = self.downpool(x)
        x_ = self.proj_1(x_)
        x_ = self.activation(x_)
        attn1 = self.conv0(x_)
        attn2 = self.conv_spatial(attn1)

        attn1 = self.conv1(attn1)
        attn2 = self.conv2(attn2)

        attn = torch.cat([attn1, attn2], dim=1)
        avg_attn = torch.mean(attn, dim=1, keepdim=True)
        max_attn, _ = torch.max(attn, dim=1, keepdim=True)
        agg = torch.cat([avg_attn, max_attn], dim=1)
        sig = self.conv_squeeze(agg).sigmoid()
        attn = attn1 * sig[:, 0, :, :].unsqueeze(1) + attn2 * sig[:, 1, :, :].unsqueeze(1)
        attn = self.conv(attn)
        x_ = x_ * attn
        x_ = self.proj_2(x_)
        x = self.uppool(x_, indices=idx)
        return x


class GA(nn.Module):
    def __init__(self, dim, head_dim=4, num_heads=None, qkv_bias=False,
                 attn_drop=0., proj_drop=0., proj_bias=False, **kwargs):
        super().__init__()

        self.head_dim = head_dim
        self.scale = head_dim ** -0.5

        self.num_heads = num_heads if num_heads else dim // head_dim
        if self.num_heads == 0:
            self.num_heads = 1

        self.attention_dim = self.num_heads * self.head_dim
        self.qkv = nn.Linear(dim, self.attention_dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(self.attention_dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1)
        N = H * W
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, H, W, self.attention_dim)
        x = self.proj(x)
        x = self.proj_drop(x)
        x = x.permute(0, 3, 1, 2)
        return x


class P_Conv(nn.Module):
    def __init__(self, dim, act_layer):
        super().__init__()
        self.p_conv = nn.Sequential(
            nn.Conv2d(dim, dim*4, 1, bias=False),
            nn.BatchNorm2d(dim*4),
            act_layer(),
            nn.Conv2d(dim*4, dim, 1, bias=False)
        )
        self.gate_fn = nn.Sigmoid()

    def forward(self, x):
        att = self.p_conv(x)
        x = x * self.gate_fn(att)

        return x


class Conv(nn.Module):
    def __init__(self, dim, act_layer):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False),
            nn.BatchNorm2d(dim),
            act_layer()
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class MBFD(nn.Module):
    def __init__(self,
                 dim,
                 stage,
                 att_kernel,
                 mlp_ratio,
                 drop_path,
                 act_layer,
                 norm_layer
                 ):
        super().__init__()
        self.dim = dim
        self.stage = stage
        self.dim_learn = dim // 4
        self.dim_untouched = dim - self.dim_learn - self.dim_learn - self.dim_learn
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        mlp_hidden_dim = int(dim * mlp_ratio)

        mlp_layer: List[nn.Module] = [
            nn.Conv2d(dim, mlp_hidden_dim, 1, bias=False),
            norm_layer(mlp_hidden_dim),
            act_layer(),
            nn.Conv2d(mlp_hidden_dim, dim, 1, bias=False)
        ]

        self.mlp = nn.Sequential(*mlp_layer)

        self.P_Conv = P_Conv(self.dim_learn, act_layer)
        self.Conv = Conv(self.dim_learn, act_layer)
        self.MRA = MRA(self.dim_learn, att_kernel)
        if stage == 2:
            self.D_GA = D_GA(self.dim_untouched)
        elif stage == 3:
            self.GA = GA(self.dim_untouched)
            self.norm = nn.BatchNorm2d(self.dim_untouched)
        else:
            self.L_Conv = L_Conv(self.dim_untouched, act_layer)
            self.norm = nn.BatchNorm2d(self.dim_untouched)
        self.drop_path = DropPath(drop_path)
        self.norm1 = nn.BatchNorm2d(self.dim)

    def forward(self, x: Tensor) -> Tensor:
        # for training/inference
        shortcut = x.clone()
        x1, x2, x3, x4 = torch.split(x, [self.dim_learn, self.dim_learn, self.dim_learn, self.dim_untouched], dim=1)
        x1 = x1 + self.P_Conv(x1)
        x2 = self.Conv(x2)
        x3 = self.MRA(x3)
        if self.stage == 2:
            x4 = x4 + self.D_GA(x4)
        elif self.stage == 3:
            x4 = self.norm(x4 + self.GA(x4))
        else:
            x4 = self.norm(x4 + self.L_Conv(x4))
        x_att = torch.cat((x1, x2, x3, x4), 1)
        x = shortcut + self.norm1(self.drop_path(self.mlp(x_att)))

        return x


class BasicStage(nn.Module):
    def __init__(self,
                 dim,
                 stage,
                 depth,
                 att_kernel,
                 mlp_ratio,
                 drop_path,
                 norm_layer,
                 act_layer
                 ):

        super().__init__()

        blocks_list = [
            MBFD(
                dim=dim,
                stage=stage,
                att_kernel=att_kernel,
                mlp_ratio=mlp_ratio,
                drop_path=drop_path[i],
                norm_layer=norm_layer,
                act_layer=act_layer
                 )
            for i in range(depth)
        ]

        self.blocks = nn.Sequential(*blocks_list)

    def forward(self, x: Tensor) -> Tensor:
        x = self.blocks(x)
        return x


class PatchEmbed(nn.Module):

    def __init__(self, patch_size, patch_stride, in_chans, embed_dim, norm_layer):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_stride, bias=False)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        x = self.norm(self.proj(x))
        return x

@ROTATED_BACKBONES.register_module()
class LWGANet(nn.Module):

    def __init__(self,
                 in_chans=3,
                 num_classes=1000,
                 embed_dim=64,
                 depths=(1, 2, 4, 2),
                 att_kernel=(11, 11, 11, 11),
                 norm_layer=nn.BatchNorm2d,
                 act_layer=nn.ReLU,
                 mlp_ratio=2.,
                 patch_size=4,
                 patch_stride=4,
                 patch_norm=True,
                 feature_dim=1280,
                 drop_path_rate=0.,
                 fork_feat=False,
                 init_cfg=None,
                 pretrained=None,
                 **kwargs):
        super().__init__()

        if embed_dim == 96:
            act_layer = nn.ReLU
        else:
            act_layer = nn.GELU

        if not fork_feat:
            self.num_classes = num_classes
        self.num_stages = len(depths)
        self.num_features = int(embed_dim * 2 ** (self.num_stages - 1))

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            patch_size=patch_size,
            patch_stride=patch_stride,
            in_chans=in_chans,
            embed_dim=embed_dim,
            norm_layer=norm_layer if patch_norm else None
        )

        # stochastic depth decay rule
        dpr = [x.item()
               for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # build layers
        stages_list = []
        for i_stage in range(self.num_stages):
            stage = BasicStage(dim=int(embed_dim * 2 ** i_stage),
                               stage=i_stage,
                               depth=depths[i_stage],
                               att_kernel=att_kernel[i_stage],
                               mlp_ratio=mlp_ratio,
                               drop_path=dpr[sum(depths[:i_stage]):sum(depths[:i_stage + 1])],
                               norm_layer=norm_layer,
                               act_layer=act_layer
                               )
            stages_list.append(stage)

            # patch merging layer
            if i_stage < self.num_stages - 1:
                stages_list.append(
                    DRFD(dim=int(embed_dim * 2 ** i_stage), act_layer=act_layer)
                )

        self.stages = nn.Sequential(*stages_list)

        self.fork_feat = fork_feat

        if self.fork_feat:
            self.forward = self.forward_det
            # add a norm layer for each output
            self.out_indices = [0, 2, 4, 6]
            for i_emb, i_layer in enumerate(self.out_indices):
                if i_emb == 0 and os.environ.get('FORK_LAST3', None):
                    raise NotImplementedError
                else:
                    layer = norm_layer(int(embed_dim * 2 ** i_emb))
                layer_name = f'norm{i_layer}'
                self.add_module(layer_name, layer)
        else:
            self.forward = self.forward_cls
            # Classifier head
            self.avgpool_pre_head = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(self.num_features, feature_dim, 1, bias=False),
                act_layer()
            )
            self.head = nn.Linear(feature_dim, num_classes) \
                if num_classes > 0 else nn.Identity()


        self.apply(self.cls_init_weights)
        self.init_cfg = copy.deepcopy(init_cfg)
        if self.fork_feat and (self.init_cfg is not None or pretrained is not None):
            self.init_weights()

    def cls_init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.Conv1d, nn.Conv2d)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.GroupNorm)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    # init for mmdetection by loading imagenet pre-trained weights
    def init_weights(self, pretrained=None):
        logger = get_root_logger()
        if self.init_cfg is None and pretrained is None:
            logger.warn(f'No pre-trained weights for '
                        f'{self.__class__.__name__}, '
                        f'training start from scratch')
            pass
        else:
            assert 'checkpoint' in self.init_cfg, f'Only support ' \
                                                  f'specify `Pretrained` in ' \
                                                  f'`init_cfg` in ' \
                                                  f'{self.__class__.__name__} '
            if self.init_cfg is not None:
                ckpt_path = self.init_cfg['checkpoint']
            elif pretrained is not None:
                ckpt_path = pretrained

            ckpt = _load_checkpoint(
                ckpt_path, logger=logger, map_location='cpu')
            if 'state_dict' in ckpt:
                _state_dict = ckpt['state_dict']
            elif 'model' in ckpt:
                _state_dict = ckpt['model']
            else:
                _state_dict = ckpt

            state_dict = _state_dict
            missing_keys, unexpected_keys = \
                self.load_state_dict(state_dict, False)

            # show for debug
            print('missing_keys: ', missing_keys)
            print('unexpected_keys: ', unexpected_keys)


    def forward_cls(self, x):
        # output only the features of last layer for image classification
        x = self.patch_embed(x)
        x = self.stages(x)
        x = self.avgpool_pre_head(x)  # B C 1 1
        x = torch.flatten(x, 1)
        x = self.head(x)

        return x

    def forward_det(self, x: Tensor) -> Tensor:
        # output the features of four stages for dense prediction
        x = self.patch_embed(x)
        outs = []
        for idx, stage in enumerate(self.stages):
            x = stage(x)
            if self.fork_feat and idx in self.out_indices:
                norm_layer = getattr(self, f'norm{idx}')
                x_out = norm_layer(x)
                outs.append(x_out)
        return outs


def LWGANet_L0_1242_e32_k11_GELU(num_classes: int = 1000, **kwargs):
    model = LWGANet(in_chans=3,
                    num_classes=num_classes,
                    embed_dim=32,
                    depths=(1, 2, 4, 2),
                    att_kernel=(11, 11, 11, 11),
                    norm_layer=nn.BatchNorm2d,
                    act_layer=nn.GELU,
                    drop_path_rate=0.,
                    **kwargs)

    return model


def LWGANet_L1_1242_e64_k11_GELU(num_classes: int = 1000, **kwargs):
    model = LWGANet(in_chans=3,
                    num_classes=num_classes,
                    embed_dim=64,
                    depths=(1, 2, 4, 2),
                    att_kernel=(11, 11, 11, 11),
                    norm_layer=nn.BatchNorm2d,
                    act_layer=nn.GELU,
                    drop_path_rate=0.0,
                    **kwargs)
    return model


def LWGANet_L2_1442_e96_k11_ReLU(num_classes: int = 1000, **kwargs):
    model = LWGANet(in_chans=3,
                    num_classes=num_classes,
                    embed_dim=96,
                    depths=(1, 4, 4, 2),
                    att_kernel=(11, 11, 11, 11),
                    norm_layer=nn.BatchNorm2d,
                    act_layer=nn.ReLU,
                    drop_path_rate=0.1,
                    **kwargs)
    return model

