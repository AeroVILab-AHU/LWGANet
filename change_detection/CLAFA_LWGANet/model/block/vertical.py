import torch
import torch.nn as nn
import torch.nn.functional as F
from .convs import ConvBnRelu
from mmcv.ops import MultiScaleDeformableAttention
from einops import rearrange
from torch import einsum


class ScaledSinuEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1,))
        inv_freq = 1. / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, x):
        n, device = x.shape[1], x.device
        t = torch.arange(n, device=device).type_as(self.inv_freq)
        sinu = einsum('i, j -> i j', t, self.inv_freq)
        emb = torch.cat((sinu.sin(), sinu.cos()), dim=-1)
        return emb * self.scale


def get_reference_points(spatial_shapes, device):
    reference_points_list = []
    for lvl, (H_, W_) in enumerate(spatial_shapes):
        ref_y, ref_x = torch.meshgrid(
            torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
            torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))
        ref_y = ref_y.reshape(-1)[None] / H_
        ref_x = ref_x.reshape(-1)[None] / W_
        ref = torch.stack((ref_x, ref_y), -1)
        reference_points_list.append(ref)
    reference_points = torch.cat(reference_points_list, 1)
    reference_points = reference_points[:, :, None]
    return reference_points


### Attend-then-Filter
class VerticalFusion(nn.Module):
    def __init__(self, channels, num_heads=4, num_points=4, kernel_layers=1, up_kernel_size=5, enc_kernel_size=3):
        super(VerticalFusion, self).__init__()
        self.norm1 = nn.LayerNorm(channels)
        self.norm2 = nn.LayerNorm(channels)
        self.pos = ScaledSinuEmbedding(channels)
        self.crossattn = MultiScaleDeformableAttention(embed_dims=channels, num_levels=1, num_heads=num_heads,
                                                       num_points=num_points, batch_first=True, dropout=0)
        convs = []
        convs.append(ConvBnRelu(in_channels=channels, out_channels=channels))
        for _ in range(kernel_layers - 1):
            convs.append(ConvBnRelu(in_channels=channels, out_channels=channels))
        self.convs = nn.Sequential(*convs)
        self.enc = ConvBnRelu(channels, up_kernel_size ** 2, kernel_size=enc_kernel_size,
                              stride=1, padding=enc_kernel_size // 2, dilation=1)

        self.upsmp = nn.Upsample(scale_factor=2, mode='nearest')
        self.unfold = nn.Unfold(kernel_size=up_kernel_size, dilation=2,
                                padding=up_kernel_size // 2 * 2)

    def get_deform_inputs(self, x1, x2):
        _, _, H1, W1 = x1.size()
        _, _, H2, W2 = x2.size()
        spatial_shapes = torch.as_tensor([(H2, W2)], dtype=torch.long, device=x2.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
        reference_points = get_reference_points([(H1, W1)], x1.device)

        return reference_points, spatial_shapes, level_start_index

    def forward(self, x1, x2):
        #Attend
        reference_points, spatial_shapes, level_start_index = self.get_deform_inputs(x1, x2)
        B, C, H, W = x1.size()
        _, _, H2, W2 = x2.size()
        x1_, x2_ = x1.clone(), x2.clone()
        x1 = rearrange(x1, 'b c h w -> b (h w) c')
        x2 = rearrange(x2, 'b c h w -> b (h w) c')
        x1, x2 = self.norm1(x1), self.norm2(x2)
        query_pos = self.pos(x1)
        x = self.crossattn(query=x1, value=x2, reference_points=reference_points, spatial_shapes=spatial_shapes,
                           level_start_index=level_start_index, query_pos=query_pos)
        x = rearrange(x, 'b (h w) c -> b c h w', h=H, w=W)

        #Filter
        kernel = self.convs(x2_)
        kernel = self.enc(kernel)
        kernel = F.softmax(kernel, dim=1)
        # x = self.upsmp(x)
        x = F.interpolate(x, size=(H2, W2), mode='nearest')
        x = self.unfold(x)
        # x = x.view(B, C, -1, H * 2, W * 2)
        x = x.view(B, C, -1, H2, W2)
        fuse = torch.einsum('bkhw,bckhw->bchw', [kernel, x])
        fuse += x2_

        return fuse

