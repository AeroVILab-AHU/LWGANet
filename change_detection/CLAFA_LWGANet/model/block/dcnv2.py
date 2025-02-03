from mmcv.ops import ModulatedDeformConv2dPack, modulated_deform_conv2d
import torch
import torch.nn as nn


class DCNv2(ModulatedDeformConv2dPack):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        out_channels = self.deform_groups * 3 * self.kernel_size[0] * self.kernel_size[1]
        pyconv_kernels = [1, 3, 5]
        pyconv_groups = [1, self.deform_groups // 2, self.deform_groups]
        pyconv_levels = []
        for pyconv_kernel, pyconv_group in zip(pyconv_kernels, pyconv_groups):
            pyconv_levels.append(nn.Sequential(nn.Conv2d(self.in_channels, out_channels, kernel_size=pyconv_kernel,
                                                         padding=pyconv_kernel // 2, groups=pyconv_group, bias=False),
                                               nn.BatchNorm2d(out_channels),
                                               nn.ReLU(True)))
        self.pyconv_levels = nn.Sequential(*pyconv_levels)
        self.offset = nn.Conv2d(out_channels * 3, out_channels, 1, bias=True)
        self.init_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
        out = []
        for level in self.pyconv_levels:
            out.append(level(x))
        out = torch.cat(out, dim=1)

        out = self.offset(out)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)
        return modulated_deform_conv2d(x, offset, mask, self.weight, self.bias,
                                       self.stride, self.padding,
                                       self.dilation, self.groups,
                                       self.deform_groups)

