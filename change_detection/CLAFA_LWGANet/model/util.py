import torch.nn as nn


def init_method(*nets, init_type='normal'):
    for net in nets:
        for module in net.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear) or isinstance(module, nn.ConvTranspose2d):
                if init_type == 'normal':
                    nn.init.normal_(module.weight.data, 0.0, 0.02)
                elif init_type == 'xavier':
                    nn.init.xavier_normal_(module.weight.data, gain=1.0)
                elif init_type == 'kaiming_normal':
                    nn.init.kaiming_normal_(module.weight.data, a=0, mode='fan_in', nonlinearity='relu')
                elif init_type == 'kaiming_normal_out':
                    nn.init.kaiming_normal_(module.weight.data, a=0, mode='fan_out', nonlinearity='relu')
                elif init_type == 'kaiming_uniform':
                    nn.init.kaiming_uniform_(module.weight.data, a=0, mode='fan_in', nonlinearity='relu')
                elif init_type == 'trunc_normal':
                    nn.init.trunc_normal_(module.weight.data, mean=0.0, std=1.0, a=- 2.0, b=2.0)
                elif init_type == 'orthogonal':
                    nn.init.orthogonal_(module.weight.data, gain=1.0)
                else:
                    raise NotImplementedError("initialization method [%s] is not implemented" % init_type)
                if module.bias is not None:
                    nn.init.constant_(module.bias.data, 0.0)
            elif isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm1d) or isinstance(module, nn.LayerNorm):
                nn.init.normal_(module.weight.data, 1.0, 0.02)
                nn.init.constant_(module.bias.data, 0.0)

    print("initialize \\backbone networks with [%s]" % init_type)

