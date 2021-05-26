from model import common

import torch.nn as nn

class EDSR(nn.Module):
    def __init__(self, args):
        super(EDSR, self).__init__()

        self.n_colors = args.n_colors
        n_resblocks = args.n_resblocks
        self.n_feats = args.n_feats
        self.kernel_size = 3
        scale = args.scale[0]
        act = nn.ReLU(True)
        self.sub_mean = common.MeanShift(args.rgb_range)
        self.add_mean = common.MeanShift(args.rgb_range, sign=1)

        self.head = nn.Conv2d(args.n_colors, self.n_feats, self.kernel_size, padding=(self.kernel_size//2), bias=True)

        m_body = [
            common.ResidualBlock(
                self.n_feats, self.kernel_size, act=act, res_scale=args.res_scale
            ) for _ in range(n_resblocks)
        ]
        self.body = nn.ModuleList(m_body)
        self.body_conv = nn.Conv2d(self.n_feats, self.n_feats, self.kernel_size, padding=(self.kernel_size//2), bias=True)

        self.upsampler = common.Upsampler(scale, self.n_feats)
        self.tail_conv = nn.Conv2d(self.n_feats, args.n_colors, self.kernel_size, padding=(self.kernel_size//2), bias=True)

    def forward(self, x, width_mult=1):
        feature_width = int(self.n_feats * width_mult)

        x = self.sub_mean(x)
        weight = self.head.weight[:feature_width, :self.n_colors, :, :]
        bias = self.head.bias[:feature_width]
        x = nn.functional.conv2d(x, weight, bias, padding=(self.kernel_size//2))

        residual = x
        for block in self.body:
            residual = block(residual, width_mult)
        weight = self.body_conv.weight[:feature_width, :feature_width, :, :]
        bias = self.body_conv.bias[:feature_width]
        residual = nn.functional.conv2d(residual, weight, bias, padding=(self.kernel_size//2))
        residual += x

        x = self.upsampler(residual, width_mult)
        weight = self.tail_conv.weight[:self.n_colors, :feature_width, :, :]
        bias = self.tail_conv.bias[:self.n_colors]
        x = nn.functional.conv2d(x, weight, bias, padding=(self.kernel_size//2))
        x = self.add_mean(x)

        return x

    def load_state_dict(self, state_dict, strict=True):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') == -1:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))
