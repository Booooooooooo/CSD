from model import common

import torch.nn as nn

# class SlimConv(nn.Module):
#     def __init__(self, in_channel, out_channel, kernel_size, padding=(kernel_size//2), bias=True):
#         self.in_channel = in_channel
#         self.out_channel = out_channel
#         self.padding = padding
#         self.bias = bias
#
#         self.conv = nn.Conv2d(in_channel, out_channel, kernel_size, padding=padding, bias=bias)
#
#     def forward(self, x, width_mult=1):
#         in_channel_width = int(width_mult * self.in_channel)
#         out_channel_width = int(width_mult * self.out_channel)
#         weight = self.conv.weight[:out_channel_width, :in_channel_width, :, :]
#         bias = None
#         if self.bias:
#             bias = self.conv.bias[:out_channel_width]
#
#         return nn.functional.conv2d(x, weight, bias, self.conv.stride)

class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.ModuleList([
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
        ])
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, width_mult=1):
        y = self.avg_pool(x)

        module = getattr(self.conv_du, '0')
        y = common.SlimModule(y, module, width_mult)
        y = self.relu(y)

        module = getattr(self.conv_du, '1')
        y = common.SlimModule(y, module, width_mult)
        y = self.sigmoid(y)

        return x * y

class RCAB(nn.Module):
    def __init__(self, n_feats, kernel_size, reduction, bias=True, act=nn.ReLU(True), res_scale=1):
        super(RCAB, self).__init__()

        modules_body = []
        for i in range(2):
            modules_body.append(nn.Conv2d(n_feats, n_feats, kernel_size, padding=(kernel_size//2), bias=bias))
        self.caLayer = CALayer(n_feats, reduction)
        self.body = nn.ModuleList(modules_body)
        self.act = act
        self.res_scale = res_scale

    def forward(self, x, width_mult=1):
        module = self.body[0]
        res = common.SlimModule(x, module, width_mult)
        res = self.act(res)
        module = self.body[1]
        res = common.SlimModule(res, module, width_mult)
        res = self.caLayer(res, width_mult)
        res += x
        return res

class ResidualGroup(nn.Module):
    def __init__(self, n_feats, kernel_size, reduction, act, res_scale, n_resblocks):
        super(ResidualGroup, self).__init__()

        modules_body = [
            RCAB(
                n_feats, kernel_size, reduction, bias=True, act=nn.ReLU(True), res_scale=1
            ) for _ in range(n_resblocks)
        ]
        self.body = nn.ModuleList(modules_body)
        self.conv = nn.Conv2d(n_feats, n_feats, kernel_size, padding=(kernel_size//2), bias=True)

    def forward(self, x, width_mult):
        res = x
        for module in self.body:
            res = module(res, width_mult)
        res = common.SlimModule(res, self.conv, width_mult)
        res += x
        return res


class RCAN(nn.Module):
    def __init__(self, args):
        super(RCAN, self).__init__()

        self.n_colors = args.n_colors
        n_resgroups = args.n_resgroups
        n_resblocks = args.n_resblocks
        n_feats = args.n_feats
        kernel_size = 3
        reduction = args.reduction
        scale = args.scale[0]
        act = nn.ReLU(True)

        self.sub_mean = common.MeanShift(args.rgb_range)

        self.head_conv = nn.Conv2d(args.n_colors, n_feats, kernel_size, padding=(kernel_size//2), bias=True)

        modules_body = [
            ResidualGroup(
                n_feats, kernel_size, reduction, act=act, res_scale=args.res_scale, n_resblocks=n_resblocks
            ) for _ in range(n_resgroups)
        ]
        # modules_body = [
        #     common.ResidualBlock(
        #         n_feats, kernel_size, act, args.res_scale
        #     ) for _ in range(n_resgroups)
        # ]
        self.body = nn.ModuleList(modules_body)
        self.body_conv = nn.Conv2d(n_feats, n_feats, kernel_size, padding=(kernel_size//2), bias=True)

        self.upsampler = common.Upsampler(scale, n_feats)
        self.tail_conv = nn.Conv2d(n_feats, args.n_colors, kernel_size, padding=(kernel_size // 2),
                                   bias=True)

        self.add_mean = common.MeanShift(args.rgb_range, sign=1)

    def forward(self, x, width_mult=1):
        x = self.sub_mean(x)
        weight = self.head_conv.weight
        n_feats = weight.shape[0]
        out_ch = int(n_feats * width_mult)
        weight = weight[:out_ch, :self.n_colors, :, :]
        bias = self.head_conv.bias[:out_ch]
        x = nn.functional.conv2d(x, weight, bias, stride=self.head_conv.stride, padding=self.head_conv.padding)

        res = x
        for module in self.body:
            res = module(res, width_mult)
        res = common.SlimModule(res, self.body_conv, width_mult)
        res += x

        x = self.upsampler(res, width_mult)
        weight = self.tail_conv.weight[:self.n_colors, :out_ch, :, :]
        bias = self.tail_conv.bias[:self.n_colors]
        x = nn.functional.conv2d(x, weight, bias, stride=self.tail_conv.stride, padding=self.tail_conv.padding)
        x = self.add_mean(x)

        return x

    def load_state_dict(self, state_dict, strict=False):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') >= 0:
                        print('Replace pre-trained upsampler to new one...')
                    else:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))

        if strict:
            missing = set(own_state.keys()) - set(state_dict.keys())
            if len(missing) > 0:
                raise KeyError('missing keys in state_dict: "{}"'.format(missing))
