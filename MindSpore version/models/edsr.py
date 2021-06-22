import mindspore
from model import common

import mindspore.nn as nn
import mindspore.ops.operations as P

class EDSR(nn.Cell):
    def __init__(self, args):
        super(EDSR, self).__init__()

        self.n_colors = args.n_colors
        n_resblocks = args.n_resblocks
        self.n_feats = args.n_feats
        self.kernel_size = 3
        scale = args.scale[0]
        act = nn.ReLU()
        self.sub_mean = common.MeanShift(args.rgb_range)
        self.add_mean = common.MeanShift(args.rgb_range, sign=1)

        self.head = nn.Conv2d(in_channels=args.n_colors, out_channels=self.n_feats, kernel_size=self.kernel_size, pad_mode='pad', padding=self.kernel_size // 2, has_bias=True)

        m_body = [
            common.ResidualBlock(
                self.n_feats, self.kernel_size, act=act, res_scale=args.res_scale
            ) for _ in range(n_resblocks)
        ]
        self.body = nn.SequentialCell(m_body)
        self.body_conv = nn.Conv2d(in_channels=self.n_feats, out_channels=self.n_feats, kernel_size=self.kernel_size, pad_mode='pad', padding=self.kernel_size // 2, has_bias=True)

        self.upsampler = common.Upsampler(scale, self.n_feats)
        self.tail_conv = nn.Conv2d(in_channels=self.n_feats, out_channels=args.n_colors, kernel_size=self.kernel_size, pad_mode='pad', padding=self.kernel_size // 2, has_bias=True)

    def construct(self, x, width_mult=1):
        feature_width = int(self.n_feats * width_mult)
        conv2d = ops.Conv2D(out_channels=feature_width, kernel_size=self.kernel_size, mode=1, pad_mode='pad',
                            pad=self.kernel_size // 2)
        biasadd = ops.BiasAdd()

        x = self.sub_mean(x)
        weight = self.head.weight[:feature_width, :self.n_colors, :, :]
        bias = self.head.bias[:feature_width]
        x = conv2d(x, weight)
        x = biasadd(x, bias)

        residual = x
        for block in self.body:
            residual = block(residual, width_mult)
        weight = self.body_conv.weight[:feature_width, :feature_width, :, :]
        bias = self.body_conv.bias[:feature_width]
        residual = conv2d(residual, weight)
        residual = biasadd(residual, bias)
        residual += x

        x = self.upsampler(residual, width_mult)
        weight = self.tail_conv.weight[:self.n_colors, :feature_width, :, :]
        bias = self.tail_conv.bias[:self.n_colors]
        conv2d = ops.Conv2D(out_channels=self.n_colors, kernel_size=self.kernel_size, mode=1, pad_mode='pad', pad=self.kernel_size//2)
        x = conv2d(x, weight)
        x = biasadd(x, bias)
        x = self.add_mean(x)

        return x