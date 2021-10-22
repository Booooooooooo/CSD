import math

import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor
import mindspore.numpy as numpy
from mindspore.common.initializer import TruncatedNormal


def weight_variable():
    """weight initial"""
    return TruncatedNormal(0.02)

def conv(in_channels, out_channels, kernel_size, stride=1, padding=0,
         pad_mode='pad', has_bias=True):
    """weight initial for conv layer"""
    weight = weight_variable()
    return nn.Conv2d(in_channels, out_channels,
                     kernel_size=kernel_size, stride=stride, padding=padding,
                     weight_init=weight, has_bias=has_bias, pad_mode=pad_mode)

def pixel_shuffle(tensor, scale_factor):
    """
    Implementation of pixel shuffle using numpy

    Parameters:
    -----------
    tensor: input tensor, shape is [N, C, H, W]
    scale_factor: scale factor to up-sample tensor

    Returns:
    --------
    tensor: tensor after pixel shuffle, shape is [N, C/(r*r), r*H, r*W],
        where r refers to scale factor
    """
    num, ch, height, width = tensor.shape
    # assert ch % (scale_factor * scale_factor) == 0

    new_ch = ch // (scale_factor * scale_factor)
    new_height = height * scale_factor
    new_width = width * scale_factor

    reshape = ops.Reshape()
    tensor = reshape(tensor, (num, new_ch, scale_factor, scale_factor, height, width))
    # new axis: [num, new_ch, height, scale_factor, width, scale_factor]
    transpose = ops.Transpose()
    tensor = transpose(tensor, (0, 1, 4, 2, 5, 3))
    tensor = reshape(tensor, (num, new_ch, new_height, new_width))
    return tensor

def MeanShift(x, rgb_range,
        rgb_mean=(0.4488, 0.4371, 0.4040), rgb_std=(1.0, 1.0, 1.0), sign=-1):

        # super(MeanShift, self).__init__(3, 3, kernel_size=1)

        std = Tensor(rgb_std)
        conv2d = ops.Conv2D(out_channel=3, kernel_size=1)
        biasadd = ops.BiasAdd()
        weight = numpy.eye(3, 3).view((3, 3, 1, 1)) / std.view(3, 1, 1, 1)
        bias = sign * rgb_range * Tensor(rgb_mean) / std
        weight = weight.astype(numpy.float32)
        bias = bias.astype(numpy.float32)

        x = conv2d(x, weight)
        x = biasadd(x, bias)
        return x

class ResidualBlock(nn.Cell):
    def __init__(self, n_feats, kernel_size, act, res_scale):
        super(ResidualBlock, self).__init__()
        self.n_feats = n_feats
        self.res_scale = res_scale
        self.kernel_size = kernel_size

        # self.conv1 = nn.Conv2d(in_channels=n_feats, out_channels=n_feats, kernel_size=kernel_size, pad_mode='pad', padding=1, has_bias=True)
        self.conv1 = conv(n_feats, n_feats, kernel_size, padding=1)
        self.act = act
        # self.conv2 = nn.Conv2d(in_channels=n_feats, out_channels=n_feats, kernel_size=kernel_size, pad_mode='pad', padding=1, has_bias=True)
        self.conv2 = conv(n_feats, n_feats, kernel_size, padding=1)

    def construct(self, x, width_mult=1):
        # round = ops.Round()
        width = int(self.n_feats * width_mult)
        # width = round(self.n_feats * width_mult)
        conv2d = ops.Conv2D(out_channel=width, kernel_size=self.kernel_size, mode=1, pad_mode='pad', pad=1)
        biasadd = ops.BiasAdd()
        weight = self.conv1.weight[:width, :width, :, :]
        bias = self.conv1.bias[:width]
        residual = conv2d(x, weight)
        if bias is not None:
            residual = biasadd(residual, bias)
        residual = self.act(residual)
        weight = self.conv2.weight[:width, :width, :, :]
        bias = self.conv2.bias[:width]
        residual = conv2d(residual, weight)
        if bias is not None:
            residual = biasadd(residual, bias)

        return x + residual * self.res_scale

class Upsampler(nn.SequentialCell):
    def __init__(self, scale_factor, nf):
        super(Upsampler, self).__init__()
        block = []
        self.nf = nf
        self.scale = scale_factor

        if scale_factor == 3:
            block += [
                # nn.Conv2d(in_channels=nf, out_channels=nf * 9, kernel_size=3, pad_mode='pad', padding=1, has_bias=True)
                conv(nf, nf*9, 3, padding=1)
            ]
            # self.pixel_shuffle = nn.PixelShuffle(3)
            # pixel_shuffle function
        else:
            self.block_num = scale_factor // 2
            # self.pixel_shuffle = nn.PixelShuffle(2)
            #self.act = nn.ReLU()

            for _ in range(self.block_num):
                block += [
                    # nn.Conv2d(in_channels=nf, out_channels=nf * 2 ** 2, kernel_size=3, pad_mode='pad', padding=1, has_bias=True)
                    conv(nf, nf*2**2, 3, padding=1)
                ]
        self.blocks = nn.SequentialCell(block)

    def construct(self, x, width_mult=1):
        res = x
        nf = self.nf
        if self.scale == 3:
            width = int(width_mult * nf)
            width9 = width * 9
            conv2d = ops.Conv2D(out_channel=width9, kernel_size=3, mode=1, pad_mode='pad', pad=1)
            biasadd = ops.BiasAdd()
            for block in self.blocks:
                weight = block.weight[:width9, :width, :, :]
                bias = block.bias[:width9]
                res = conv2d(res, weight)
                if bias:
                    res = biasadd(res, bias)
                res = pixel_shuffle(res, self.scale)
        else:
            width = int(width_mult * nf)
            width4 = width * 4
            conv2d = ops.Conv2D(out_channel=width4, kernel_size=3, mode=1, pad_mode='pad', pad=1)
            biasadd = ops.BiasAdd()
            for block in self.blocks:
                weight = block.weight[:width4, :width, :, :]
                bias = block.bias[:width4]
                # print(res.shape)
                res = conv2d(res, weight)
                # print(res.shape)
                if bias is not None:
                    res = biasadd(res, bias)
                res = pixel_shuffle(res, 2)
            #res = self.act(res)

        return res

def SlimModule(input, module, width_mult):
    weight = module.weight
    out_ch, in_ch = weight.shape[:2]
    out_ch = int(out_ch * width_mult)
    in_ch = int(in_ch * width_mult)
    weight = weight[:out_ch, :in_ch, :, :]
    bias = module.bias

    conv2d = ops.Conv2D(out_channel=out_ch, kernel_size=module.kernel_size, mode=1, pad_mode=module.pad_mode, pad=module.padding)
    biasadd =ops.BiasAdd()
    out = conv2d(input, weight)
    if bias is not None:
        bias = module.bias[:out_ch]
        out = biasadd(out, bias)
    return out
