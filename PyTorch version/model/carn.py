from model import common

import torch
import torch.nn as nn

class BasicBlock(nn.Module):
    def __init__(self,
                 in_channels, out_channels,
                 ksize=3, stride=1, pad=1):
        super(BasicBlock, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, ksize, stride, pad)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x, width_mult):
        out = common.SlimModule(x, self.conv, width_mult)
        out = self.act(out)
        return out

class Block(nn.Module):
    def __init__(self, n_feats):
        super(Block, self).__init__()

        self.act = nn.ReLU(inplace=True)

        self.b1 = common.ResidualBlock(n_feats, 3, act=self.act, res_scale=1)
        self.b2 = common.ResidualBlock(n_feats, 3, self.act, 1)
        self.b3 = common.ResidualBlock(n_feats, 3, self.act, 1)
        self.c1 = BasicBlock(n_feats*2, n_feats, 1, 1, 0)
        self.c2 = BasicBlock(n_feats*3, n_feats, 1, 1, 0)
        self.c3 = BasicBlock(n_feats*4, n_feats, 1, 1, 0)

    def forward(self, x, width_mult=1):
        c0 = o0 = x

        b1 = self.b1(o0, width_mult)
        c1 = torch.cat([c0, b1], dim=1)
        o1 = self.c1(c1, width_mult)

        b2 = self.b2(o1, width_mult)
        c2 = torch.cat([c1, b2], dim=1)
        o2 = self.c2(c2, width_mult)

        b3 = self.b3(o2, width_mult)
        c3 = torch.cat([c2, b3], dim=1)
        o3 = self.c3(c3, width_mult)

        return o3

class CARN(nn.Module):
    def __init__(self, args):
        super(CARN, self).__init__()

        scale = args.scale[0]
        self.act = nn.ReLU(inplace=True)
        self.n_feat = args.n_feats

        self.sub_mean = common.MeanShift(args.rgb_range)
        self.add_mean = common.MeanShift(args.rgb_range, sign=1)

        self.entry = nn.Conv2d(3, self.n_feat, 3, 1, 1)
        self.b1 = Block(self.n_feat)
        self.b2 = Block(self.n_feat)
        self.b3 = Block(self.n_feat)
        self.c1 = BasicBlock(self.n_feat*2, self.n_feat, 1, 1, 0)
        self.c2 = BasicBlock(self.n_feat*3, self.n_feat, 1, 1, 0)
        self.c3 = BasicBlock(self.n_feat*4, self.n_feat, 1, 1, 0)

        self.upsample = common.Upsampler(scale, self.n_feat)
        self.exit = nn.Conv2d(self.n_feat, 3, 3, 1, 1)

    def forward(self, x, width_mult=1):
        x = self.sub_mean(x)
        nf = int(self.n_feat * width_mult)
        weight = self.entry.weight[:nf, :3, :, :]
        bias = self.entry.bias[:nf]
        x = nn.functional.conv2d(x, weight, bias, stride=self.entry.stride, padding=self.entry.padding)
        c0 = o0 = x

        b1 = self.b1(o0, width_mult)
        c1 = torch.cat([c0, b1], dim=1)
        o1 = self.c1(c1, width_mult)

        b2 = self.b2(o1, width_mult)
        c2 = torch.cat([c1, b2], dim=1)
        o2 = self.c2(c2, width_mult)

        b3 = self.b3(o2, width_mult)
        c3 = torch.cat([c2, b3], dim=1)
        o3 =self.c3(c3, width_mult)

        out = self.upsample(o3, width_mult)
        weight = self.exit.weight[:3, :nf, :, :]
        bias = self.exit.bias[:3]
        out = nn.functional.conv2d(out, weight, bias, stride=self.exit.stride, padding=self.exit.padding)
        out = self.add_mean(out)

        return out

