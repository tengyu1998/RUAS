import torch
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
from .operations import *


def conv_layer(in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1):
    padding = int((kernel_size - 1) / 2) * dilation
    return nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding, bias=True, dilation=dilation,
                     groups=groups)


class SearchBlock(nn.Module):

    def __init__(self, channel, genotype):
        super(SearchBlock, self).__init__()

        self.stride = 1
        self.channel = channel

        op_names, indices = zip(*genotype)

        self.dc = self.distilled_channels = self.channel  # // 2
        self.rc = self.remaining_channels = self.channel
        self.c1_d = OPS[op_names[0]](self.channel, self.dc)
        self.c1_r = OPS[op_names[1]](self.channel, self.rc)
        self.c2_d = OPS[op_names[2]](self.channel, self.dc)
        self.c2_r = OPS[op_names[3]](self.channel, self.rc)
        self.c3_d = OPS[op_names[4]](self.channel, self.dc)
        self.c3_r = OPS[op_names[5]](self.channel, self.rc)
        self.c4 = OPS[op_names[6]](self.channel, self.dc)
        self.act = nn.LeakyReLU(negative_slope=0.05, inplace=False)
        self.c5 = conv_layer(self.dc * 4, self.channel, 1)

    def forward(self, input):
        distilled_c1 = self.act(self.c1_d(input))
        r_c1 = (self.c1_r(input))
        r_c1 = self.act(r_c1 + input)

        distilled_c2 = self.act(self.c2_d(r_c1))
        r_c2 = (self.c2_r(r_c1))
        r_c2 = self.act(r_c2 + r_c1)

        distilled_c3 = self.act(self.c3_d(r_c2))
        r_c3 = (self.c3_r(r_c2))
        r_c3 = self.act(r_c3 + r_c2)

        r_c4 = self.act(self.c4(r_c3))

        out = torch.cat([distilled_c1, distilled_c2, distilled_c3, r_c4], dim=1)
        out_fused = self.c5(out)

        return out_fused


class IEM(nn.Module):
    def __init__(self, channel, genetype):
        super(IEM, self).__init__()
        self.channel = channel
        self.genetype = genetype
        self.cell = SearchBlock(self.channel, self.genetype)
        self.activate = nn.Sigmoid()

    def max_operation(self, x):
        pad = nn.ConstantPad2d(1, 0)
        x = pad(x)[:, :, 1:, 1:]
        x = torch.max(x[:, :, :-1, :], x[:, :, 1:, :])
        x = torch.max(x[:, :, :, :-1], x[:, :, :, 1:])
        return x

    def forward(self, input_y, input_u, k):
        if k == 0:
            t_hat = self.max_operation(input_y)
        else:
            t_hat = input_u
        t = t_hat
        t = self.cell(t)
        t = self.activate(t)
        t = torch.clamp(t, 0.0001, 1.0)
        u = torch.clamp(input_y / t, 0.0, 1.0)

        return u, t


class EnhanceNetwork(nn.Module):
    def __init__(self, iteratioin, channel, genotype):
        super(EnhanceNetwork, self).__init__()
        self.iem_nums = iteratioin
        self.channel = channel
        self.genotype = genotype.enhance_op

        self.iems = nn.ModuleList()
        for i in range(self.iem_nums):
            self.iems.append(IEM(self.channel, self.genotype))

    def max_operation(self, x):

        pad = nn.ConstantPad2d(1, 0)
        x = pad(x)[:, :, 1:, 1:]
        x = torch.max(x[:, :, :-1, :], x[:, :, 1:, :])
        x = torch.max(x[:, :, :, :-1], x[:, :, :, 1:])
        return x

    def forward(self, input):
        t_list = []
        u_list = []
        u = torch.ones_like(input)
        for i in range(self.iem_nums):
            u, t = self.iems[i](input, u, i)
            u_list.append(u)
            t_list.append(t)

        return u_list, t_list