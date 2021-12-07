# -*- coding:utf-8 -*-

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable

from model_detection.layers import *
from data.config import cfg
from .genotypes import *
from .operations import *


class DetectionNetwork(nn.Module):
    """Single Shot Multibox Architecture
    The network is composed of a base VGG network followed by the
    added multibox conv layers.  Each multibox layer branches into
        1) conv2d for class conf scores
        2) conv2d for localization predictions
        3) associated priorbox layer to produce default bounding
           boxes specific to the layer's feature map size.
    See: https://arxiv.org/pdf/1512.02325.pdf for more details.

    Args:
        phase: (string) Can be "test" or "train"
        size: input image size
        base: VGG16 layers for input, size of either 300 or 500
        extras: extra layers that feed to multibox loc and conf layers
        head: "multibox head" consists of loc and conf conv layers
    """

    def __init__(self, phase, num_classes, genotype):
        super(DetectionNetwork, self).__init__()

        self.base = vgg(vgg_cfg, 3)
        self.genotype = genotype
        self.add_extras = add_extras(extras_cfg, 1024, self.genotype)
        self.head1 = multibox(self.base, self.add_extras, num_classes, self.genotype)

        self.phase = phase
        self.num_classes = num_classes
        self.vgg = nn.ModuleList(self.base)

        self.L2Normof1 = L2Norm(256, 10)
        self.L2Normof2 = L2Norm(512, 8)
        self.L2Normof3 = L2Norm(512, 5)

        self.extras = nn.ModuleList(self.add_extras)

        self.loc_pal1 = nn.ModuleList(self.head1[0])
        self.conf_pal1 = nn.ModuleList(self.head1[1])

        self.criterion = MultiBoxLoss(cfg, True)

        if self.phase == 'test':
            self.softmax = nn.Softmax(dim=-1)
            self.detect = Detect(cfg)

    def _upsample_prod(self, x, y):
        _, _, H, W = y.size()
        return F.upsample(x, size=(H, W), mode='bilinear') * y

    def forward(self, x, targets):
        size = x.size()[2:]
        pal1_sources = list()
        loc_pal1 = list()
        conf_pal1 = list()

        # apply vgg up to conv4_3 relu
        for k in range(16):
            x = self.vgg[k](x)
        of1 = x
        s = self.L2Normof1(of1)
        pal1_sources.append(s)
        # apply vgg up to fc7
        for k in range(16, 23):
            x = self.vgg[k](x)
        of2 = x
        s = self.L2Normof2(of2)
        pal1_sources.append(s)

        for k in range(23, 30):
            x = self.vgg[k](x)
        of3 = x
        s = self.L2Normof3(of3)
        pal1_sources.append(s)

        for k in range(30, len(self.vgg)):
            x = self.vgg[k](x)
        of4 = x
        pal1_sources.append(of4)

        for k in range(2):
            x = F.relu(self.extras[k](x), inplace=True)

        of5 = x
        pal1_sources.append(of5)
        for k in range(2, 4):
            x = F.relu(self.extras[k](x), inplace=True)

        of6 = x
        pal1_sources.append(of6)

        for (x, l, c) in zip(pal1_sources, self.loc_pal1, self.conf_pal1):
            loc_pal1.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf_pal1.append(c(x).permute(0, 2, 3, 1).contiguous())

        features_maps = []
        for i in range(len(loc_pal1)):
            feat = []
            feat += [loc_pal1[i].size(1), loc_pal1[i].size(2)]
            features_maps += [feat]

        loc_pal1 = torch.cat([o.view(o.size(0), -1)
                              for o in loc_pal1], 1)
        conf_pal1 = torch.cat([o.view(o.size(0), -1)
                               for o in conf_pal1], 1)

        priorbox = PriorBox(size, features_maps, cfg)
        with torch.no_grad():
            self.priors_pal1 = Variable(priorbox.forward())

        if self.phase == 'test':

            output = self.detect.forward(
                loc_pal1.view(loc_pal1.size(0), -1, 4),
                self.softmax(conf_pal1.view(conf_pal1.size(0), -1,
                                            self.num_classes)),  # conf preds
                self.priors_pal1.type(type(x.data))
            )

        else:
            output = (
                loc_pal1.view(loc_pal1.size(0), -1, 4),
                conf_pal1.view(conf_pal1.size(0), -1, self.num_classes),
                self.priors_pal1)

        return output



vgg_cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
           512, 512, 512, 'M']

extras_cfg = [256, 'S', 512, 128, 'S', 256]

fem_cfg = [256, 512, 512, 1024, 512, 256]


def vgg(cfg, i, batch_norm=False):
    layers = []
    in_channels = i
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'C':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=3, dilation=3)
    conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
    layers += [conv6,
               nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]
    return layers


flag = 0
def add_extras(cfg, i, genotype):
    # Extra layers added to VGG for feature scaling
    global flag
    layers = []
    in_channels = i

    op_names = genotype.detection_op

    for k, v in enumerate(cfg):
        if in_channels != 'S':
            if v == 'S':
                layers += [OPS2[op_names[flag]](in_channels, cfg[k + 1])]
                flag += 1
            else:
                layers += [OPS[op_names[flag]](in_channels, v)]
                flag += 1

        in_channels = v
    return layers


def multibox(vgg, extra_layers, num_classes, genotype):
    global flag
    op_names = genotype.detection_op
    loc_layers = []
    conf_layers = []
    vgg_source = [14, 21, 28, -2]

    for k, v in enumerate(vgg_source):
        loc_layers += [OPS[op_names[flag]](vgg[v].out_channels, 4)]
        conf_layers += [OPS[op_names[flag + 6]](vgg[v].out_channels, num_classes)]
        flag += 1

    for k, v in enumerate(extra_layers[1::2], 2):
        loc_layers += [OPS[op_names[flag]](v.out_channels, 4)]
        conf_layers += [OPS[op_names[flag + 6]](v.out_channels, num_classes)]
        flag += 1

    return (loc_layers, conf_layers)
