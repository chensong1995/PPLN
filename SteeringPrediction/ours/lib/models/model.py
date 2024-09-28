import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.models.modules import SpikingConv2d, SpikingLinear

import pdb

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x1 = self.relu(self.bn1(self.conv1(x)))
        x2 = self.bn2(self.conv2(x1))
        if x2.shape == x.shape:
            shortcut = x
        else:
            out_channels = x2.shape[1]
            shortcut = F.pad(x[:, :, ::2, ::2],
                             (0, 0, # no padding on w
                              0, 0, # no padding on h
                              out_channels // 4, out_channels // 4), # padding on c
                             'constant',
                             0)
        x3 = self.relu(x2 + shortcut)
        return x3

class ResLayer(nn.Module):
    def __init__(self, in_channels, out_channels, num_blocks, stride):
        super(ResLayer, self).__init__()
        self.layers = []
        self.layers.append(ResBlock(in_channels, out_channels, stride))
        for _ in range(num_blocks - 1):
            self.layers.append(ResBlock(out_channels, out_channels, 1))
        self.layers = nn.ModuleList(self.layers)

    def forward(self, x, t):
        for layer in self.layers:
            x = layer(x)
        return x

class SpikingResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, norm='none', temp=None):
        super(SpikingResBlock, self).__init__()
        self.conv1 = SpikingConv2d(in_channels, out_channels, 3, stride=stride,
                                   padding=1, bias=False, spiking_segment_count=3,
                                   norm=norm, temp=temp)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = SpikingConv2d(out_channels, out_channels, 3, stride=1,
                                   padding=1, bias=False, spiking_segment_count=3,
                                   norm=norm, temp=temp)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, t):
        x1 = self.relu(self.bn1(self.conv1(x, t)))
        x2 = self.bn2(self.conv2(x1, t))
        if x2.shape == x.shape:
            shortcut = x
        else:
            out_channels = x2.shape[1]
            shortcut = F.pad(x[:, :, ::2, ::2],
                             (0, 0, # no padding on w
                              0, 0, # no padding on h
                              out_channels // 4, out_channels // 4), # padding on c
                             'constant',
                             0)
        x3 = self.relu(x2 + shortcut)
        return x3

class SpikingResLayer(nn.Module):
    def __init__(self, in_channels, out_channels, num_blocks, stride, norm='none',
                 temp=None):
        super(SpikingResLayer, self).__init__()
        self.layers = []
        self.layers.append(SpikingResBlock(in_channels, out_channels, stride,
                                           norm=norm, temp=temp))
        for _ in range(num_blocks - 1):
            self.layers.append(SpikingResBlock(out_channels, out_channels, 1,
                                               norm=norm, temp=temp))
        self.layers = nn.ModuleList(self.layers)

    def forward(self, x, t):
        for layer in self.layers:
            x = layer(x, t)
        return x

class SpikingLinearResBlock(nn.Module):
    def __init__(self, channels=512, norm='none', temp=None):
        super(SpikingLinearResBlock, self).__init__()
        self.linear1 = SpikingLinear(channels, channels,
                                     spiking_segment_count=3, norm=norm, temp=temp)
        self.linear2 = SpikingLinear(channels, channels,
                                     spiking_segment_count=3, norm=norm, temp=temp)
        self.relu = nn.ReLU()

    def forward(self, x, t):
        x1 = self.relu(self.linear1(x, t))
        x2 = self.linear2(x1, t)
        return x + x2

class SpikingLinearResNet(nn.Module):
    def __init__(self, channels=512, blocks=5, norm='none', temp=None):
        super(SpikingLinearResNet, self).__init__()
        self.relu = nn.ReLU()
        self.blocks = [SpikingLinearResBlock(channels=channels, norm=norm, temp=temp) for _ in range(blocks)]
        self.blocks = nn.ModuleList(self.blocks)

    def forward(self, x, t):
        for block in self.blocks:
            x = self.relu(block(x, t))
        return x

class Backbone(nn.Module):
    def __init__(self, group_size=10, norm='none', temp=None):
        super(Backbone, self).__init__()
        self.conv1 = nn.Conv2d(2 * group_size, 16, 3, stride=1, padding=1,
                               bias=False)
        self.layer1 = ResLayer(16, 16, 5, 1)
        self.layer2 = ResLayer(16, 32, 5, 2)
        self.layer3 = ResLayer(32, 64, 4, 2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear1 = SpikingLinear(64, 32, spiking_segment_count=3, norm=norm,
                                     temp=temp)
        self.linear2 = SpikingLinearResNet(channels=32, blocks=2, norm=norm,
                                           temp=temp)
        self.linear3 = SpikingLinear(32, 1, spiking_segment_count=3, norm=norm,
                                     temp=temp)
        self.bn16 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        t = torch.linspace(-1, 1, group_size)
        self.register_buffer('t', t)

    def forward(self, x, t):
        x1 = self.relu(self.bn16(self.conv1(x)))
        x2 = self.layer1(x1, t)
        x3 = self.layer2(x2, t)
        x4 = self.layer3(x3, t)
        x5 = torch.flatten(self.avgpool(x4), 1)
        x6 = self.relu(self.linear1(x5, t))
        x7 = self.linear2(x6, t)
        x8 = self.linear3(x7, t).squeeze(dim=-1)
        return x8

class PredNet(nn.Module):
    def __init__(self, use_radians=False, group_size=10, norm='constant', temp=None):
        super(PredNet, self).__init__()
        self.backbone = Backbone(group_size=group_size, norm=norm, temp=temp)
        self.use_radians = use_radians
        self.criterion = nn.MSELoss()

    def forward(self, batch):
        aps = batch['aps']
        dvs = batch['dvs']
        net_in = torch.cat([aps, dvs], axis=1)
        angle_pred = self.backbone(net_in, batch['timestamp'])
        angle_gt = batch['angle']
        if self.use_radians:
            # the raw prediction is in radians
            angle_gt_rad = angle_gt / 180. * math.pi
            loss = self.criterion(angle_pred, angle_gt_rad)
            # conver to degrees
            angle_pred = angle_pred / math.pi * 180.
        else:
            loss = self.criterion(angle_pred, angle_gt)
        return angle_pred, loss
