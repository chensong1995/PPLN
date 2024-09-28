import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import pdb

class SpikingConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, bias=True, padding_mode='zeros',
                 spiking_segment_count=3, norm='none', temp=50):
        super(SpikingConv2d, self).__init__(\
                in_channels, out_channels * spiking_segment_count * 3, kernel_size,
                stride=stride, padding=padding, dilation=dilation, groups=groups,
                bias=bias, padding_mode=padding_mode)
        if norm != 'none':
            self.norm_net = nn.Conv2d(in_channels, out_channels, kernel_size,
                                      stride=stride, padding=padding,
                                      dilation=dilation, groups=groups, bias=bias,
                                      padding_mode=padding_mode)
        self.out_channels = out_channels
        self.spiking_segment_count = spiking_segment_count
        self.norm = norm
        self.softmax = nn.Softmax(dim=2)
        self.temp = temp

    def forward(self, x, t):
        # predict segment parameters
        params = super().forward(x)
        bs, _, h, w = params.shape
        params = params.reshape((bs,
                                 self.out_channels,
                                 self.spiking_segment_count,
                                 3,
                                 h,
                                 w))
        weights = self.softmax(params[:, :, :, 0])
        keypoints = torch.cat([torch.zeros_like(weights[:, :, :1]),
                               torch.cumsum(weights, dim=2)], dim=2)
        keypoints = keypoints * 2 - 1
        slopes = torch.tanh(params[:, :, :, 1])
        intercepts = params[:, :, :, 2]

        # assemble kernels
        t = t.reshape((bs, 1, 1, 1, 1))
        starts = keypoints[:, :, :self.spiking_segment_count]
        ends = keypoints[:, :, 1:]
        residual = slopes * t + intercepts
        validity = ((t >= starts) & (t < ends)) | ((t == 1) & (ends == 1))
        left_margin = torch.sum((t - starts) * validity, dim=2)
        right_margin = torch.sum((ends - t) * validity, dim=2)
        left_weight = F.sigmoid(-self.temp * left_margin)
        right_weight = F.sigmoid(-self.temp * right_margin)
        left_validity = torch.cat([validity[:, :, :1], validity[:, :, :-1]],
                                  dim=2)
        right_validity = torch.cat([validity[:, :, 1:], validity[:, :, -1:]],
                                   dim=2)
        self_weight = -(left_weight + right_weight - 1)
        result = left_weight * torch.sum(residual * left_validity, dim=2) + \
                 self_weight * torch.sum(residual * validity, dim=2) + \
                 right_weight * torch.sum(residual * right_validity, dim=2)

        # apply normalization
        if self.norm == 'constant':
            integral = 0.5 * slopes * (ends * ends - starts * starts) + \
                       intercepts * weights
            integral = torch.sum(integral, dim=2)
            norm_target = self.norm_net(x)
            result = result - integral + norm_target
        return result

class SpikingConvTranspose2d(nn.ConvTranspose2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 output_padding=0, groups=1, bias=True, dilation=1,
                 padding_mode='zeros', spiking_segment_count=3, norm='none',
                 temp=50):
        super(SpikingConvTranspose2d, self).__init__(\
                in_channels, out_channels * spiking_segment_count * 3, kernel_size,
                stride=stride, padding=padding, output_padding=output_padding,
                groups=groups, bias=bias, dilation=dilation,
                padding_mode=padding_mode)
        if norm != 'none':
            self.norm_net = nn.ConvTranspose2d(in_channels, out_channels, kernel_size,
                                               stride=stride, padding=padding,
                                               output_padding=output_padding,
                                               groups=groups, bias=bias,
                                               dilation=dilation,
                                               padding_mode=padding_mode)
        self.out_channels = out_channels
        self.spiking_segment_count = spiking_segment_count
        self.norm = norm
        self.softmax = nn.Softmax(dim=2)
        self.temp = temp

    def forward(self, x, t):
        # predict segment parameters
        params = super().forward(x)
        bs, _, h, w = params.shape
        params = params.reshape((bs,
                                 self.out_channels,
                                 self.spiking_segment_count,
                                 3,
                                 h,
                                 w))
        weights = self.softmax(params[:, :, :, 0])
        keypoints = torch.cat([torch.zeros_like(weights[:, :, :1]),
                               torch.cumsum(weights, dim=2)], dim=2)
        keypoints = keypoints * 2 - 1
        slopes = torch.tanh(params[:, :, :, 1])
        intercepts = params[:, :, :, 2]

        # assemble kernels
        t = t.reshape((bs, 1, 1, 1, 1))
        starts = keypoints[:, :, :self.spiking_segment_count]
        ends = keypoints[:, :, 1:]
        residual = slopes * t + intercepts
        validity = ((t >= starts) & (t < ends)) | ((t == 1) & (ends == 1))
        left_margin = torch.sum((t - starts) * validity, dim=2)
        right_margin = torch.sum((ends - t) * validity, dim=2)
        left_weight = F.sigmoid(-self.temp * left_margin)
        right_weight = F.sigmoid(-self.temp * right_margin)
        left_validity = torch.cat([validity[:, :, :1], validity[:, :, :-1]],
                                  dim=2)
        right_validity = torch.cat([validity[:, :, 1:], validity[:, :, -1:]],
                                   dim=2)
        self_weight = -(left_weight + right_weight - 1)
        result = left_weight * torch.sum(residual * left_validity, dim=2) + \
                 self_weight * torch.sum(residual * validity, dim=2) + \
                 right_weight * torch.sum(residual * right_validity, dim=2)

        # apply normalization
        if self.norm == 'constant':
            integral = 0.5 * slopes * (ends * ends - starts * starts) + \
                       intercepts * weights
            integral = torch.sum(integral, dim=2)
            norm_target = self.norm_net(x)
            result = result - integral + norm_target
        return result


class SpikingLinear(nn.Module):
    def __init__(self, in_features, out_features,
                 spiking_segment_count=10, bias=True, norm='constant', temp=50):
        super(SpikingLinear, self).__init__()
        self.param_net = nn.Linear(in_features,
                                   out_features * spiking_segment_count * 3,
                                   bias=bias)
        if norm != 'none':
            self.norm_net = nn.Linear(in_features, out_features)
        self.softmax = nn.Softmax(dim=1)
        self.out_features = out_features
        self.spiking_segment_count = spiking_segment_count
        self.norm = norm
        self.temp = temp

    def forward(self, x, t, norm_target=None):
        bs = x.shape[0]
        t = t.reshape((-1,))
        # predict line segment parameters
        params = self.param_net(x)
        params = params.reshape(bs, self.spiking_segment_count, self.out_features, 3)
        
        # assemble keypoints
        weights = self.softmax(params[:, :, :, 0])
        keypoints = torch.cat([torch.zeros_like(weights[:, :1]),
                               torch.cumsum(weights, dim=1)], dim=1)
        keypoints = keypoints * 2 - 1
        # slope and intercept
        slopes = torch.tanh(params[:, :, :, 1])
        intercepts = params[:, :, :, 2]

        t = t.unsqueeze(dim=-1).unsqueeze(dim=-1)

        starts = keypoints[:, :self.spiking_segment_count]
        ends = keypoints[:, 1:]
        residual = slopes * t + intercepts
        validity = ((t >= starts) & (t < ends)) | ((t == 1) & (ends == 1))
        left_margin = torch.sum((t - starts) * validity, dim=1)
        right_margin = torch.sum((ends - t) * validity, dim=1)
        left_weight = F.sigmoid(-self.temp * left_margin)
        right_weight = F.sigmoid(-self.temp * right_margin)
        left_validity = torch.cat([validity[:, :1], validity[:, :-1]],
                                  dim=1)
        right_validity = torch.cat([validity[:, 1:], validity[:, -1:]],
                                   dim=1)
        self_weight = -(left_weight + right_weight - 1)
        result = left_weight * torch.sum(residual * left_validity, dim=1) + \
                 self_weight * torch.sum(residual * validity, dim=1) + \
                 right_weight * torch.sum(residual * right_validity, dim=1)

        if self.norm != 'none':
            integral = 0.5 * slopes * (ends * ends - starts * starts) + \
                       intercepts * weights
            integral = torch.sum(integral, dim=1) / 2
            if self.norm == 'constant':
                norm_target = self.norm_net(x)
            result = result - integral + norm_target
        return result

if __name__ == '__main__':
    x = torch.zeros((3, 2, 180, 240)).cuda()
    sconv = SpikingConv2d(2, 4, 3, spiking_segment_count=8, padding=1, stride=2).cuda()
    t = torch.zeros((3,)).cuda()
    x1 = sconv(x, t)
    print(x1.shape)
    x = torch.zeros((3, 2, 90, 120)).cuda()
    sconvt = SpikingConvTranspose2d(2, 4, 5, stride=2, padding=2, output_padding=(1, 1), spiking_segment_count=8).cuda()
    x1 = sconvt(x, t)
    print(x1.shape)
    x = torch.zeros((3, 128)).cuda()
    slinear = SpikingLinear(128, 32, norm='constant').cuda()
    x1 = slinear(x, t)
    print(x1.shape)
