import math

import torch
import torch.nn as nn

from lib.modules import SpikingConv2d, SpikingConvTranspose2d, SpikingLinear

import pdb

class SpikingUNet(nn.Module):
    def __init__(self,
                 in_channels=1,
                 out_channels=10,
                 norm='none',
                 temp=None):
        super(SpikingUNet, self).__init__()
        self.en_conv1 = SpikingConv2d(in_channels, 64, 5, stride=2, padding=2,
                                      spiking_segment_count=3, norm=norm, temp=temp)
        self.en_conv2 = SpikingConv2d(64, 128, 5, stride=2, padding=2,
                                      spiking_segment_count=3, norm=norm, temp=temp)
        self.en_conv3 = SpikingConv2d(128, 256, 5, stride=2, padding=2,
                                      spiking_segment_count=3, norm=norm, temp=temp)
        self.en_conv4 = SpikingConv2d(256, 512, 5, stride=2, padding=2,
                                      spiking_segment_count=3, norm=norm, temp=temp)
        self.res1_1 = SpikingConv2d(512, 512, 3, stride=1, padding=1, bias=False,
                                    spiking_segment_count=3, norm=norm, temp=temp)
        self.res1_2 = SpikingConv2d(512, 512, 3, stride=1, padding=1, bias=False,
                                    spiking_segment_count=3, norm=norm, temp=temp)
        self.res2_1 = SpikingConv2d(512, 512, 3, stride=1, padding=1, bias=False,
                                    spiking_segment_count=3, norm=norm, temp=temp)
        self.res2_2 = SpikingConv2d(512, 512, 3, stride=1, padding=1, bias=False,
                                    spiking_segment_count=3, norm=norm, temp=temp)
        self.de_conv1 = SpikingConvTranspose2d(512, 256, 5, stride=2, padding=2,
                                               output_padding=(0, 1), bias=False,
                                               spiking_segment_count=3, norm=norm,
                                               temp=temp)
        self.de_conv2 = SpikingConvTranspose2d(256 + 256, 128, 5, stride=2,
                                               padding=2, output_padding=(0, 1),
                                               bias=False, spiking_segment_count=3,
                                               norm=norm, temp=temp)
        self.de_conv3 = SpikingConvTranspose2d(128 + 128, 64, 5, stride=2,
                                               padding=2, output_padding=(1, 1),
                                               bias=False, spiking_segment_count=3,
                                               norm=norm, temp=temp)
        self.de_conv4 = SpikingConvTranspose2d(64 + 64, 32, 5, stride=2, padding=2,
                                               output_padding=(1, 1), bias=False,
                                               spiking_segment_count=3, norm=norm,
                                               temp=temp)
        self.pred = SpikingConv2d(32 + in_channels, out_channels, 1, stride=1,
                                  spiking_segment_count=3, norm=norm, temp=temp)
        self.relu = nn.ReLU(inplace=True)
        self.bn512 = nn.BatchNorm2d(512)

    def forward(self, x, t):
        x1 = self.relu(self.en_conv1(x, t))
        x2 = self.relu(self.en_conv2(x1, t))
        x3 = self.relu(self.en_conv3(x2, t))
        x4 = self.relu(self.en_conv4(x3, t))
        x5_1 = self.relu(self.bn512(self.res1_1(x4, t)))
        x5_2 = self.bn512(self.res1_2(x5_1, t))
        x5 = self.relu(x5_2 + x5_1)
        x6_1 = self.relu(self.bn512(self.res2_1(x5, t)))
        x6_2 = self.bn512(self.res2_2(x6_1, t))
        x6 = self.relu(x6_2 + x6_1)
        x7 = self.relu(self.de_conv1(x6, t))
        x8 = self.relu(self.de_conv2(torch.cat([x7, x3], dim=1), t))
        x9 = self.relu(self.de_conv3(torch.cat([x8, x2], dim=1), t))
        x10 = self.relu(self.de_conv4(torch.cat([x9, x1], dim=1), t))
        x_out = self.pred(torch.cat([x10, x], dim=1), t)
        return x_out

class UNet(nn.Module):
    def __init__(self,
                 in_channels=1,
                 out_channels=10,
                 multiplier=3):
        super(UNet, self).__init__()
        self.en_conv1 = nn.Conv2d(in_channels, 64 * multiplier, 5, stride=2, padding=2)
        self.en_conv2 = nn.Conv2d(64 * multiplier, 128 * multiplier, 5, stride=2, padding=2)
        self.en_conv3 = nn.Conv2d(128 * multiplier, 256 * multiplier, 5, stride=2, padding=2)
        self.en_conv4 = nn.Conv2d(256 * multiplier, 512 * multiplier, 5, stride=2, padding=2)
        self.res1_1 = nn.Conv2d(512 * multiplier, 512 * multiplier, 3, stride=1, padding=1, bias=False)
        self.res1_2 = nn.Conv2d(512 * multiplier, 512 * multiplier, 3, stride=1, padding=1, bias=False)
        self.res2_1 = nn.Conv2d(512 * multiplier, 512 * multiplier, 3, stride=1, padding=1, bias=False)
        self.res2_2 = nn.Conv2d(512 * multiplier, 512 * multiplier, 3, stride=1, padding=1, bias=False)
        self.de_conv1 = nn.ConvTranspose2d(512 * multiplier, 256 * multiplier, 5, stride=2, padding=2,
                                           output_padding=(0, 1), bias=False)
        self.de_conv2 = nn.ConvTranspose2d((256 + 256)  * multiplier, 128  * multiplier, 5, stride=2,
                                           padding=2, output_padding=(0, 1),
                                           bias=False)
        self.de_conv3 = nn.ConvTranspose2d((128 + 128) * multiplier, 64 * multiplier, 5, stride=2,
                                           padding=2, output_padding=(1, 1),
                                           bias=False)
        self.de_conv4 = nn.ConvTranspose2d((64 + 64) * multiplier, 32 * multiplier, 5, stride=2, padding=2,
                                           output_padding=(1, 1), bias=False)
        self.pred = nn.Conv2d(32 * multiplier + in_channels, out_channels, 1, stride=1)
        self.relu = nn.ReLU(inplace=True)
        self.bn512 = nn.BatchNorm2d(512 * multiplier)

    def forward(self, x, t):
        x1 = self.relu(self.en_conv1(x))
        x2 = self.relu(self.en_conv2(x1))
        x3 = self.relu(self.en_conv3(x2))
        x4 = self.relu(self.en_conv4(x3))
        x5_1 = self.relu(self.bn512(self.res1_1(x4)))
        x5_2 = self.bn512(self.res1_2(x5_1))
        x5 = self.relu(x5_2 + x5_1)
        x6_1 = self.relu(self.bn512(self.res2_1(x5)))
        x6_2 = self.bn512(self.res2_2(x6_1))
        x6 = self.relu(x6_2 + x6_1)
        x7 = self.relu(self.de_conv1(x6))
        x8 = self.relu(self.de_conv2(torch.cat([x7, x3], dim=1)))
        x9 = self.relu(self.de_conv3(torch.cat([x8, x2], dim=1)))
        x10 = self.relu(self.de_conv4(torch.cat([x9, x1], dim=1)))
        x_out = self.pred(torch.cat([x10, x], dim=1))
        return x_out

class PredNet(nn.Module):
    def __init__(self,
                 event_channels=26,
                 constant_norm=False,
                 sconv=False,
                 temp=None):
        super(PredNet, self).__init__()
        self.event_channels = event_channels
        self.sconv = sconv
        # prediction networks
        if sconv:
            self.unet = SpikingUNet(in_channels=1 + event_channels,
                                    out_channels=1,
                                    norm='constant' if constant_norm else 'none',
                                    temp=temp)
        else:
            self.unet = UNet(in_channels=1 + event_channels,
                             out_channels=1)

    def forward(self, batch):
        sharp_frame = self.unet(torch.cat([batch['blurry_frame'],
                                           batch['event_map']], dim=1),
                                batch['timestamps'])
        return { 'sharp_frame': sharp_frame }
