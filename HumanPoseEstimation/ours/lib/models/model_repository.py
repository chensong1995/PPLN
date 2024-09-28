from torch import nn
import torch
from torch.nn import functional as F

from lib.models.modules import SpikingConv2d, SpikingConvTranspose2d

import pdb

class PredictNet(nn.Module):
    def __init__(self, group_size=10, norm='constant', temp=None):
        super(PredictNet, self).__init__()
        self.sconv1 = SpikingConv2d(group_size, 4, 1, stride=1, bias=False,
                                    spiking_segment_count=3, norm=norm, temp=temp)
        self.sconv2 = SpikingConv2d(4, 4, 1, stride=1, bias=False,
                                    spiking_segment_count=3, norm=norm, temp=temp)
        self.sconv3 = SpikingConv2d(4, 4, 1, stride=1, bias=False,
                                    spiking_segment_count=3, norm=norm, temp=temp)

        self.conv1 = nn.Conv2d(4, 16, 3, stride=1, dilation=1,
                               padding=(1, 1), bias=False)
        self.conv2 = nn.Conv2d(16, 32, 3, stride=1, dilation=1,
                               padding=(1, 1), bias=False)
        self.conv3 = nn.Conv2d(32, 32, 3, stride=1, dilation=1,
                               padding=(1, 1), bias=False)
        self.conv4 = nn.Conv2d(32, 32, 3, stride=1, dilation=1,
                               padding=(1, 1), bias=False)
        self.conv5 = nn.Conv2d(32, 32, 3, stride=1, dilation=2,
                               padding=(2, 2), bias=False)
        self.conv6 = nn.Conv2d(32, 32, 3, stride=1, dilation=2,
                               padding=(2, 2), bias=False)
        self.conv7 = nn.Conv2d(32, 64, 3, stride=1, dilation=2,
                               padding=(2, 2), bias=False)
        self.conv8 = nn.Conv2d(64, 64, 3, stride=1, dilation=2,
                               padding=(2, 2), bias=False)

        self.sconv4 = SpikingConv2d(64, 32, 1, stride=1, bias=False,
                                    spiking_segment_count=3, norm=norm, temp=temp)
        self.sconv5 = SpikingConv2d(32, 32, 1, stride=1, bias=False,
                                    spiking_segment_count=3, norm=norm, temp=temp)
        self.sconv6 = SpikingConv2d(32, 64, 1, stride=1, bias=False,
                                    spiking_segment_count=3, norm=norm, temp=temp)

        self.deconv1 = nn.ConvTranspose2d(64, 32, 3, stride=2, dilation=1,
                                          padding=(1, 1), output_padding=(1, 1),
                                          bias=False)
        self.conv9 = nn.Conv2d(32, 32, 3, stride=1, dilation=2,
                               padding=(2, 2), bias=False)
        self.conv10 = nn.Conv2d(32, 32, 3, stride=1, dilation=2,
                                padding=(2, 2), bias=False)
        self.conv11 = nn.Conv2d(32, 32, 3, stride=1, dilation=2,
                                padding=(2, 2), bias=False)
        self.conv12 = nn.Conv2d(32, 32, 3, stride=1, dilation=2,
                                padding=(2, 2), bias=False)
        self.deconv2 = nn.ConvTranspose2d(32, 16, 3, stride=2, dilation=1,
                                          padding=(1, 1), output_padding=(1, 1),
                                          bias=False)
        self.conv13 = nn.Conv2d(16, 16, 3, stride=1, dilation=1, padding=(1, 1),
                                bias=False)
        self.conv14 = nn.Conv2d(16, 16, 3, stride=1, dilation=1, padding=(1, 1),
                                bias=False)
        self.conv15 = nn.Conv2d(16, 13, 3, stride=1, dilation=1, padding=(1, 1),
                                bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.criterion = nn.MSELoss()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, batch):
        x_in = batch['event']
        t = batch['timestamp']
        x1 = self.relu(self.sconv1(x_in, t))
        x2 = self.relu(self.sconv2(x1, t))
        x3 = self.relu(self.sconv3(x2, t))
        x4 = self.maxpool(self.relu(self.conv1(x3)))
        x5 = self.relu(self.conv2(x4))
        x6 = self.relu(self.conv3(x5))
        x7 = self.maxpool(self.relu(self.conv4(x6)))
        x8 = self.relu(self.conv5(x7))
        x9 = self.relu(self.conv6(x8))
        x10 = self.relu(self.conv7(x9))
        x11 = self.relu(self.conv8(x10))

        x12 = self.relu(self.sconv4(x11, t))
        x13 = self.relu(self.sconv5(x12, t))
        x14 = self.relu(self.sconv6(x13, t) + x11)

        x15 = self.relu(self.deconv1(x14))
        x16 = self.relu(self.conv9(x15))
        x17 = self.relu(self.conv10(x16))
        x18 = self.relu(self.conv11(x17))
        x19 = self.relu(self.conv12(x18))
        x20 = self.relu(self.deconv2(x19))
        x21 = self.relu(self.conv13(x20))
        x22 = self.relu(self.conv14(x21))
        x23 = self.relu(self.conv15(x22))

        heatmap_pred = x23
        heatmap_gt = batch['heatmap']
        loss = self.criterion(heatmap_pred, heatmap_gt)
        return heatmap_pred, loss
