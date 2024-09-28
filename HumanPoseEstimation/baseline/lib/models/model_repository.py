from torch import nn
import torch
from torch.nn import functional as F
import pdb

class PredictNet(nn.Module):
    def __init__(self):
        super(PredictNet, self).__init__()
        # see table 3 in DHP19
        self.features = nn.Sequential(
                # layer 1
                nn.Conv2d(1, 16, kernel_size=3, stride=1, dilation=1, padding=(1, 1), bias=False),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                # layer 2
                nn.Conv2d(16, 32, kernel_size=3, stride=1, dilation=1, padding=(1, 1), bias=False),
                nn.ReLU(inplace=True),
                # layer 3
                nn.Conv2d(32, 32, kernel_size=3, stride=1, dilation=1, padding=(1, 1), bias=False),
                nn.ReLU(inplace=True),
                # layer 4
                nn.Conv2d(32, 32, kernel_size=3, stride=1, dilation=1, padding=(1, 1), bias=False),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                # layer 5
                nn.Conv2d(32, 64, kernel_size=3, stride=1, dilation=2, padding=(2, 2), bias=False),
                nn.ReLU(inplace=True),
                # layer 6
                nn.Conv2d(64, 64, kernel_size=3, stride=1, dilation=2, padding=(2, 2), bias=False),
                nn.ReLU(inplace=True),
                # layer 7
                nn.Conv2d(64, 64, kernel_size=3, stride=1, dilation=2, padding=(2, 2), bias=False),
                nn.ReLU(inplace=True),
                # layer 8
                nn.Conv2d(64, 64, kernel_size=3, stride=1, dilation=2, padding=(2, 2), bias=False),
                nn.ReLU(inplace=True),
                # layer 9
                nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, dilation=1, padding=(1, 1), output_padding=(1, 1), bias=False),
                nn.ReLU(inplace=True),
                # layer 10
                nn.Conv2d(32, 32, kernel_size=3, stride=1, dilation=2, padding=(2, 2), bias=False),
                nn.ReLU(inplace=True),
                # layer 11
                nn.Conv2d(32, 32, kernel_size=3, stride=1, dilation=2, padding=(2, 2), bias=False),
                nn.ReLU(inplace=True),
                # layer 12
                nn.Conv2d(32, 32, kernel_size=3, stride=1, dilation=2, padding=(2, 2), bias=False),
                nn.ReLU(inplace=True),
                # layer 13
                nn.Conv2d(32, 32, kernel_size=3, stride=1, dilation=2, padding=(2, 2), bias=False),
                nn.ReLU(inplace=True),
                # layer 14
                nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, dilation=1, padding=(1, 1), output_padding=(1, 1), bias=False),
                nn.ReLU(inplace=True),
                # layer 15
                nn.Conv2d(16, 16, kernel_size=3, stride=1, dilation=1, padding=(1, 1), bias=False),
                nn.ReLU(inplace=True),
                # layer 16
                nn.Conv2d(16, 16, kernel_size=3, stride=1, dilation=1, padding=(1, 1), bias=False),
                nn.ReLU(inplace=True),
                # layer 17
                nn.Conv2d(16, 13, kernel_size=3, stride=1, dilation=1, padding=(1, 1), bias=False),
                nn.ReLU(inplace=True)
                )
        self.criterion = nn.MSELoss()

    def forward(self, batch):
        heatmap_pred = self.features(batch['event'])
        heatmap_gt = batch['heatmap']
        loss = self.criterion(heatmap_pred, heatmap_gt)
        return heatmap_pred, loss
