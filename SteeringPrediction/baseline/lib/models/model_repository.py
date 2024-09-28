import math
from torch import nn
import torch
from lib.models.resnet import resnet32
import pdb

class PredictNet(nn.Module):
    def __init__(self, use_radians=False):
        super(PredictNet, self).__init__()
        self.use_radians = use_radians
        self.resnet = resnet32()
        # change the number of input and output channels
        self.resnet.conv1 = nn.Conv2d(2, 16,
                                      kernel_size=3, stride=1, padding=1, bias=False)
        self.resnet.linear = nn.Linear(64, 1)
        for m in self.resnet.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                std = 2.0 / math.sqrt(fan_in)
                nn.init.normal_(m.weight, 0, std)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        self.criterion = nn.MSELoss()

    def forward(self, batch):
        aps = batch['aps']
        dvs = batch['dvs']
        net_in = torch.stack([aps, dvs], axis=1)
        angle_pred = self.resnet(net_in).view((-1,))
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
