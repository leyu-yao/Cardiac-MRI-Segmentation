# -*- coding: utf-8 -*-
"""
Created on Tue Apr 09 14:11:50 2019
@author: Dynasting
"""

import torch.nn as nn
import torch
from torch import autograd


class DoubleConv3d(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch):
        super(DoubleConv3d, self).__init__()
        self.conv1 = nn.Conv3d(in_ch, mid_ch, 3, padding=1)
        #self.bn1 = nn.BatchNorm3d(mid_ch)
        self.rl1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(mid_ch, out_ch, 3, padding=1)
        #self.bn2 = nn.BatchNorm3d(out_ch)
        self.rl2 = nn.ReLU(inplace=True)

    def forward(self, x):
        y = self.conv1(x)
        #y = self.bn1(y)
        y = self.rl1(y)
        y = self.conv2(y)
        #y = self.bn2(y)
        y = self.rl2(y)
        return y
