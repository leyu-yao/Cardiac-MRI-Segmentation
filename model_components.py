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
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, mid_ch, 3, padding=1),
            nn.BatchNorm3d(mid_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(mid_ch, out_ch, 3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        return self.conv(input)


class DenseBlock(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch):
        super(DenseBlock, self).__init__()

        self.conv1 = nn.Conv3d(in_ch, mid_ch, 3, padding=1)
        self.rl1 = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm3d(mid_ch)

        self.conv2 = nn.Conv3d(mid_ch, out_ch, 3, padding=1)
        self.rl2 = nn.ReLU(inplace=True)
        self.bn2 = nn.BatchNorm3d(out_ch)

    def forward(self, x):
        m =  self.rl1(self.bn1(self.conv1(x)))
        return self.rl1(self.bn1(self.conv1(m))) + self.rl1(self.bn1(self.conv1(x)))
    
if __name__ == '__main__':
    m=DenseBlock(1,3,5).to('cuda')
    a=torch.rand(1,1,8,8,8).to('cuda')
    b=m(a)
    