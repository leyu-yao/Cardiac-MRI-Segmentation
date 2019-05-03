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

class DoubleConv2d(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv2d, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        return self.conv(input)



class DenseBlock2dA(nn.Module):
    def __init__(self, ch):
        super(DenseBlock2dA, self).__init__()

        
        self.conv = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1),
            nn.BatchNorm2d(ch),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(ch, ch, 3, padding=1),
            nn.BatchNorm2d(ch),
            nn.ReLU(inplace=True)
            )

    
    def forward(self, x):
        return x + self.conv(x)

class DenseBlock2dB(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DenseBlock2dB, self).__init__()

        
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

        self.conv2 = nn.Sequential(            
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
            )

    
    def forward(self, x):
        return self.conv1(x) + self.conv2(self.conv1(x))
    
class InceptionX2d(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch):
        super(InceptionX2d, self).__init__()
        self.conv3 = nn.Conv2d(in_ch, mid_ch, 3, padding=1)
        self.conv5 = nn.Conv2d(in_ch, mid_ch, 5, padding=2)
        self.conv7 = nn.Conv2d(in_ch, mid_ch, 7, padding=3)
        self.bn1 = nn.BatchNorm2d(3*mid_ch)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(3*mid_ch, out_ch, 1)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu2 = nn.ReLU(inplace=True)
        
    def forward(self, x):
        x3 = self.conv3(x)
        x5 = self.conv5(x)
        x7 = self.conv7(x)
        x = torch.cat([x3, x5, x7],dim=1)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv1(x)
        x = self.bn2(x)
        x = self.relu2(x)
        return x
