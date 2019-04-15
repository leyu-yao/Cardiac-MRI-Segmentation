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
            #nn.BatchNorm3d(mid_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(mid_ch, out_ch, 3, padding=1),
            #nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        return self.conv(input)