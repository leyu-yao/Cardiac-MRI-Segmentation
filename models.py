# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 14:11:50 2019
@author: Dynasting
"""

import torch.nn as nn
import torch
from torch import autograd

from model_components import DoubleConv3d, ConvBlock

class Unet3d(nn.Module):
    def __init__(self,in_ch,out_ch):
        super(Unet3d, self).__init__()
        self.name = "Unet3d"
        self.conv1 = DoubleConv3d(in_ch, 32, 64)
        self.pool1 = nn.MaxPool3d(2)
        
        self.conv2 = DoubleConv3d(64, 64, 128)
        self.pool2 = nn.MaxPool3d(2)
        
        self.conv3 = DoubleConv3d(128, 128, 256)
        self.pool3 = nn.MaxPool3d(2)
        
        self.conv4 = DoubleConv3d(256, 256, 512)
        
        self.up1 = nn.ConvTranspose3d(512, 512, 2, stride=2)
        

        self.conv5 = DoubleConv3d(256+512, 256, 256)
        self.up2 = nn.ConvTranspose3d(256, 256, 2, stride=2)
        
        self.conv6 = DoubleConv3d(128+256, 128, 128)

        self.up3 = nn.ConvTranspose3d(128, 128, 2, stride=2)
        
        self.conv7 = DoubleConv3d(64+128, 64, 64)
        self.conv8 = nn.Conv3d(64,out_ch, 1)

        self.out_ch = out_ch
        


    def forward(self,x):
        #print(x.size())
        c1=self.conv1(x)
        p1=self.pool1(c1)
        c2=self.conv2(p1);del p1
        p2=self.pool2(c2)
        c3=self.conv3(p2);del p2
        p3=self.pool3(c3)
        c4=self.conv4(p3);del p3
        up_1 = self.up1(c4);del c4
        merge1 = torch.cat([up_1, c3], dim=1);del up_1;del c3
        c5=self.conv5(merge1);del merge1
        up_2 = self.up2(c5);del c5
        merge2 = torch.cat([up_2, c2], dim=1);del up_2;del c2
        c6=self.conv6(merge2);del merge2
        up_3 = self.up3(c6);del c6
        merge3 = torch.cat([up_3, c1], dim=1);del up_3;del c1
        c7=self.conv7(merge3);del merge3
        c8=self.conv8(c7);del c7
        
        if self.out_ch == 1:
            out = nn.Sigmoid()(c8);del c8
        else:
            out = nn.Softmax(dim=1)(c8);del c8
        return out
    

class DSUnet3d(nn.Module):
    def __init__(self,in_ch,out_ch):
        super(DSUnet3d, self).__init__()
        self.name = "DSUnet3d"

        self.relu = nn.ReLU(inplace=True)

        # %% downsample path

        self.conv1 = nn.Conv3d(in_ch, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm3d(64)
        self.pool1 = nn.MaxPool3d(2)

        self.conv2 = nn.Conv3d(64, 128, 3, padding=1)
        self.bn2 = nn.BatchNorm3d(128)
        self.pool2 = nn.MaxPool3d(2)

        self.conv3a = nn.Conv3d(128, 256, 3, padding=1)
        self.bn3a = nn.BatchNorm3d(256)
        self.conv3b = nn.Conv3d(256, 256, 3, padding=1)
        self.bn3b = nn.BatchNorm3d(256)        
        self.pool3 = nn.MaxPool3d(2)

        self.conv4a = nn.Conv3d(256, 512, 3, padding=1)
        self.bn4a = nn.BatchNorm3d(512)
        self.conv4b = nn.Conv3d(512, 512, 3, padding=1)
        self.bn4b = nn.BatchNorm3d(512)        
        self.pool4 = nn.MaxPool3d(2)

        self.conv5a = nn.Conv3d(512, 512, 3, padding=1)
        self.bn5a = nn.BatchNorm3d(512)
        self.conv5b = nn.Conv3d(512, 512, 3, padding=1)
        self.bn5b = nn.BatchNorm3d(512) 

        # %% upsample path
        self.unpool1 = nn.ConvTranspose3d(512, 512, 2, stride=2)

        self.conv6 = nn.Conv3d(1024, 256, 3, padding=1)
        self.bn6 = nn.BatchNorm3d(256)

        self.unpool2 = nn.ConvTranspose3d(256, 256, 2, stride=2)

        self.conv7 = nn.Conv3d(512, 128, 3, padding=1)
        self.bn7 = nn.BatchNorm3d(128)

        self.unpool3 = nn.ConvTranspose3d(128, 128, 2, stride=2)

        self.conv8 = nn.Conv3d(256, 128, 3, padding=1)
        self.bn8 = nn.BatchNorm3d(128)

        self.unpool4 = nn.ConvTranspose3d(128, 64, 2, stride=2)

        self.conv9a = nn.Conv3d(128, 32, 3, padding=1)
        self.bn9a = nn.BatchNorm3d(32)

        self.conv9b = nn.Conv3d(32, out_ch, 3, padding=1)
        #self.bn9b = nn.BatchNorm3d(out_ch)

        # %%side outputs
        self.side1 = nn.Conv3d(256, out_ch, 3, padding=1) #1/8
        #self.bns1 = nn.BatchNorm3d(out_ch)
        self.side2 = nn.Conv3d(128, out_ch, 3, padding=1) #1/4
        #self.bns2 = nn.BatchNorm3d(out_ch)
        self.side3 = nn.Conv3d(128, out_ch, 3, padding=1) #1/2
        #self.bns3 = nn.BatchNorm3d(out_ch)


    def forward(self,x):
        # %% downsample
        c1 = self.relu(self.bn1(self.conv1(x)))
        p1 = self.pool1(c1)

        c2 = self.relu(self.bn2(self.conv2(p1)))
        p2 = self.pool2(c2)

        c3 = self.relu(self.bn3b(self.conv3b(self.relu(self.bn3a(self.conv3a(p2))))))
        p3 = self.pool3(c3)

        c4 = self.relu(self.bn4b(self.conv4b(self.relu(self.bn4a(self.conv4a(p3))))))
        p4 = self.pool4(c4)

        c5 = self.relu(self.bn5b(self.conv5b(self.relu(self.bn5a(self.conv5a(p4))))))

        # %% up path
        u1 = self.unpool1(c5)

        c6 = self.relu(self.bn6(self.conv6(torch.cat([u1, c4], dim=1))))

        u2 = self.unpool2(c6)

        c7 = self.relu(self.bn7(self.conv7(torch.cat([u2, c3], dim=1))))

        u3 = self.unpool3(c7)

        c8 = self.relu(self.bn8(self.conv8(torch.cat([u3, c2], dim=1))))

        u4 = self.unpool4(c8)

        #c9 = self.relu(self.bn9b(self.conv9b(self.relu(self.bn9a(self.conv9a(torch.cat([u4, c1], dim=1)))))))
        c9 = self.relu(self.conv9b(self.relu(self.bn9a(self.conv9a(torch.cat([u4, c1], dim=1))))))

        out  = nn.Softmax(dim=1)(c9)

        # %% side path
        side_out_1  = self.side1(c6)
        side_out_2  = self.side2(c7)
        side_out_3  = self.side3(c8)
        return out, side_out_1, side_out_2, side_out_3

        
class SmallUnet3d(nn.Module):
    def __init__(self,in_ch,out_ch):
        super(SmallUnet3d, self).__init__()
        self.name = "SmallUnet3d"
        self.conv1 = DoubleConv3d(in_ch, 16, 32)
        self.pool1 = nn.MaxPool3d(2)
        
        self.conv2 = DoubleConv3d(32, 32, 64)
        self.pool2 = nn.MaxPool3d(2)
        
        self.conv3 = DoubleConv3d(64, 64, 128)
        self.pool3 = nn.MaxPool3d(2)
        
        self.conv4 = DoubleConv3d(128, 128, 256)
        
        self.up1 = nn.ConvTranspose3d(256, 256, 2, stride=2)
        

        self.conv5 = DoubleConv3d(128+256, 128, 128)
        self.up2 = nn.ConvTranspose3d(128, 128, 2, stride=2)
        
        self.conv6 = DoubleConv3d(64+128, 64, 64)

        self.up3 = nn.ConvTranspose3d(64, 64, 2, stride=2)
        
        self.conv7 = DoubleConv3d(32+64, 32, 32)
        self.conv8 = nn.Conv3d(32,out_ch, 1)
        


    def forward(self,x):
        #print(x.size())
        c1=self.conv1(x)
        p1=self.pool1(c1)
        c2=self.conv2(p1);del p1
        p2=self.pool2(c2)
        c3=self.conv3(p2);del p2
        p3=self.pool3(c3)
        c4=self.conv4(p3);del p3
        up_1 = self.up1(c4);del c4
        merge1 = torch.cat([up_1, c3], dim=1);del up_1;del c3
        c5=self.conv5(merge1);del merge1
        up_2 = self.up2(c5);del c5
        merge2 = torch.cat([up_2, c2], dim=1);del up_2;del c2
        c6=self.conv6(merge2);del merge2
        up_3 = self.up3(c6);del c6
        merge3 = torch.cat([up_3, c1], dim=1);del up_3;del c1
        c7=self.conv7(merge3);del merge3
        c8=self.conv8(c7);del c7
        
        if self.out_ch == 1:
            out = nn.Sigmoid()(c8);del c8
        else:
            out = nn.Softmax(dim=1)(c8);del c8
        return out
    
class SmallSMallUnet3d(nn.Module):
    def __init__(self,in_ch,out_ch):
        super(SmallSMallUnet3d, self).__init__()
        self.name = "SmallSmallUnet3d"
        self.conv1 = DoubleConv3d(in_ch, 8, 16)
        self.pool1 = nn.MaxPool3d(2)
        
        self.conv2 = DoubleConv3d(16, 16, 32)
        self.pool2 = nn.MaxPool3d(2)
        
        self.conv3 = DoubleConv3d(32, 32, 64)
        self.pool3 = nn.MaxPool3d(2)
        
        self.conv4 = DoubleConv3d(64, 64, 128)
        
        self.up1 = nn.ConvTranspose3d(128, 128, 2, stride=2)
        

        self.conv5 = DoubleConv3d(64+128, 64, 64)
        self.up2 = nn.ConvTranspose3d(64, 64, 2, stride=2)
        
        self.conv6 = DoubleConv3d(32+64, 32, 32)

        self.up3 = nn.ConvTranspose3d(32, 32, 2, stride=2)
        
        self.conv7 = DoubleConv3d(16+32, 16, 16)
        self.conv8 = nn.Conv3d(16,out_ch, 1)
        


    def forward(self,x):
        #print(x.size())
        c1=self.conv1(x)
        p1=self.pool1(c1)
        c2=self.conv2(p1);del p1
        p2=self.pool2(c2)
        c3=self.conv3(p2);del p2
        p3=self.pool3(c3)
        c4=self.conv4(p3);del p3
        up_1 = self.up1(c4);del c4
        merge1 = torch.cat([up_1, c3], dim=1);del up_1;del c3
        c5=self.conv5(merge1);del merge1
        up_2 = self.up2(c5);del c5
        merge2 = torch.cat([up_2, c2], dim=1);del up_2;del c2
        c6=self.conv6(merge2);del merge2
        up_3 = self.up3(c6);del c6
        merge3 = torch.cat([up_3, c1], dim=1);del up_3;del c1
        c7=self.conv7(merge3);del merge3
        c8=self.conv8(c7);del c7
        
        if self.out_ch == 1:
            out = nn.Sigmoid()(c8);del c8
        else:
            out = nn.Softmax(dim=1)(c8);del c8
        return out


