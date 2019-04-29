# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 14:11:50 2019
@author: Dynasting
"""

import torch.nn as nn
import torch
from torch import autograd

from model_components import  DoubleConv3d, DenseBlock



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




class Unet3d_depth5(nn.Module):
    def __init__(self,in_ch,out_ch):
        super(Unet3d_depth5, self).__init__()
        self.name = "Unet3d_depth5"

        self.conv1 = DoubleConv3d(in_ch, 8, 16)
        self.pool1 = nn.MaxPool3d(2)
        
        self.conv2 = DoubleConv3d(16, 16, 32)
        self.pool2 = nn.MaxPool3d(2)
        
        self.conv3 = DoubleConv3d(32, 32, 64)
        self.pool3 = nn.MaxPool3d(2)

        self.conv4 = DoubleConv3d(64, 64, 128)
        self.pool4 = nn.MaxPool3d(2)
        
        self.conv5 = DoubleConv3d(128, 128, 256)
        self.pool5 = nn.MaxPool3d(2)

        self.conv6 = DoubleConv3d(256, 256, 512)

        
        self.up1 = nn.ConvTranspose3d(512, 512, 2, stride=2)
        

        self.conv7 = DoubleConv3d(256+512, 256, 256)
        self.up2 = nn.ConvTranspose3d(256, 256, 2, stride=2)
        
        self.conv8 = DoubleConv3d(128+256, 128, 128)
        self.up3 = nn.ConvTranspose3d(128, 128, 2, stride=2)

        self.conv9 = DoubleConv3d(64+128, 64, 64)
        self.up4 = nn.ConvTranspose3d(64, 64, 2, stride=2)

        self.conv10 = DoubleConv3d(32+64, 32, 32)
        self.up5 = nn.ConvTranspose3d(32, 32, 2, stride=2)
        
        self.conv11 = DoubleConv3d(16+32, 16, 16)

        self.conv12 = nn.Conv3d(16, out_ch, 1)

        self.out_ch = out_ch
        


    def forward(self,x):

        #print(x.size())


        c1=self.conv1(x)
        p1=self.pool1(c1)

        c2=self.conv2(p1)
        p2=self.pool2(c2)

        c3=self.conv3(p2)
        p3=self.pool3(c3)

        c4=self.conv4(p3)
        p4=self.pool4(c4)

        c5=self.conv5(p4)
        p5=self.pool5(c5)

        c6=self.conv6(p5)

        up_1 = self.up1(c6)
        merge1 = torch.cat([up_1, c5], dim=1)
        c7=self.conv7(merge1)


        up_2 = self.up2(c7)
        merge2 = torch.cat([up_2, c4], dim=1)
        c8=self.conv8(merge2)

        up_3 = self.up3(c8)
        merge3 = torch.cat([up_3, c3], dim=1)
        c9=self.conv9(merge3)

        up_4 = self.up4(c9)
        merge4 = torch.cat([up_4, c2], dim=1)
        c10=self.conv10(merge4)

        up_5 = self.up5(c10)
        merge5 = torch.cat([up_5, c1], dim=1)
        c11=self.conv11(merge5)


        c12=self.conv12(c11)
        
        if self.out_ch == 1:
            out = nn.Sigmoid()(c12)
        else:
            out = nn.Softmax(dim=1)(c12)
        return out



class Unet3d_depth6(nn.Module):
    def __init__(self,in_ch,out_ch):
        super(Unet3d_depth6, self).__init__()
        self.name = "Unet3d_depth6"

        self.conv0 = DoubleConv3d(in_ch, 4, 8)
        self.pool0 = nn.MaxPool3d(2)

        self.conv1 = DoubleConv3d(8, 8, 16)
        self.pool1 = nn.MaxPool3d(2)
        
        self.conv2 = DoubleConv3d(16, 16, 32)
        self.pool2 = nn.MaxPool3d(2)
        
        self.conv3 = DoubleConv3d(32, 32, 64)
        self.pool3 = nn.MaxPool3d(2)

        self.conv4 = DoubleConv3d(64, 64, 128)
        self.pool4 = nn.MaxPool3d(2)
        
        self.conv5 = DoubleConv3d(128, 128, 256)
        self.pool5 = nn.MaxPool3d(2)

        self.conv6 = DoubleConv3d(256, 256, 512)

        
        self.up1 = nn.ConvTranspose3d(512, 512, 2, stride=2)
        

        self.conv7 = DoubleConv3d(256+512, 256, 256)
        self.up2 = nn.ConvTranspose3d(256, 256, 2, stride=2)
        
        self.conv8 = DoubleConv3d(128+256, 128, 128)
        self.up3 = nn.ConvTranspose3d(128, 128, 2, stride=2)

        self.conv9 = DoubleConv3d(64+128, 64, 64)
        self.up4 = nn.ConvTranspose3d(64, 64, 2, stride=2)

        self.conv10 = DoubleConv3d(32+64, 32, 32)
        self.up5 = nn.ConvTranspose3d(32, 32, 2, stride=2)
        
        self.conv11 = DoubleConv3d(16+32, 16, 16)
        self.up6 = nn.ConvTranspose3d(16, 16, 2, stride=2)

        self.conv12 = DoubleConv3d(8+16, 8, 8)

        self.conv13 = nn.Conv3d(8, out_ch, 1)

        self.out_ch = out_ch
        


    def forward(self,x):

        #print(x.size())


        c0=self.conv0(x)
        p0=self.pool0(c0)

        c1=self.conv1(p0)
        p1=self.pool1(c1)

        c2=self.conv2(p1)
        p2=self.pool2(c2)

        c3=self.conv3(p2)
        p3=self.pool3(c3)

        c4=self.conv4(p3)
        p4=self.pool4(c4)

        c5=self.conv5(p4)
        p5=self.pool5(c5)

        c6=self.conv6(p5)

        up_1 = self.up1(c6)
        merge1 = torch.cat([up_1, c5], dim=1)
        c7=self.conv7(merge1)


        up_2 = self.up2(c7)
        merge2 = torch.cat([up_2, c4], dim=1)
        c8=self.conv8(merge2)

        up_3 = self.up3(c8)
        merge3 = torch.cat([up_3, c3], dim=1)
        c9=self.conv9(merge3)

        up_4 = self.up4(c9)
        merge4 = torch.cat([up_4, c2], dim=1)
        c10=self.conv10(merge4)

        up_5 = self.up5(c10)
        merge5 = torch.cat([up_5, c1], dim=1)
        c11=self.conv11(merge5)

        up_6 = self.up6(c11)
        merge6 = torch.cat([up_6, c0], dim=1)
        c12=self.conv12(merge6)


        c13=self.conv13(c12)
        
        if self.out_ch == 1:
            out = nn.Sigmoid()(c13)
        else:
            out = nn.Softmax(dim=1)(c13)
        return out
