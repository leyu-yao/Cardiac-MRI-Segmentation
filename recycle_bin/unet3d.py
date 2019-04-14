# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 14:11:50 2019

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
        
        #out = nn.Sigmoid()(c8);del c8
        out = nn.Softmax(dim=1)(c8);del c8
        return out

class SmallUnet3d(nn.Module):
    def __init__(self,in_ch,out_ch):
        super(SmallUnet3d, self).__init__()
        self.name = "SmallUnet3d"
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
        
        #out = nn.Sigmoid()(c8);del c8
        out = nn.Softmax(dim=1)(c8);del c8
        return out

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
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


class Unet2d(nn.Module):
    def __init__(self,in_ch,out_ch):
        super(Unet2d, self).__init__()
        self.name = "Unet2d"
        self.conv1 = DoubleConv(in_ch, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.conv4 = DoubleConv(256, 512)
        self.pool4 = nn.MaxPool2d(2)
        self.conv5 = DoubleConv(512, 1024)
        self.up6 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.conv6 = DoubleConv(1024, 512)
        self.up7 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv7 = DoubleConv(512, 256)
        self.up8 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv8 = DoubleConv(256, 128)
        self.up9 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv9 = DoubleConv(128, 64)
        self.conv10 = nn.Conv2d(64,out_ch, 1)

    def forward(self,x):
        c1=self.conv1(x)
        p1=self.pool1(c1)
        c2=self.conv2(p1)
        p2=self.pool2(c2)
        c3=self.conv3(p2)
        p3=self.pool3(c3)
        c4=self.conv4(p3)
        p4=self.pool4(c4)
        c5=self.conv5(p4)
        up_6= self.up6(c5)
        merge6 = torch.cat([up_6, c4], dim=1)
        c6=self.conv6(merge6)
        up_7=self.up7(c6)
        merge7 = torch.cat([up_7, c3], dim=1)
        c7=self.conv7(merge7)
        up_8=self.up8(c7)
        merge8 = torch.cat([up_8, c2], dim=1)
        c8=self.conv8(merge8)
        up_9=self.up9(c8)
        merge9=torch.cat([up_9,c1],dim=1)
        c9=self.conv9(merge9)
        c10=self.conv10(c9)
        #out = nn.Sigmoid()(c10)
        out = nn.Softmax(dim=1)(c10)
        return out


if __name__=='__main__':
    model = Unet3d(1,8)
    
    import nibabel as nib

    sample_filename = 'mr_train_1003_image.nii.gz'
    
    # read nii.gz file
    img = nib.load(sample_filename)
    
    # check shape
    print(img.shape)
    # (512, 512, 160)
    
    # transform the matrix into numpy array
    data = img.get_fdata()
    
    data = data.astype(np.float32)
    
    modelsize(model,torch.from_numpy(data[np.newaxis,np.newaxis,:,:,:]))



