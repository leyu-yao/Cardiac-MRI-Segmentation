# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 11:41:36 2019
@author: Dynasting
"""


import torch
from torch.autograd import Function
from itertools import repeat
import numpy as np
import sys
from torch import nn

from timeTick import timeTicker


def dice_loss(pred, target):

    smooth = 1
    
    N = pred.shape[0]
    pred_flat = pred.view(N,-1)
    target_flat = target.view(N,-1)
    intersection = pred_flat * target_flat
    
    num = 2 * intersection.sum() + smooth
    
    #den1 = (pred_flat * pred_flat).sum()
    #den2 = (target_flat * target_flat).sum()

    den1 = (pred_flat).sum()
    den2 = (target_flat).sum()
    
    den = den1 + den2 + smooth
    
    dice = num / den
    
    return 1 - dice
    

class DiceLoss(nn.Module):

    def __init__(self):
        super(DiceLoss, self).__init__()

    #@timeTicker
    def forward(self, pred, target):
        dice = 0
        
        smooth = 1
        C = pred.shape[1]
        N = pred.shape[0]
        
        for _ in range(C):
            
            
            pred_flat = pred[:,_].view(N,-1)
            target_flat = target[:,_].view(N,-1)
            intersection = pred_flat * target_flat
            
            num = 2 * intersection.sum() + smooth
            
            #den1 = (pred_flat * pred_flat).sum()
            #den2 = (target_flat * target_flat).sum()
        
            den1 = (pred_flat).sum()
            den2 = (target_flat).sum()
            
            den = den1 + den2 + smooth
            
            dice += num / den
        
        return 1 - dice/C
    
    
class CrossEntropyDiceLoss(nn.Module):
    
    def __init__(self,  w_dice=0.5, w_cross_entropy=0.5):
        super(CrossEntropyDiceLoss, self).__init__()
        self.w_dice = w_dice
        self.w_cross_entropy = w_cross_entropy
            
    #@timeTicker
    def forward(self, input, target):

        # calculate dice loss
        dl = DiceLoss()(input, target)
         
       
        # calculate cross entropy
#        out = input.view(-1).clamp(1e-3,1)
#        target = target.view(-1)
#        log_prob = torch.log(out)/target.sum()
#        self.cross_entropy = -torch.dot(target, log_prob)
        
        ce = torch.nn.functional.cross_entropy(input, target.argmax(dim=1))
        

    
        return dl * self.w_dice + ce * self.w_cross_entropy

class EdgeLoss(nn.Module):

    def __init__(self):
        super(EdgeLoss, self).__init__()
        self.dim = None
        
        self.xf_2d = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], 
                                  dtype=torch.float32).cuda().expand(1, 1, 3, 3)
        self.yf_2d = torch.tensor([[1, 2, 1],  [0, 0, 0],  [-1, -2, -1]], 
                                  dtype=torch.float32).cuda().expand(1, 1, 3, 3)
                
    
    def forward(self, input, target):

        if self.dim is None:
            self.dim = len(input.shape) - 2
            
        channels = input.shape[1]
        
        
        
        if self.dim == 2:
            
            sr = 0
            
            for c in range(channels):
                input_c = input[:,c,:,:]
                target_c = target[:,c,:,:]
                

            
                edge_inp_x = torch.nn.functional.conv2d(input_c.unsqueeze(1), self.xf_2d)
                edge_tar_x = torch.nn.functional.conv2d(target_c.unsqueeze(1), self.xf_2d)
                edge_inp_y = torch.nn.functional.conv2d(input_c.unsqueeze(1), self.yf_2d)
                edge_tar_y = torch.nn.functional.conv2d(target_c.unsqueeze(1), self.yf_2d)
            

                edge_inp = torch.sqrt(torch.pow(edge_inp_x, 2) + torch.pow(edge_inp_y, 2))
                edge_tar = torch.sqrt(torch.pow(edge_tar_x, 2) + torch.pow(edge_tar_y, 2))
                
                sr += torch.mean((edge_inp - edge_tar).pow(2))
                
            return sr/channels

class ComposedLoss(nn.Module):
    def __init__(self, losses, weights):
        super(ComposedLoss, self).__init__()
        self.losses = losses
        self.weights = weights
        
    def forward(self, input, target):
        val = 0
        for loss, w  in zip(self.losses, self.weights):
            val += w * loss(input, target)
        return val

# %% volume size weighted cross entropy
class wCross(nn.Module):
    def __init__(self):
        super(wCross, self).__init__()

    def forward(self, input, target):
        
        N,C,D,H,W = input.shape
        target_label = target.argmax(dim=1)
        weight = torch.zeros(C,).cuda()

        voxel = N * D * H * W
        for _ in range(C):
            n = (target_label == _).sum()
            weight[_] = 1 - n.float() / voxel

        loss = torch.nn.functional.nll_loss(torch.log(input), target_label, weight=weight)
        return loss

# %% volume size weighted focal loss
class wFocal(nn.Module):
    def __init__(self, gamma=1):
        super(wFocal, self).__init__()
        self.gamma = gamma

    def forward(self, input, target):
        
        N,C,D,H,W = input.shape
        target_label = target.argmax(dim=1)
        weight = torch.zeros(C,).cuda()

        voxel = N * D * H * W
        for _ in range(C):
            n = (target_label == _).sum()
            weight[_] = 1 - n.float() / voxel

        coef = torch.pow((1 - input), self.gamma)

        loss = torch.nn.functional.nll_loss(coef * torch.log(input) , target_label, weight=weight)
        return loss


# %% mDSC
class mDSC(nn.Module):
    def __init__(self):
        super(mDSC, self).__init__()

    def forward(self, output, target):
        N, C, D, H, W = output.shape
        #output_ = torch.nn.Softmax(dim=1)(output)
        voxels = D * H * W

        eps = 1e-5
        out = 0
        for n in range(N):
            for c in range(C):
                #up = 2 / voxels * (output[n, c] * target[n, c]).sum()
                up = 2 * (output[n, c] * target[n, c]).sum()
                down = (output[n, c] * output[n, c]).sum() + (target[n, c] * target[n, c]).sum() + eps
                out +=  up / down

        return 1 - out / N / C


# %% hybird Loss
class Hybird_Loss(nn.Module):
    def __init__(self, w4Cross=0.1, w4mDSC=2):
        super(Hybird_Loss, self).__init__()
        self.w4Cross = w4Cross
        self.w4mDSC = w4mDSC

    def forward(self, output, target):
        return self.w4Cross * wFocal()(output, target) + self.w4mDSC * mDSC()(output, target)
        #return self.w4Cross * wCross()(output, target) + self.w4mDSC * mDSC()(output, target)
        #return self.w4mDSC * mDSC()(output, target)
        #return self.w4Cross * wCross()(output, target)




# %% side loss
class Side_Loss(nn.Module):
    def __init__(self, rescale=1):
        '''
        rescale = 1 2 3 
        means 2x 4x 8x of the resolution
        '''
        super(Side_Loss, self).__init__()
        self.rescale = rescale
        self.mp = nn.MaxPool3d(2, stride=2)

    def forward(self, outputs, target):
        
        # rescale target to same size
        y = self.mp(target)
        for _ in range(1,self.rescale):
            y = self.mp(y)
    
        return Hybird_Loss()(outputs, y)
        
# %% Total Loss
class Total_Loss(nn.Module):
    def __init__(self, b1=0.2, b2=0.4, b3=0.8):
        super(Total_Loss, self).__init__()
        self.b1, self.b2, self.b3 = b1, b2, b3

    def forward(self, output, side1, side2, side3, target):
        return Hybird_Loss()(output, target) + self.b1 * Side_Loss(3)(side1, target) + self.b2 * Side_Loss(2)(side2, target) + self.b3 * Side_Loss(1)(side3, target) 


if __name__== "__main__":
    x = torch.rand(1,3,4,5).cuda()
    y = torch.rand(1,3,4,5).cuda()
    
    c = Total_Loss()
    l=c(x,y)