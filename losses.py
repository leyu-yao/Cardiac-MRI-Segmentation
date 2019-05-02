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
    