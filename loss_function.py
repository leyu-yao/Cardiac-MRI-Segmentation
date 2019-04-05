
import torch
from torch.autograd import Function
from itertools import repeat
import numpy as np
import sys
from torch import nn

# Intersection = dot(A, B)
# Union = dot(A, A) + dot(B, B)
# The Dice loss function is defined as
# 1/2 * intersection / union
#
# The derivative is 2[(union * target - 2 * intersect * input) / union^2]





class CrossEntropy3d(nn.Module):
    def __init__(self, *args, **kwargs):
        super(CrossEntropy3d, self).__init__()
        
    def forward(self, prob, target):
        prob = prob.view(-1).clamp(1e-3,1)
        target = target.view(-1)
        log_prob = torch.log(prob)/target.sum()
        
        sigma = -torch.dot(target, log_prob)
        

        #print("loss = %4f"%(sigma),flush=True)
        
        return  sigma
    
'''  
class DiceLoss(nn.Module):
    def __init__(self, *args, **kwargs):
         super(DiceLoss, self).__init__()

    def forward(self, input, target, save=True):
        eps = 0.000001

        input = input.view(-1)
        target = target.view(-1)
        intersect = torch.dot(input, target)
        # binary values so sum the same as sum of squares
        result_sum = torch.sum(input)
        target_sum = torch.sum(target)
        union = result_sum + target_sum + (2*eps)

        # the target volume can be empty - so we still want to
        # end up with a score of 1 if the result is 0/0
        IoU = intersect / union
        
        #sys.stdout.flush()
        
        #print('union: {:.3f}\t intersect: {:.6f}\t target_sum: {:.0f} IoU: result_sum: {:.0f} IoU {:.7f}'.format(
            union, intersect, target_sum, result_sum, 2*IoU))
        out = torch.FloatTensor(1).fill_(2*IoU)
        self.intersect, self.union = intersect, union
        return 1 - out
'''


'''
a auxiliary function for dice loss
input (N,C,D,W,H)   [0.7,   0.2,  0.1]
ouput (N,C,D,W,H)   [1.0,   0.0,  0.0]

x = torch.rand(1,3,4,5)
y=one_hot(x)


'''
def one_hot(x):
    label = torch.argmax(x, dim=1, keepdim=True)
    device = x.device
    y = torch.FloatTensor(*x.shape).to(device)
    '''
    for _ in range(C):
        y[:,_].zero_()
        y[:,_].scatter(1,label,1)
    '''
    y.zero_()
    y.scatter_(1, label, 1)
    
    return y


'''
This function calculates dice of two planes in one-hot mode.
pred, target  (N, *)
type = torch.Tensor

In the function, the input like [0.9,0.05,0.05] should be 
    transformed in to [1,0,0] externally
    
    Currently no one hot 

'''
def dice_loss(pred, target):

    smooth = 1
    
    N = pred.shape[0]
    pred_flat = pred.view(N,-1)
    target_flat = target.view(N,-1)
    intersection = pred_flat * target_flat
    
    num = 2 * intersection.sum() + smooth
    
    den1 = (pred_flat * pred_flat).sum()
    den2 = (target_flat * target_flat).sum()
    
    den = den1 + den2 + smooth
    
    dice = num / den
    
    return 1 - dice
    
    
'''    

weights torch.tensor.to('cuda')
This DiceLoss takes input as
input   (N,C,*)
target  (N,C,*)
For binary-class, only use (N,1,*)
For multi-class, sum (N,1:C,*)

This loss can take either 2-classes or multi-classes.


    

参考资料
https://blog.csdn.net/a362682954/article/details/81226427
'''

class DiceLoss(nn.Module):

    def __init__(self, num_of_classes, weights=None):
        super(DiceLoss, self).__init__()
        self.C = num_of_classes
        if weights is None:
            self.weights = torch.ones(self.C) / self.C
        else:
            self.weights = weights / weights.sum()
        
 
    def forward(self, input, target):
        
        #pred_one_hot = one_hot(input)
        
 
        C = target.shape[1]
        
        #if C == 2:
        #    return dice_loss(input[:,1], target[:,1])
        

      
        totalLoss = 0
         
        for i in range(C):
            diceLoss = dice_loss(input[:,i], target[:,i]) * self.weights[i]
            totalLoss += diceLoss
         
        return totalLoss
    


'''
A combination of cross entropy and dice loss.
L = w1 * cross_entropy + w2 * dice_loss 
'''
class CrossEntropyDiceLoss(nn.Module):
    
    def __init__(self, num_of_classes, weights_for_class=None, weights=None):
        super(CrossEntropyDiceLoss, self).__init__()
        self.C = num_of_classes
        if weights is None:
            self.weights = (torch.ones(2) / 2).to('cuda')
        else:
            self.weights = weights / weights.sum()
            
        if weights_for_class is None:
            self.weights_for_class = (torch.ones(self.C) / self.C).to('cuda')
        else:
            self.weights_for_class = weights_for_class / weights_for_class.sum()
            
    def forward(self, input, target):

        # calculate dice loss
        self.dice_loss = 0
         
        for i in range(self.C):
            self.dice_loss += dice_loss(input[:,i], target[:,i]) * float(self.weights_for_class[i])
         
        # calculate cross entropy
#        out = input.view(-1).clamp(1e-3,1)
#        target = target.view(-1)
#        log_prob = torch.log(out)/target.sum()
#        self.cross_entropy = -torch.dot(target, log_prob)
        
        self.cross_entropy = torch.nn.functional.cross_entropy(input, target.argmax(dim=1), 
                            weight=self.weights_for_class)
        

    
        return self.dice_loss * float(self.weights[0]) + self.cross_entropy*float(self.weights[1])
    
        