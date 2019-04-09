# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 00:13:32 2019

@author: Dynasting

This module implements serveral useful transformation function for 3d data, 
which can be used for data augumentation. 



Transforms are called in dataset.py Dataset.__getitem__(index)
e.g.

X                           Y
torch.Size([1, 32, 32, 32]) torch.Size([5, 32, 32, 32])

transform :
tensor(C,X,Y,Z) => tensor (C,X,Y,Z)

All transform should have prototype:

    X, Y = self.__call__(X, Y)

usage : tran = transform(para)
        X, Y = tran(X, Y)
"""

from torchvision import transforms
import random

# ADD Transforms Tomorrow

class Transpose(object):

    def __init__(self, dim1=1, dim2=2):
        self.dim1=dim1
        self.dim2=dim2
    def __call__(self, X, Y):
        pass