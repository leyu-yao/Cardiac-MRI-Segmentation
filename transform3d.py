# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 00:13:32 2019

@author: Dynasting

This module implements serveral useful transformation function for 3d data, 
which can be used for data augumentation. 

transforms can be divided into 2 main forms 
X_transforms and Y transforms 

Transforms are called in dataset.py Dataset.__getitem__(index)
e.g.

X                           Y
torch.Size([1, 32, 32, 32]) torch.Size([5, 32, 32, 32])

X_transforms :
tensor(C,X,Y,Z) => tensor (C,X,Y,Z)

Y_transforms :
tensor(C,X,Y,Z) => tensor (C,X,Y,Z)

Generally, a transform.__call__(self, inp) takes input as tensor(C,X,Y,Z). 
X_transform and Y_transform are generally the same. 
"""

from torchvision import transforms

# ADD Transforms Tomorrow