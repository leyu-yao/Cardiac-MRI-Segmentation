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
    X, Y  should be tensor (C, *, *, *)

usage : tran = transform(para)
        X, Y = tran(X, Y)
"""
import torch
import cv2
import numpy as np 
from torchvision import transforms
from random import choice

# ADD Transforms Tomorrow
class CLAHE(object):
    '''
    This operation only afffects X, does not affect Y. 
    C,X,Y
    '''
    def __call__(self, X):
        c,d,h = X.shape
        X_norm =  (255*(X - X.min()) / (X.max() - X.min())).astype(np.uint8)
        
        img_2d = X_norm.reshape(d,h).astype(np.uint8)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        equ = clahe.apply(img_2d)
        img_out = equ.reshape(c,d,h).astype(np.float32)
        
        return img_out

def clahe_for_tensor(X):
    '''
    1 1 d h
    '''
    _, c, d, h = X.shape
    X = X.cpu().numpy()
    X_norm = (255*(X - X.min()) / (X.max() - X.min())).astype(np.uint8)
    img = X_norm[0,0,:,:]
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    equ = clahe.apply(img)
    X_tensor = torch.from_numpy(equ[np.newaxis, np.newaxis, :, :].astype(np.float32)).cuda()
    return X_tensor


class data_augumentation_2d(object):

    def __init__(self,size):
        self.size = size
        self.mp = torch.nn.AdaptiveMaxPool2d((256,256))
    def __call__(self, X, Y):
        return self.mp(X.unsqueeze(0)).squeeze(0), self.mp(Y.unsqueeze(0)).squeeze(0)
class DummyTransform(object):


    def __call__(self, X, Y):
        return X, Y

class Transpose(object):

    def __init__(self, dim1=1, dim2=2):
        self.dim1=dim1
        self.dim2=dim2
    def __call__(self, X, Y):
        return X.transpose(self.dim1, self.dim2), Y.transpose(self.dim1, self.dim2)

class DummyTransform(object):


    def __call__(self, X, Y):
        return X, Y

class Normalization(object):
    '''
    normalize sample x, does not affect y 
    '''
    def __call__(self, x, y):
        return (x - x.mean()) / x.std(), y

        
class ComposedTransformer(object):
    '''
    ComposedTransformer
    usage :
        com = ComposedTransformer(t1, t2)
        # t1 t2 must have __call__(self,X,Y)
        X,Y=com(X,Y)
    '''
    def __init__(self, *args):
        self.trans = [x for x in args]
    
    def __len__(self):
        return len(self.trans)

    def __getitem__(self, index):
        return self.trans[index]

    def __call__(self, X, Y):
        for tran in self.trans:
            X, Y = tran(X, Y)

        return X, Y

class RandomTransformer(object):
    '''
    RandomTransformer
    usage :
        com = ComposedTransformer(t1, t2)
        # t1 t2 must have __call__(self,X,Y)
        X,Y=com(X,Y)
    '''
    def __init__(self, *args):
        self.trans = [x for x in args]
    
    def __len__(self):
        return len(self.trans)

    def __getitem__(self, index):
        return self.trans[index]

    def __call__(self, X, Y):
        tran = choice(self.trans)
        X, Y = tran(X, Y)

        return X, Y


if __name__ == "__main__":
    X = torch.rand(1,3,4,5)
    Y = torch.rand(5,3,4,5)
    
    tran = Transpose()
    X, Y = tran(X, Y)