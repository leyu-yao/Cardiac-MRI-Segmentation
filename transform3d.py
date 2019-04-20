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
from torchvision import transforms
from random import choice

# ADD Transforms Tomorrow



class Transpose(object):

    def __init__(self, dim1=1, dim2=2):
        self.dim1=dim1
        self.dim2=dim2
    def __call__(self, X, Y):
        return X.transpose(self.dim1, self.dim2), Y.transpose(self.dim1, self.dim2)

class DummyTransform(object):


    def __call__(self, X, Y):
        return X, Y


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