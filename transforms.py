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



class Transpose(object):

    def __init__(self, dim1=1, dim2=2):
        self.dim1=dim1
        self.dim2=dim2
    def __call__(self, X, Y):
        return X.transpose(self.dim1, self.dim2), Y.transpose(self.dim1, self.dim2)

class DummyTransform(object):


    def __call__(self, X, Y):
        return X, Y


class DownSample(object):
    '''
    down sample x and y at the same time
    '''
    def __init__(self, object_size):
        '''
        object_size (D, H, W) tuple
        '''
        self.object_size = object_size
        self.maxpool = torch.nn.AdaptiveMaxPool3d(self.object_size)

    def __call__(self, x, y):
        #input C,D,H,W
        return self.maxpool(x.unsqueeze(0)).squeeze(0), self.maxpool(y.unsqueeze(0)).squeeze(0)


class SmartDownSample(object):
    '''
    down sample x and y at the same time
    this is called smart because it keeps the d,h,w ratio. 
    '''
    def __init__(self, object_size):
        '''
        object_size (D, H, W) tuple
        '''
        self.object_size = object_size
        self.memory_size = object_size[0] * object_size[1] *object_size[2]



    def __call__(self, x, y):
        #input C,D,H,W
        C,D,H,W = x.shape
        input_size = D * H * W
        ratio = input_size / self.memory_size

        def base_floor(x, base=8):
            mo = x % base
            return (x - mo)

        D_desired = base_floor(D//ratio)
        H_desired = base_floor(H//ratio)
        W_desired = base_floor(W//ratio)

        maxpool = torch.nn.AdaptiveMaxPool3d((D_desired, H_desired, W_desired))
        return maxpool(x.unsqueeze(0)).squeeze(0), maxpool(y.unsqueeze(0)).squeeze(0)





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
    X = torch.rand(1,100,100,20)
    Y = torch.rand(5,100,100,20)
    
    tran = DownSample((50,50,25))
    X, Y = tran(X, Y)