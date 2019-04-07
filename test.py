# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 20:27:13 2019

@author: Dynasting

This file contains some functions for testing pre-trained model.
"""


import torch
import numpy as np

import metrics
from models import Unet3d
from util import read_nii_as_np

'''
To do:
1. 考虑封装一个类，带模型带数据集
2. filename -> np
3. np -> tensor
4. tensor -> metric
5. tensor -> np
6. np -> nii
'''

class Np_Cutter(object):
    '''
    takes in numpy X
    return x-cut in numpy
    '''
    pass

class Np_Tensor_Converter(object):
    '''
    takes in numpy array X,Y,Z
    return tensor on device 1,1,X,XY,Z
    '''
    def __init__(self, device):
        # save device
        self.device = device
    
    def __call_(self, inp):
        pass


class Test(object):
    '''
    Class Test for test of a pre-trained model 
    on a certain dataset.
    '''

    def __init__(self, workspace, deivce, ckp):
        '''
        construt model and load weights
        make dataset with filename
        '''
        pass

    def __call_(self):
        '''
        run test on all data
        '''
        pass
        