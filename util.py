# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 20:05:22 2019
@author: Dynasting
"""
import numpy as np
import shutil
import os
import sys
import random
import argparse
import torch
import nibabel as nib

def make_dataset(root, label_elem='_label.nii.gz'):
    imgs=[]
    files = os.listdir(root)
    for file in files:
        if label_elem in file:
            mask = os.path.join(root, file)
            img = mask.replace('label','image')
            imgs.append((img,mask))
 
        else:
            continue
    
    return imgs

def one_hot(np_label, num_classes = 5):
    shape = np_label.shape
    out = np.zeros((num_classes, *shape), dtype=np.float32)
    
    numl = [  0, 205, 420, 500, 550, 600, 820, 850]
    # Need to handle num!=8 other class -> class0
    for i, num in enumerate(numl[0:num_classes]):
        if i==0:
            continue
        out[i,:,:,:][np_label==num]=1

    out[0,:,:,:] = 1 - np.sum(out[1:,:,:,:], axis=0)
    
    return out


def write_nii(inp, filename=None, affine=np.eye(4)):
    '''
    possible input
    tensor 1,1,W,H,D
    np arr 1, W, H, D
    '''
    if filename is None:
        id = hash(inp)
        fn = "%s.nii.gz" % str(id)
    else:
        fn = filename

    # tensor
    if type(inp) == torch.Tensor:
        if inp.shape[1] == 1:
            out = inp[0,0,:,:,:].cpu().numpy()
        else:
            out = torch.argmax(inp[0],dim=0,keepdim=False).cpu().numpy()
        img = nib.Nifti1Image(out.astype(np.float32), affine)
        nib.save(img, fn)
    
    # array
    elif type(inp) == np.array:
        out = inp[0,:,:,:]
        img = nib.Nifti1Image(out, affine)
        nib.save(img, fn)


def fix_unicode_bug():
    '''
    fix print unicode bug
    '''
    import win_unicode_console
    win_unicode_console.enable()