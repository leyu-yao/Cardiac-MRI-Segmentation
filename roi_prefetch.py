import numpy as np  
import torch 
import os
import sys
import matplotlib.pyplot as plt
import nibabel as nib 




def roi_detect_np(y):
    '''
    
    y D,H,W np.array

    return (slice(), slice(), slice())
    '''


    r = np.any(y, axis=(1, 2))
    c = np.any(y, axis=(0, 2))
    z = np.any(y, axis=(0, 1))

    rmin, rmax = np.where(r)[0][[0, -1]]
    cmin, cmax = np.where(c)[0][[0, -1]]
    zmin, zmax = np.where(z)[0][[0, -1]]

    return (slice(rmin, rmax), slice(cmin, cmax), slice(zmin, zmax))

def roi_train_prerare(dst_dir, src_dir, num_classes=8):
    '''
    read nii files as np
    slice roi
    get one-hot 
    output numpy arr in file system
    '''
    pass