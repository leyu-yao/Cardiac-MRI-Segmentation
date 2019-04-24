import numpy as np  
import torch 
import os
import sys
import matplotlib.pyplot as plt
import nibabel as nib 

import util


def roi_detect_np(y):
    '''
    
    y D,H,W np.array

    return (slice(), slice(), slice())

    return value usage  x[res]
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
    img_msk_list = util.make_dataset(src_dir)

    idx = 0


    for x_path, y_path in img_msk_list: 

        img_nib = nib.load(x_path)
        mask_nib = nib.load(y_path)

        # np.array D,H,W
        x = img_nib.get_fdata().astype(np.float32)
        y = mask_nib.get_fdata().astype(np.float32)

        roi_slice = roi_detect_np(y)

        