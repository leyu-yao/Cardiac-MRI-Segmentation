# -*- coding: utf-8 -*-
"""
Created on Sat May  4 20:14:32 2019

@author: Dynasting
"""
import cv2
import nibabel as nib
import numpy as np 
from  scipy import ndimage

path = './train/mr_train_1003_image.nii.gz'
outpath = '1003_image_rotate.nii.gz'
nb = nib.load(path)

img = nb.get_fdata()
affine = nb.affine

d,h,w = img.shape
img_out = ndimage.rotate(img, 10.0, reshape=False, mode='nearest')



#img_out = (img-img.min())/(img.max()-img.min())

img_wb = nib.Nifti1Image(img_out,affine)
nib.save(img_wb, outpath)