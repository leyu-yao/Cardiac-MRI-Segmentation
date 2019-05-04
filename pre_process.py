# -*- coding: utf-8 -*-
"""
Created on Sat May  4 20:14:32 2019

@author: Dynasting
"""
import cv2
import nibabel as nib
import numpy as np 

path = '5_image.nii.gz'
outpath = '5_image_norm.nii.gz'
nb = nib.load(path)

img = nb.get_fdata()
affine = nb.affine

d,h,w = img.shape
img_2d = img.reshape(d,h*w).astype(np.uint8)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
equ = clahe.apply(img_2d)
img_out = equ.reshape(d,h,w).astype(np.float32)

#img_out = (img-img.min())/(img.max()-img.min())

img_wb = nib.Nifti1Image(img_out,affine)
nib.save(img_wb, outpath)