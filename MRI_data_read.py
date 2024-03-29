# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 21:02:50 2019

@author: Dynasting
"""

import nibabel as nib
import numpy as np
import cv2

from nilearn import plotting


sample_filename = 'mr_train_1002_image.nii.gz'

# read nii.gz file
img = nib.load(sample_filename)

# check shape
print(img.shape)
# (512, 512, 160)

img = img.get_fdata()[:,:,70]

img = 255 * (img-img.min()) / (img.max() - img.min())

#equ = cv2.equalizeHist(img.astype(np.uint8))
equ = clahe.apply(img.astype(np.uint8))
cv2.imwrite("raw.jpg",img)

cv2.imwrite("equ.jpg",equ)

# transform the matrix into numpy array
#data = img.get_fdata()

# visualize
#display = plotting.plot_img(sample_filename)
# display.close()

#write back
#img_wb = nib.Nifti1Image(data, np.eye(4))
#nib.save(img, 'wb.nii.gz')