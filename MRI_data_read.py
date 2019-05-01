# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 21:02:50 2019

@author: Dynasting
"""
from tqdm import tqdm
import nibabel as nib
import numpy as np
import util
import torch
from nilearn import plotting
import os
from scipy import ndimage
import warnings
warnings.filterwarnings('ignore')








imgs = util.make_dataset('./train')



for imgfn,mskfn in imgs:
    #sample_filename = './test/mr_train_1003_image.nii.gz'
    
    # read nii.gz file
    img = nib.load(imgfn)
    msk = nib.load(mskfn)
    affine = img.affine
    
    d,h,w = (img.header.get_zooms())
    
    image = img.get_fdata()
    label = msk.get_fdata()
    D,H,W = img.shape
    

    
    image_new = torch.nn.functional.upsample(torch.from_numpy(image[np.newaxis,np.newaxis]), size = (int(D*d),int(H*h),int(W*w)))
    label_new = torch.nn.functional.upsample(torch.from_numpy(label[np.newaxis,np.newaxis]), size = (int(D*d),int(H*h),int(W*w)))
    
    #rmin, rmax, cmin, cmax, zmin, zmax = util.bbox_3D(label_new[0,0])
    #Sd,Sh,Sw = rmax - rmin, cmax - cmin, zmax - zmin
    
    #loc = ndimage.find_objects(label_new)[0]
    cs,ch,cw = ndimage.measurements.center_of_mass(label_new[0,0].numpy())
    print(cs,ch,cw,'in ',image_new.shape[2:])
#    img_wb = nib.Nifti1Image(image_new.squeeze().numpy(), affine)
#    img_wb.header.set_zooms((1,1,1))
#    nib.save(img_wb, os.path.join('./train_size_norm', os.path.split(imgfn)[1]))
#    img_wb = nib.Nifti1Image(lagel_new.squeeze().numpy(), affine)
#    img_wb.header.set_zooms((1,1,1))
#    nib.save(img_wb, os.path.join('./train_size_norm', os.path.split(mskfn)[1]))
#    
    
# check shape
#print(img.shape)
# (512, 512, 160)

# transform the matrix into numpy array
#data = img.get_fdata()

# visualize
#display = plotting.plot_img(sample_filename)
# display.close()

#write back
#img_wb = nib.Nifti1Image(data, np.eye(4))
#nib.save(img, 'wb.nii.gz')