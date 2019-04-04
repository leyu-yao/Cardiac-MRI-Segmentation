# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 20:27:13 2019

@author: Dynasting

This file contains some functions for testing pre-trained model.
"""


import torch
import numpy as np
import unet3d
import util
import metrics
from unet3d import Unet2d


class Stacker():
    def __init__(self, img_fn, mask_fn):
        self.img_fn = img_fn
        self.mask_fn = mask_fn
        # oh = one-hot
        (self.img_np, self.mask_np_oh, self.affine) = util.read_nii_as_np(img_fn, mask_fn)
        self.slice_num = 0
        self.MAX_SLICE_NUM = img_np.shape[2]
    
    # return value :  (img_Tensor,target_Tensor)
    # (N C W H)
    def unstack(self):
        img_slice_np = self.img_np[:,:,self.slice_num]#x,y
        img_tensor = torch.from_numpy(img_slice_np[np.newaxis,np.newaxis,:,:])#n,c,x,y
        
        msk_slice_np = self.mask_np_oh[:,:,:,self.slice_num]#c,x,y
        target_tensor = torch.from_numpy(msk_slice_np[np.newaxis,:,:])#n,c,x,y
        
        self.slice_num += 1
        
        return img_tensor, target_tensor
'''
This function is used to test a network. 
The network takes in 2d slices and then stack the slices to obtain 3d data.
'''

def test_2d_slice(root, ckp, metric, device='cuda'):
    img_mask_list = util.make_dataset(root, label_elem='_label.nii.gz')
    
    model = Unet2d(1, 5).to(device)
    model.load_state_dict(torch.load(ckp,map_location='cuda'))
    model.eval()
    
    with torch.no_grad():
        for img,mask in img_mask_list:
            stacker = Stacker(img, mask)
            for _ in range(stacker.MAX_SLICE_NUM):
                x, y = stacker.unstack()
                outputs=model(x.to(device))
                labels = y.to(device)
                
                # Need to save all outputs to stack over-all
                
                # This part should be replaced by some metrics
                criterion = CrossEntropyDiceLoss(num_of_classes=5,
                    weights_for_dice=torch.tensor([1, 1., 2., 2., 2.]).to(device))
                loss = criterion(outputs, labels)
                average_loss += loss.item()
                print("image loss = %0.3f" % loss.item())
            
        