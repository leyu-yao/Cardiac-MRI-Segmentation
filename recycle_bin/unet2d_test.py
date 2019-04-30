'''
This is a test program for unet 2d slice solution. 
This module can be integrated to main_rev.py  
now it is in experiment. 
'''

import torch 
import numpy as np 
import cv2 
import matplotlib.pyplot as plt 
import os
import sys
import nibabel as nib 
import argparse


import util
import models
import loss_function


class test_on_2dUnet(object):
    def __init__(self, workspace, ckp, device, resolution, num_classes):
        self.workspace = workspace
        self.ckp = ckp
        self.device = device
        self.resolution = resolution
        self.num_classes = num_classes

        self.imgs = util.make_dataset(workspace)
        # load model
        self.model = models.Unet2d(1,self.num_classes).to(device)
        self.model.load_state_dict(torch.load(ckp, map_location=device))
        self.model.eval()
        self.criterion = loss_function.DiceLoss(num_classes)
        self.mp = torch.nn.AdaptiveMaxPool2d(resolution)

    def __call__(self):
        with torch.no_grad():
            for x_path, y_path in self.imgs:
                img = nib.load(x_path)
                mask = nib.load(y_path)
                
                affine = img.affine
                #sys.stdout.flush()
                #print('processing %s' % x_path)

                # get size  (d,h,w)
            
                (D, H, W) = img.shape
                
                up_sample = torch.nn.AdaptiveMaxPool2d((D, H))

                # transform to np  d,h,w
                img = img.get_fdata().astype(np.float32)
                mask = mask.get_fdata().astype(np.float32)



                # x_tensor
                x_tensor = torch.from_numpy(img[np.newaxis,np.newaxis,:,:,:]).to(self.device)

                # y_tensor
                y_tensor = torch.from_numpy(util.one_hot(mask, 8)[np.newaxis,:,:,:,:]).to(self.device)

                # out tensor
                out_tensor = torch.zeros(1,8,D,H,W)#.to(self.device)

                for w in range(W):
                    x = self.mp(x_tensor[:,:,:,:,w])
                    out_tensor[:,:,:,:,w] = up_sample(self.model(x))

                #out_tensor = out_tensor.argmax(dim=1, keepdim=True)

                #score = 1 - self.criterion(out_tensor, y_tensor)
                #print("score of %s is %0.3f" % (x_path, score))

                result_np = out_tensor[0].argmax(dim=0).cpu().numpy()

                res = nib.Nifti1Image(result_np.astype(np.float32), affine)

                nib.save(res, x_path.replace('image', 'output'))
                
                del x_tensor, y_tensor, out_tensor, result_np



workspace = './test'
ckp = './results/4.24test2d/weights_30_dice_2d.pth'
device = torch.device('cuda')
resolution = (256, 256)
num_classes = 8





test = test_on_2dUnet(workspace, ckp, device, resolution, num_classes)
test()