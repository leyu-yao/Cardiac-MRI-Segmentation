# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 14:20:08 2019
@author: Dynasting
"""

import nibabel as nib
import numpy as np
import os
import sys
import argparse
import matplotlib.pyplot as plt
import cv2
import torch
from tqdm import tqdm 

import transforms
import util

#cv2.imwrite("filename.png", np.zeros((10,10)))  write image

def _make_dataset(root, content='_label.nii.gz'):
    imgs=[]
    files = os.listdir(root)
    for file in files:
        if content in file:
            mask = os.path.join(root, file)
            img = mask.replace('label','image')
            imgs.append((img,mask))
 
        else:
            continue
    
    return imgs


def generate(dst, src, block_size, stride, num_classes):
    '''
    TO DO
    get list of file
    read in img and mask
    mask one-hot
    img and mask cut into []
    '''
    img_lst = _make_dataset(src)
    
    idx = 0
    
    for fn_num, (img_fn,mask_fn) in enumerate(img_lst):
        
        img = nib.load(img_fn)
        mask = nib.load(mask_fn)
        
        print('processing %d/%d' % (fn_num + 1, len(img_lst)))
        
        # get size  (d,h,w)
        
        (D, H, W) = img.shape
        
        # transform to np 
        img = img.get_fdata()
        mask = mask.get_fdata()
        
        # one hot
        mask = util.one_hot(mask, num_of_class=num_classes)
        
        # add axis
        img = img[np.newaxis,:,:,:]
        

        
        # np.float32
        img = img.astype(np.float32)
        mask = mask.astype(np.float32)
        
        
        # cut
        # iter times
        d = 1 + int((D-block_size[0])/stride[0])
        h = 1 + int((H-block_size[1])/stride[1])
        w = 1 + int((W-block_size[2])/stride[2])
        
        with tqdm(total=(d+1)*(h+1)*(w+1)) as pbar:
            for dd in range(d+1):
                for hh in range(h+1):
                    for ww in range(w+1):
                        x1 = dd*stride[0]
                        x2 = dd*stride[0]+block_size[0]
                        y1 = hh*stride[1]
                        y2 = hh*stride[1]+block_size[1]
                        z1 = ww*stride[2]
                        z2 = ww*stride[2]+block_size[2]

                        if x2 > D:
                            x1 = D - block_size[0]
                            x2 = D
                        if y2 > H:
                            y1 = H - block_size[1]
                            y2 = H
                        if z2 > W:
                            z1 = W - block_size[2]
                            z2 = W


                        block_img = img[:, x1:x2, y1:y2, z1:z2]
                        block_mask = mask[:, x1:x2, y1:y2, z1:z2]
                        
                        tran=transforms.Normalization()
                        block_img,_ = tran(block_img,None)
                        if (block_mask[0,:,:,:]==1).all():
                            pbar.update()
                            continue
                        
                        # save np file
                        img_name = os.path.join(dst, '%d_image.npy'%idx)
                        mask_name = os.path.join(dst, '%d_label.npy'%idx)
                        np.save(img_name, block_img)
                        np.save(mask_name, block_mask)
                        
                        img_wb = nib.Nifti1Image(block_img[0],np.eye(4))
                        nib.save(img_wb, '%d_image.nii.gz'%idx)
                        
                        img_wb = nib.Nifti1Image(np.argmax(block_mask,axis=0).astype(np.int32),np.eye(4))
                        nib.save(img_wb, '%d_label.nii.gz'%idx)
                        
                        pbar.update()

                        
                        idx += 1

        


def show_size(src):
    img_lst = _make_dataset(src)
    
    idx = 0
    
    for img_fn,mask_fn in img_lst:
        
        img = nib.load(img_fn)
        mask = nib.load(mask_fn)
        
        sys.stdout.flush()
        print('processing %s' % img_fn)
        print(img.shape)

        

                    

        
        
if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument("action", type=str)
    parse.add_argument("dst_dir", type=str)
    parse.add_argument("src_dir", type=str)
    parse.add_argument("--block_size", nargs='+', type=int, default=(96,96,96))
    parse.add_argument("--stride", nargs='+', type=int, default=(48,48,48))
    parse.add_argument("--num_classes", type=int, default=8)
    #parse.add_argument("--device", type=str, default='cuda')

    
    args = parse.parse_args()
    if args.action == 'generate':
        generate(args.dst_dir, args.src_dir, tuple(args.block_size), tuple(args.stride), args.num_classes)
