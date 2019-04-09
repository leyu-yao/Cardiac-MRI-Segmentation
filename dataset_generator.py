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

    return out

'''
TO DO
get list of file
read in img and mask

mask one-hot
img and mask cut into []
'''
def generate(dst, src, block_size, stride):
    img_lst = _make_dataset(src)
    
    idx = 0
    
    for img_fn,mask_fn in img_lst:
        
        img = nib.load(img_fn)
        mask = nib.load(mask_fn)
        
        sys.stdout.flush()
        print('processing %s' % img_fn)
        
        # get size  (d,h,w)
        
        (D, H, W) = img.shape
        
        # transform to np 
        img = img.get_fdata()
        mask = mask.get_fdata()
        
        # one hot
        mask = util.one_hot(mask, num_of_class=5)
        
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
        
        for dd in range(d):
            for hh in range(h):
                for ww in range(w):
                    block_img = img[:, dd*stride[0]:dd*stride[0]+block_size[0],
                                    hh*stride[1]:hh*stride[1]+block_size[1],
                                    ww*stride[2]:ww*stride[2]+block_size[2]]
                    block_mask = mask[:, dd*stride[0]:dd*stride[0]+block_size[0],
                                    hh*stride[1]:hh*stride[1]+block_size[1],
                                    ww*stride[2]:ww*stride[2]+block_size[2]]
                    
                    if (block_mask[0,:,:,:]==1).all():
                        continue
                    
                    # save np file
                    img_name = os.path.join(dst, '%d_image.npy'%idx)
                    mask_name = os.path.join(dst, '%d_label.npy'%idx)
                    np.save(img_name, block_img)
                    np.save(mask_name, block_mask)
                    
#                    # save to nii
#                    img_name = os.path.join(dst, '%d_image.nii'%idx)
#                    mask_name = os.path.join(dst, '%d_label.nii'%idx)
#                    #np.save(img_name, block_img)
#                    
#                    img_nii = nib.Nifti1Image(block_img, np.eye(4))
#                    
#                    nib.save(img_nii, img_name)
                    
                    idx += 1
        
def show_from_cv(img, fignum,title=None):
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.figure(fignum)
    plt.imshow(img)
    if title is not None:
        plt.title(title)
    #plt.pause(1)
    
def show_label_from_cv(img, fignum,title=None):
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.figure(fignum)
    for i in range(1,9):
        plt.subplot(2,4,i)
        plt.imshow(img[i-1])
    if title is not None:
        plt.title(title)
    plt.pause(1)

def generate_2d_test_data(dst, src, block_size, stride):
    img_lst = _make_dataset(src)
    
    idx = 0
    
    for img_fn,mask_fn in img_lst:
        
        img = nib.load(img_fn)
        mask = nib.load(mask_fn)
        
        sys.stdout.flush()
        print('processing %s' % img_fn)
        
        # get size  (d,h,w)
        
        (D, H, W) = img.shape
        
        # transform to np 
        img = img.get_fdata()
        mask = mask.get_fdata()
        
        # one hot
        mask = util.one_hot(mask)
        
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
        
        for dd in range(d):
            for hh in range(h):
                for ww in range(w):
                    block_img = img[:, dd*stride[0]:dd*stride[0]+block_size[0],
                                    hh*stride[1]:hh*stride[1]+block_size[1],
                                    ww*stride[2]:ww*stride[2]+block_size[2]]
                    block_mask = mask[:, dd*stride[0]:dd*stride[0]+block_size[0],
                                    hh*stride[1]:hh*stride[1]+block_size[1],
                                    ww*stride[2]:ww*stride[2]+block_size[2]]
                    
                    if (block_mask[0,:,:,:]==1).all():
                        continue
                    
                    if block_mask[1].sum() < block_mask.sum() * 0.02:
                        continue
                    
                    # save np file
                    img_name = os.path.join(dst, '%d_image.npy'%idx)
                    mask_name = os.path.join(dst, '%d_label.npy'%idx)
                    np.save(img_name, block_img)
                    np.save(mask_name, block_mask)
                    
                    
                    
                    
                    # plot the img 
                    for layer in range(0,block_img.shape[3],4):
                        #show_from_cv(block_img[0,:,:,layer],1)
                        plt.figure(1)
                        plt.ion() 
                        
                        plt.subplot(1,2,1)
                        plt.imshow(150*block_mask[1,:,:,layer])

                        plt.subplot(1,2,2)
                        plt.imshow(block_img[0,:,:,layer])

                        plt.pause(0.001)
                        plt.show()
                    
#                    # save to nii
#                    img_name = os.path.join(dst, '%d_image.nii'%idx)
#                    mask_name = os.path.join(dst, '%d_label.nii'%idx)
#                    #np.save(img_name, block_img)
#                    
#                    img_nii = nib.Nifti1Image(block_img, np.eye(4))
#                    
#                    nib.save(img_nii, img_name)
                    
                    idx += 1
        
        
        
def generate_no_cut(dst, src, block_size):
    img_lst = _make_dataset(src)
    
    idx = 0
    
    for img_fn,mask_fn in img_lst:
        
        img = nib.load(img_fn)
        mask = nib.load(mask_fn)
        
        sys.stdout.flush()
        print('processing %s' % img_fn)
        
        # get size  (d,h,w)
        
        (D, H, W) = img.shape
        # 512 512 160/140
        
        # transform to np 
        img = img.get_fdata()
        mask = mask.get_fdata()
        
        # one hot
        mask = util.one_hot(mask)
        
        # add axis
        img = img[np.newaxis,:,:,:]
        

        
        # np.float32
        img = img.astype(np.float32)
        mask = mask.astype(np.float32)
        
        # range
        d1 = int(D/2) - int(block_size[0]/2)
        d2 = d1 + block_size[0]
        
        h1 = int(H/2) - int(block_size[1]/2)
        h2 = h1 + block_size[1]
        
        w1 = int(W/2) - int(block_size[2]/2)
        w2 = w1 + block_size[2]
                    
        # save np file
        img_name = os.path.join(dst, '%d_image.npy'%idx)
        mask_name = os.path.join(dst, '%d_label.npy'%idx)
        np.save(img_name, img[:,d1:d2,h1:h2,w1:w2])
        np.save(mask_name, mask[:,d1:d2,h1:h2,w1:w2])
        
#                    # save to nii
#                    img_name = os.path.join(dst, '%d_image.nii'%idx)
#                    mask_name = os.path.join(dst, '%d_label.nii'%idx)
#                    #np.save(img_name, block_img)
#                    
#                    img_nii = nib.Nifti1Image(block_img, np.eye(4))
#                    
#                    nib.save(img_nii, img_name)
        
        idx += 1
        

def generate_2d_slices_numpy(dst, src, visualize=False):
    img_lst = _make_dataset(src)
    
    idx = 0
    
    for img_fn,mask_fn in img_lst:
        
        img = nib.load(img_fn)
        mask = nib.load(mask_fn)
        
        sys.stdout.flush()
        print('processing %s' % img_fn)
        
        # get size  (d,h,w)
        
        (D, H, W) = img.shape
        # 512 512 160/140
        
        # transform to np 
        img = img.get_fdata()
        mask = mask.get_fdata()
        
        # one hot
        mask = util.one_hot(mask, num_of_class=5)

        # add axis
        img = img[np.newaxis,:,:,:]

        # np.float32
        img = img.astype(np.float32)
        mask = mask.astype(np.float32)



        for _ in range(W):
            
            # 数据清洗
#            cnt = mask[1:,:,:,_].sum()
#            if cnt == 0:
#                print("ignoring slice%03d" % _)
#                continue
            
            print("taking slice%03d" % _)
            
            img_name = os.path.join(dst, '%d_image.npy'%idx)
            mask_name = os.path.join(dst, '%d_label.npy'%idx)
            np.save(img_name, img[:,:,:,_])
            np.save(mask_name, mask[:,:,:,_])
            
            


            if visualize:
                plt.subplot(1,2,1)
                plt.title("image")
                plt.imshow(img[0,:,:,_])
                
                plt.subplot(1,2,2)
                plt.title("mask")
                plt.imshow(150*mask[1:,:,:,_].sum(0))
                plt.pause(0.0001)

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
    parse.add_argument("--block_size", nargs='+', type=tuple, default=(32,32,32))
    parse.add_argument("--stride", type=tuple, default=(100,100,50))

    
    args = parse.parse_args()
    if args.action == 'g':
        generate(args.dst_dir, args.src_dir, tuple(args.block_size), args.stride)
    elif args.action == 't':
        generate_2d_test_data(args.dst_dir, args.src_dir, tuple(args.block_size), args.stride)
    elif args.action == 'g_no_cut':
        generate_no_cut(args.dst_dir, args.src_dir, tuple(args.block_size))
    elif args.action == 'g_slice':
        generate_2d_slices_numpy(args.dst_dir, args.src_dir, visualize=False)
    elif args.action == 'show_size':
        show_size(args.src_dir)


