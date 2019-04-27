import numpy as np 
import torch 
import nibabel as nib 
from tqdm import tqdm 
import sys
import os
import matplotlib.pyplot as plt 
import argparse
import time
import itertools

import util
import models



def broswe_data_shape(root):
    data = util.make_dataset(root)

    for x, y in data:
        img = nib.load(x)
        print(x, img.shape)


class Resizer(object):
    def __init__(self, shape):
        #self.w, self.h = shape
        self.MP = torch.nn.AdaptiveMaxPool2d(shape).to('cuda')

    def __call__(self, x, y):
        '''
        takes x, y numpy C,D,H
        '''
        with torch.no_grad(): 
            x_t = torch.from_numpy(x[np.newaxis, :, :, :]).to('cuda')
            y_t = torch.from_numpy(y[np.newaxis, :, :, :]).to('cuda')

            return self.MP(x_t)[0].cpu().numpy(), self.MP(y_t)[0].cpu().numpy()

def generate_z_slices(dst_dir, src_dir, shapes, visualize):
    img_lst = util.make_dataset(src_dir)

    resizers = []
    for shape in shapes:
        resizers.append(Resizer(shape))

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
        mask = util.one_hot(mask, num_of_class=8)

        # add axis
        img = img[np.newaxis,:,:,:]

        # np.float32
        img = img.astype(np.float32)
        mask = mask.astype(np.float32)



        for _ in tqdm(range(W)):
            if visualize:
                plt.subplot(1,2,1)
                plt.title("image")
                plt.imshow(img[0,:,:,_])
                
                plt.subplot(1,2,2)
                plt.title("mask")
                plt.imshow(150*mask[1:,:,:,_].sum(0))
                plt.pause(0.001)


            # Generate different size
            for resizer in resizers:
                img_name = os.path.join(dst_dir, '%d_image.npy'%idx)
                mask_name = os.path.join(dst_dir, '%d_label.npy'%idx)
                x_s, y_s = resizer(img[:,:,:,_], mask[:, :, :, _])
                np.save(img_name, x_s)
                np.save(mask_name, y_s)
                idx += 1

def generate_x_slices(dst_dir, src_dir, shapes, visualize):
    img_lst = util.make_dataset(src_dir)

    resizers = []
    for shape in shapes:
        resizers.append(Resizer(shape))

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
        mask = util.one_hot(mask, num_of_class=8)

        # add axis
        img = img[np.newaxis,:,:,:]

        # np.float32
        img = img.astype(np.float32)
        mask = mask.astype(np.float32)



        for _ in tqdm(range(D)):
            if visualize:
                plt.subplot(1,2,1)
                plt.title("image")
                plt.imshow(img[0,_,:,:])
                
                plt.subplot(1,2,2)
                plt.title("mask")
                plt.imshow(150*mask[1:,_,:,:].sum(0))
                plt.pause(0.001)


            # Generate different size
            for resizer in resizers:
                img_name = os.path.join(dst_dir, '%d_image.npy'%idx)
                mask_name = os.path.join(dst_dir, '%d_label.npy'%idx)
                x_s, y_s = resizer(img[:, _, :, :], mask[:, _, :, :])
                np.save(img_name, x_s)
                np.save(mask_name, y_s)
                idx += 1

def generate_y_slices(dst_dir, src_dir, shapes, visualize):
    img_lst = util.make_dataset(src_dir)

    resizers = []
    for shape in shapes:
        resizers.append(Resizer(shape))

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
        mask = util.one_hot(mask, num_of_class=8)

        # add axis
        img = img[np.newaxis,:,:,:]

        # np.float32
        img = img.astype(np.float32)
        mask = mask.astype(np.float32)



        for _ in tqdm(range(H)):
            if visualize:
                plt.subplot(1,2,1)
                plt.title("image")
                plt.imshow(img[0, :, _, :])
                
                plt.subplot(1,2,2)
                plt.title("mask")
                plt.imshow(150*mask[1:,:,_,:].sum(0))
                plt.pause(0.001)


            # Generate different size
            for resizer in resizers:
                img_name = os.path.join(dst_dir, '%d_image.npy'%idx)
                mask_name = os.path.join(dst_dir, '%d_label.npy'%idx)
                x_s, y_s = resizer(img[:, :, _, :], mask[:, :, _, :])
                np.save(img_name, x_s)
                np.save(mask_name, y_s)
                idx += 1


if __name__ == "__main__":
    generate_z_slices('./train_z', './train', [(128, 128),(256,256)], False)