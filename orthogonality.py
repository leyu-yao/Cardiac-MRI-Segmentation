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
import argparse

import util
import models
import main_rev
import transform3d


def broswe_data_shape(root):
    data = util.make_dataset(root)

    for x, y in data:
        img = nib.load(x)
        print(x, img.shape)


class Resizer(object):
    def __init__(self, shape):
        #self.w, self.h = shape
        self.MPy = torch.nn.Upsample(size=shape).to('cuda')
        self.MPx = torch.nn.Upsample(size=shape, mode='bilinear').to('cuda')
    def __call__(self, x, y):
        '''
        takes x, y numpy C,D,H
        '''
        with torch.no_grad(): 
            x_t = torch.from_numpy(x[np.newaxis, :, :, :]).to('cuda')
            y_t = torch.from_numpy(y[np.newaxis, :, :, :]).to('cuda')

            return self.MPx(x_t)[0].cpu().numpy(), self.MPy(y_t)[0].cpu().numpy()

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

def Generate():
    shapes_main = [(256,256)]
    shapes_side = [(256,128)]
    generate_z_slices('./train_z', './train', shapes_main, False)
    generate_x_slices('./train_x', './train', shapes_side, False)
    generate_y_slices('./train_y', './train', shapes_side, False)

def train_orthogonality(batch_size, num_epochs, workspace, weight_name, ckp):
    #tran = transform3d.data_augumentation_2d(288)
    tran = transform3d.Normalization()
    main_rev.train2d(num_classes=8, batch_size=batch_size, num_epochs=num_epochs, 
    workspace=workspace, device='cuda', transform=None,weight_name=weight_name, ckp=ckp)

if __name__ == "__main__":
    # shapes_main = [(128,128),(192,192),(256,256),(320,320)]
    # shapes_side = [(256,128),(512,128),(256,140),(256,192),(320,128)]
    # generate_z_slices('./train_z', './train', [(128, 128),(256,256)], False)

    parse = argparse.ArgumentParser()

    parse.add_argument("action", type=str)
    parse.add_argument("--workspace", type=str)
    parse.add_argument("--ckp", type=str, help="the path of model weight file")
    parse.add_argument("--num_epochs", type=int, default=25)
    parse.add_argument("--batch_size", type=int, default=25)
    args = parse.parse_args()

    if args.action == 'Generate':
        Generate()

    elif args.action == 'train_z':
        train_orthogonality(args.batch_size, args.num_epochs, './train_z', '2d_z', args.ckp)
    
    elif args.action == 'train_x':
        train_orthogonality(args.batch_size, args.num_epochs, './train_x', '2d_x', args.ckp)

    elif args.action == 'train_y':
        train_orthogonality(args.batch_size, args.num_epochs, './train_y', '2d_y', args.ckp)
