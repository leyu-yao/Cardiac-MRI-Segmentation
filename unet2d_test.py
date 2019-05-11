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
from tqdm import tqdm 
import argparse 

import util
import models
import losses
import transform3d


class test_on_2dUnet_Z(object):
    def __init__(self, workspace, ckp, device, resolution, num_classes):
        self.workspace = workspace
        self.ckp = ckp
        self.device = device
        self.resolution = resolution
        self.num_classes = num_classes

        self.imgs = util.make_dataset(workspace)
        # load model
        #self.model = models.Unet2d(1,self.num_classes).to(device)
        self.model = models.InceptionXUnet2d(1,self.num_classes).to(device)
        self.model.load_state_dict(torch.load(ckp, map_location=device))
        self.model.eval()
        self.criterion = losses.DiceLoss()
        self.mp = torch.nn.Upsample(size=resolution, mode='bilinear')

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
                
                up_sample = torch.nn.Upsample(size=(D, H))

                # transform to np  d,h,w
                img = img.get_fdata().astype(np.float32)
                mask = mask.get_fdata().astype(np.float32)



                # x_tensor
                x_tensor = torch.from_numpy(img[np.newaxis,np.newaxis,:,:,:]).to(self.device)

                # y_tensor
                y_tensor = torch.from_numpy(util.one_hot(mask, 8)[np.newaxis,:,:,:,:]).to(self.device)

                # out tensor
                out_tensor = torch.zeros(1,8,D,H,W)#.to(self.device)

                for w in tqdm(range(W)):
                    x = self.mp(x_tensor[:,:,:,:,w])

                    x = transform3d.clahe_for_tensor(x)

                    out_tensor[:,:,:,:,w] = up_sample(self.model(x))

                #out_tensor = out_tensor.argmax(dim=1, keepdim=True)

                #score = 1 - self.criterion(out_tensor, y_tensor)
                #print("score of %s is %0.3f" % (x_path, score))

                #np.save(x_path.replace('image', 'output_z_np'), out_tensor[0].cpu().numpy())
                result_np = out_tensor[0].argmax(dim=0).cpu().numpy()

                res = nib.Nifti1Image(result_np.astype(np.float32), affine)

                nib.save(res, x_path.replace('image', 'output_z'))
                
                del x_tensor, y_tensor, out_tensor, result_np



class test_on_2dUnet_X(object):
    def __init__(self, workspace, ckp, device, resolution, num_classes):
        self.workspace = workspace
        self.ckp = ckp
        self.device = device
        self.resolution = resolution
        self.num_classes = num_classes

        self.imgs = util.make_dataset(workspace)
        # load model
        #self.model = models.Unet2d(1,self.num_classes).to(device)
        self.model = models.InceptionXDenseUnet2d(1,self.num_classes).to(device)
        self.model.load_state_dict(torch.load(ckp, map_location=device))
        self.model.eval()
        self.criterion = losses.DiceLoss()
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
                
                up_sample = torch.nn.AdaptiveMaxPool2d((H, W))

                # transform to np  d,h,w
                img = img.get_fdata().astype(np.float32)
                mask = mask.get_fdata().astype(np.float32)



                # x_tensor
                x_tensor = torch.from_numpy(img[np.newaxis,np.newaxis,:,:,:]).to(self.device)

                # y_tensor
                y_tensor = torch.from_numpy(util.one_hot(mask, 8)[np.newaxis,:,:,:,:]).to(self.device)

                # out tensor
                out_tensor = torch.zeros(1,8,D,H,W)#.to(self.device)

                for d in tqdm(range(D)):
                    x = self.mp(x_tensor[:,:,d,:,:])
                    out_tensor[:,:,d,:,:] = up_sample(self.model(x))

                #out_tensor = out_tensor.argmax(dim=1, keepdim=True)

                #score = 1 - self.criterion(out_tensor, y_tensor)
                #print("score of %s is %0.3f" % (x_path, score))

                
                #np.save(x_path.replace('image', 'output_x_np'), out_tensor[0].cpu().numpy())

                result_np = out_tensor[0].argmax(dim=0).cpu().numpy()

                res = nib.Nifti1Image(result_np.astype(np.float32), affine)

                nib.save(res, x_path.replace('image', 'output_x'))
                
                del x_tensor, y_tensor, out_tensor, result_np



class test_on_2dUnet_Y(object):
    def __init__(self, workspace, ckp, device, resolution, num_classes):
        self.workspace = workspace
        self.ckp = ckp
        self.device = device
        self.resolution = resolution
        self.num_classes = num_classes

        self.imgs = util.make_dataset(workspace)
        # load model
        #self.model = models.Unet2d(1,self.num_classes).to(device)
        self.model = models.InceptionXDenseUnet2d(1,self.num_classes).to(device)
        self.model.load_state_dict(torch.load(ckp, map_location=device))
        self.model.eval()
        self.criterion = losses.DiceLoss()
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
                
                up_sample = torch.nn.AdaptiveMaxPool2d((D, W))

                # transform to np  d,h,w
                img = img.get_fdata().astype(np.float32)
                mask = mask.get_fdata().astype(np.float32)



                # x_tensor
                x_tensor = torch.from_numpy(img[np.newaxis,np.newaxis,:,:,:]).to(self.device)

                # y_tensor
                y_tensor = torch.from_numpy(util.one_hot(mask, 8)[np.newaxis,:,:,:,:]).to(self.device)

                # out tensor
                out_tensor = torch.zeros(1,8,D,H,W)#.to(self.device)

                for h in tqdm(range(H)):
                    x = self.mp(x_tensor[:,:,:,h,:])
                    out_tensor[:,:,:,h,:] = up_sample(self.model(x))

                #out_tensor = out_tensor.argmax(dim=1, keepdim=True)

                #score = 1 - self.criterion(out_tensor, y_tensor)
                #print("score of %s is %0.3f" % (x_path, score))

                
                #np.save(x_path.replace('image', 'output_y_np'), out_tensor[0].cpu().numpy())

                result_np = out_tensor[0].argmax(dim=0).cpu().numpy()

                res = nib.Nifti1Image(result_np.astype(np.float32), affine)

                nib.save(res, x_path.replace('image', 'output_y'))
                
                del x_tensor, y_tensor, out_tensor, result_np



if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument("axis", type=str)
    parse.add_argument("ckp", type=str)
    parse.add_argument("workspace", type=str)
    parse.add_argument("--resolution", nargs='+', type=int, default=(224,224))

    args = parse.parse_args()

    if args.axis == 'z':
        test = test_on_2dUnet_Z(args.workspace, args.ckp, torch.device('cuda'), args.resolution, 8)
    elif args.axis == 'y':
        test = test_on_2dUnet_Y(args.workspace, args.ckp, torch.device('cuda'), args.resolution, 8)
    elif args.axis == 'x':
        test = test_on_2dUnet_X(args.workspace, args.ckp, torch.device('cuda'), args.resolution, 8)
    
    test()