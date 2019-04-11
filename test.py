# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 20:27:13 2019

@author: Dynasting

This file contains some functions for testing pre-trained model.
"""
import argparse

import torch
import numpy as np

import util
import metrics
from models import Unet3d
from util import read_nii_as_np, make_dataset

'''
To do:
1. 考虑封装一个类，带模型带数据集
2. filename -> np
3. np -> tensor
4. tensor -> metric
5. tensor -> np
6. np -> nii
'''
class Network_pretrained(object):
    '''
    a network pretrained 
    method : __call__ (tensor)
    
    usage 
    >> net = Network_pretrained(ckp, device, num_classes)
    >> y = net(x)

    input x tensor (1,1,X,Y,Z)
    output y tensor(1,C,X,Y,Z)
    on given device
    '''
    def __init__(self, ckp, device, num_classes=5):
        self.ckp = ckp
        self.device = device

        self.model = Unet3d(1, num_classes).to(device)
        
        self.model.load_state_dict(torch.load(ckp,map_location=device))
        self.model.eval()

    def __call__(self, x):
        '''
        para x tensor in (1,1,X,Y,Z)
        return y tensor in (1,C,X,Y,Z)
        '''
        with torch.no_grad():
            output = self.model(x.to(self.device))
        return output


class Np_Cutter(object):
    '''
    store block-size
    
    generate block_list from array
    '''
    def __init__(self, block_size, stride):
        self.block_size = block_size
        self.stride = stride

    def __call__(self, arr):
        '''
        calculate num_blocks
        return list of (block_arr, (pos_x, pos_y, pos_z))
        '''
        D, H, W = arr.shape
        d = 1 + int((D-self.block_size[0])/self.stride[0])
        h = 1 + int((H-self.block_size[1])/self.stride[1])
        w = 1 + int((W-self.block_size[2])/self.stride[2])

        res = []
        
        for dd in range(d+1):
            for hh in range(h+1):
                for ww in range(w+1):
                    x1 = dd*self.stride[0]
                    x2 = dd*self.stride[0]+self.block_size[0]
                    y1 = hh*self.stride[1]
                    y2 = hh*self.stride[1]+self.block_size[1]
                    z1 = ww*self.stride[2]
                    z2 = ww*self.stride[2]+self.block_size[2]

                    if x2 > D:
                        x1 = D - self.block_size[0]
                        x2 = D
                    if y2 > H:
                        y1 = H - self.block_size[1]
                        y2 = H
                    if z2 > W:
                        z1 = W - self.block_size[2]
                        z2 = W

                    # 3d
                    block_arr = arr[x1:x2, y1:y2, z1:z2]
                    pos_tuple = (x1, y1, z1)
                    
                    res.append((block_arr.astype(np.float32), pos_tuple))

        return res



class Np_Tensor_Converter(object):
    '''
    takes in numpy array X,Y,Z
    return tensor on device 1,1,X,XY,Z
    '''
    def __init__(self, device):
        # save device
        self.device = device

    def np2tensor(self, np_arr):
        '''
        X x,y,z np => 1,1,x,y,z tensor
        Y C,X,Y,Z => 1,C,x,y,z 
        '''
        shape = np_arr.shape
        if len(shape) == 4:
            return torch.from_numpy(np_arr[np.newaxis,:,:,:,:]).to(self.device)
        elif len(shape) == 3:
            np_arr_ = np_arr[np.newaxis,np.newaxis,:,:,:]
            return torch.from_numpy(np_arr_).to(self.device)
        pass

    def tensor2np(self, tensor):
        '''
        1,C,x,y,z tensor => C,x,y,z np
        '''
        return tensor.squeeze().cpu().numpy()
        
    
        

class Crusher(object):
    def __init__(self, res_shape_tuple, block_size_tuple):
        '''
        res_shape_tuple C,X,Y,Z
        res C,X,Y,Z
        marker X,Y,Z
        '''
        self.res = np.zeros(tuple(res_shape_tuple))
        self.marker = np.zeros(tuple(res_shape_tuple)[1:])
        self.dx, self.dy, self.dz = block_size_tuple
    
    def update(self, arr, pos):
        '''
        arr output_np C,X,Y,Z
        pos (x,y,z)
        '''
        x,y,z = pos
        print(pos)
        self.res[:, x:x+self.dx, y:y+self.dy, z:z+self.dz] = arr
        self.marker[x:x+self.dx, y:y+self.dy, z:z+self.dz] += 1

    def __call__(self):
        '''
        crush
        fix the overlap
        return whole np_array C,X,Y,Z
        '''
        c = self.res.shape[0]
        for _ in range(c):
            self.res[_] /= self.marker
        return self.res


class Test_on_file(object):
    '''
    1.read file from disk 
    2.save to np
    3.cut np into block lists(arr,pos)
    4.collect output from model
    5.reconstruct pred np
    6.init a AUC meter
    '''

    def __init__(self, img_fn, msk_fn, block_size, cvt, net, stride):
        '''
        block_size tuple
        Read nii file. \n
        Divide into block lists. Using Np_cutter \n
        '''
        self.net = net
        #self.device = device
        self.block_size = block_size
        self.img_fn = img_fn
        self.msk_fn = msk_fn
        (self.img_np, self.mask_np, self.affine) = read_nii_as_np(img_fn, msk_fn, 8)
        #(X,Y,Z) (C,X,Y,Z)
        self.cutter = Np_Cutter(block_size, stride)
        self.block_list = self.cutter(self.img_np)#[(block_arr, (x, y, z))]

        #self.img_tensor = None
        #self.msk_tensor = None
        self.cvt = cvt
        self.crusher = Crusher(self.mask_np.shape, block_size)
        self.AUC = metrics.Metric_AUC()

    def __call__(self):
        '''
        1. Iterate through block list \n
        during iteration \n
        1)calculate output
        2)update by crusher
        2. Comfirm y_pred with marker
        3. Calculate score
        4. Prepare NII array
        5. Write NII to disk
        '''
        for block_arr, pos in self.block_list:
            x_tensor = self.cvt.np2tensor(block_arr)
            y_tensor = self.net(x_tensor)
            self.crusher.update(self.cvt.tensor2np(y_tensor), pos)

        pred_np = self.crusher()
        pred_tensor = self.cvt.np2tensor(pred_np);del pred_np
        score = self.AUC(pred_tensor, self.cvt.np2tensor(self.mask_np))

        #output file
        fn = self.msk_fn.replace("label", "pred")
        
        util.output_write_to_file(pred_tensor, fn, self.affine)

        return score


class Test(object):
    '''
    Class Test for test of a pre-trained model 
    on a certain dataset.
    '''

    def __init__(self, workspace, device, ckp, block_size, num_classes, stride):
        '''
        construt model and load weights
        make dataset with filename
        '''
        self.workspace = workspace
        self.device = device
        self.ckp = ckp
        self.block_size = block_size
        self.num_classes = num_classes
        self.net = Network_pretrained(ckp, device, num_classes)
        self.cvt = Np_Tensor_Converter(self.device)
        self.stride = stride

    def __call__(self):
        '''
        run test on all data
        '''
        img_msk_fn_list = make_dataset(self.workspace)

        for img_fn, msk_fn in img_msk_fn_list:
            file_test = Test_on_file(img_fn, msk_fn, self.block_size, self.cvt, self.net, self.stride)
            score = file_test()
            print(score)



if __name__ == "__main__":
    # workspace = "./test"
    # device = 'cuda'
    # ckp = "weights_1_Unet3d.pth"
    # block_size = (32,32,32)
    # num_classes = 5
    # test = Test(workspace, device, ckp, block_size, num_classes)
    # test()

    parse = argparse.ArgumentParser()
    parse.add_argument("action", type=str)
    parse.add_argument("--workspace", type=str, default="./test")
    parse.add_argument("--device", type=str, default="cuda")
    parse.add_argument("--ckp", type=str)
    parse.add_argument("--block_size", type=int, nargs='+', default=(128,128,64))
    parse.add_argument("--stride", type=int, nargs='+', default=(64,64,32))
    parse.add_argument("--num_classes", type=int, default=5)

    args = parse.parse_args()

    if args.action == "test":
        test = Test(args.workspace, args.device, args.ckp, 
                    tuple(args.block_size), args.num_classes, args.stride)
        test()
        
