# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 20:05:22 2019

@author: Dynasting
"""
import numpy as np
import shutil
import os
import sys
import random
import argparse
import torch
import nibabel as nib

def make_dataset(root, label_elem='_label.nii.gz'):
    imgs=[]
    files = os.listdir(root)
    for file in files:
        if label_elem in file:
            mask = os.path.join(root, file)
            img = mask.replace('label','image')
            imgs.append((img,mask))
 
        else:
            continue
    
    return imgs





def split_dataset(dst_dir, src_dir, fraction=0.3, img_elem='_image.npy',mask_elem='_label.npy'):
    '''
    function:
        randomly choose a fraction of (img, mask) to another folder
    '''

    img_mask_tuple_list = make_dataset(src_dir, label_elem=mask_elem)
    
    if img_mask_tuple_list is None:
        return
    
    total_num = len(img_mask_tuple_list)
    
    to_be_mov_num = int(fraction * total_num)
    
    random.shuffle(img_mask_tuple_list)
    
    list_to_be_mov = img_mask_tuple_list[0:to_be_mov_num]
    
    for (img,mask) in list_to_be_mov:
        (_, img_filename) = os.path.split(img)
        (_, msk_filename) = os.path.split(mask)
        
        new_img = os.path.join(dst_dir, img_filename)
        new_msk = os.path.join(dst_dir, msk_filename)
        
        shutil.move(img, new_img)
        shutil.move(mask, new_msk)



def output_write_to_file(output, filename, affine=None):
    '''
    This function takes in output of a model [N,C,D,H,W]  N = 1
    Determine the class predicted
    Generate image of (C,D,H,W) C = 1
    Save to a .NII File
    Visualize NII file if needed

    output_write_to_file(torch.rand(1,3,500,500,100), 'test.nii.gz')

    '''
    mat = torch.argmax(output[0], dim=0, keepdim=False)

    # to increase visualization quality
    # Now using the same value as label
    numl = [  0, 205, 420, 500, 550, 600, 820, 850]
    #mat *= 120
    for i, value in enumerate(numl):
        mat[mat==i] = value


    # mat (D,H,W)
    
    # write to nii
    
    # get affine
    if affine is None:
        affine = np.eye(4)# have to be 4
    
    data_np = mat.cpu().numpy().astype(np.int16)
    img = nib.Nifti1Image(data_np, affine)
    
    #nib.save(img, os.path.join('build','test4d.nii.gz'))
    nib.save(img, filename)

def tensor_or_arr_write_to_nii_file(inp, filename=None, affine=np.eye(4)):
    '''
    possible input
    tensor 1,1,W,H,D
    np arr 1, W, H, D
    '''
    if filename is None:
        id = hash(inp)
        fn = "%s.nii.gz" % str(id)
    else:
        fn = filename

    # tensor
    if type(inp) == torch.Tensor:
        out = inp[0,0,:,:,:].cpu().numpy()
        img = nib.Nifti1Image(out, affine)
        nib.save(img, fn)
    
    # array
    elif type(inp) == numpy.array:
        out = inp[0,:,:,:]
        img = nib.Nifti1Image(out, affine)
        nib.save(img, fn)

def one_hot(np_label, num_of_class = 5):
    shape = np_label.shape
    out = np.zeros((num_of_class, *shape), dtype=np.float32)
    
    numl = [  0, 205, 420, 500, 550, 600, 820, 850]
    # Need to handle num!=8 other class -> class0
    for i, num in enumerate(numl[0:num_of_class]):
        if i==0:
            continue
        out[i,:,:,:][np_label==num]=1

    out[0,:,:,:] = 1 - np.sum(out[1:,:,:,:], axis=0)
    

    # binary class
#    res = np.zeros((2,*shape),dtype=np.float32)
#    #res[0,:,:,:] = np.eyes(out.size) - out[0,:,:,:]
#    res[0,:,:,:] = out[0,:,:,:]
#    res[1,:,:,:] = 1 - res[0,:,:,:]
    
    
    
    return out

def label_to_fore_and_background(np_label):
    '''
    input np_label D,H,W
    output 1,D,H,W
    '''
    shape = np_label.shape
    out = np.zeros((1, *shape), dtype=np.float32)

    numl = [  0, 205, 420, 500, 550, 600, 820, 850]
    
    for i, num in enumerate(numl[1:]):
        out[0,:,:,:][np_label==num]=1

    return out

def read_nii_as_np(img_fn, mask_fn, num_classes):
    '''
    output size
    (x,y,z) (c,x,y,z) (4,4)
    '''
    # read nii.gz file
    img = nib.load(img_fn)
    msk = nib.load(mask_fn)
    
    affine=img.affine
    
    # transform the matrix into numpy array
    img_np = img.get_fdata()
    mask_np = msk.get_fdata()
    mask_np_oh = one_hot(mask_np, num_classes)
    
    return img_np, mask_np_oh, affine


def fix_unicode_bug():
    '''
    fix print unicode bug
    '''
    import win_unicode_console
    win_unicode_console.enable()

def remap_nii(fn, gain=1):
    '''
    remap label [0, 1, 2, 3, 4, 5, 6, 7] to [  0, 205, 420, 500, 550, 600, 820, 850]
    '''
    img = nib.load(fn)

    affine = img.affine
    mat = img.get_fdata()
    mat /= gain

    numl = [  0, 205, 420, 500, 550, 600, 820, 850]
    for i, value in enumerate(numl):
        mat[mat==i] = value
    
    img = nib.Nifti1Image(mat, affine)
    
    #nib.save(img, os.path.join('build','test4d.nii.gz'))
    nib.save(img, fn)

def remap_label_val(y, numl1, numl2):
    '''
    y: numpy or tensor as D,H,W
    '''
    for v1, v2 in zip(numl1, numl2):
        y[y==v1] = v2
    
    return y
     
if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    
    parse.add_argument("action", type=str)
    parse.add_argument("--dst_dir", type=str)
    parse.add_argument("--src_dir", type=str)
    parse.add_argument("--fraction", type=float, default=0.3)
    parse.add_argument("--img_elem", type=str, default='_image.npy')
    parse.add_argument("--mask_elem", type=str, default='_label.npy')
    parse.add_argument("--filename", type=str)
    args = parse.parse_args()

    if args.action=="split":
        split_dataset(args.dst_dir, args.src_dir, args.fraction, args.img_elem,
                      args.mask_elem)
        #split ./test2d ./train2d 0.3

    elif args.action == "remap":
        remap_nii(args.filename, 120)




    
