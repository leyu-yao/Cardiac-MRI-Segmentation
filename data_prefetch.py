'''
this module prepare data directly for training and validation. 
'''
import numpy as np 
import torch
import sys 
import os
import matplotlib.pyplot as plt 
import nibabel as nib 
import PIL.Image as Image 
import sys

import util


def slice_2d_image(dst_dir, src_dir, num_classes):

    '''
    to use torch api



    '''
    x_y_fn_list = util.make_dataset(src_dir)

    idx = 0

    for x_path, y_path in x_y_fn_list:
        img_nib = nib.load(x_path)
        mask_nib = nib.load(y_path)
        
        

        # np.array D,H,W
        x = img_nib.get_fdata().astype(np.float32)
        y = mask_nib.get_fdata().astype(np.float32)

        # process y  as D,H,W  0~num_classes-1
        numl1 = [0, 205, 420, 500, 550, 600, 820, 850][:num_classes]
        numl2 = list(range(num_classes))

        y = util.remap_label_val(y, numl1, numl2)

        num_slices = x.shape[2]

        for i in range(num_slices):
            sys.stdout.flush()
            sys.stdout.write("processing %s, %d/%d" % (x_path, i+1, num_slices))
            img = x[:,:,i]
            mask = y[:,:,i]

            # tranform format , value range to save as png
            img = (255.0 / img.max() * (img - img.min())).astype(np.uint8)
            mask = mask.astype(np.uint8)

            Image.fromarray(img).save(os.path.join(dst_dir,"image%d.png"%idx))
            Image.fromarray(mask).save(os.path.join(dst_dir,"mask%d.png"%idx))

            idx += 1


if __name__ == "__main__":
    src_dir = './train'
    dst_dir = './train2d'
    num_classes = 8
    slice_2d_image(dst_dir, src_dir, num_classes)






