import torch.utils.data as data
import os
import torch
import numpy as np
import nibabel as nib
import util

class Dataset(data.Dataset):
    '''
    standard dataset
    read raw file from ./train
    give tensor as C,H,W,D
    '''
    def __init__(self, root, num_classes=8, transform=None):
        imgs = util.make_dataset(root, label_elem='_label.nii.gz')
        self.imgs = imgs
        self.transform = transform
        self.num_classes = num_classes

    def __getitem__(self, index):
        x_path, y_path = self.imgs[index]

        # read nii file
        # ---------------


        img_nib = nib.load(x_path)
        mask_nib = nib.load(y_path)

        # np.array D,H,W
        x = img_nib.get_fdata().astype(np.float32)
        y = mask_nib.get_fdata().astype(np.float32)

        # to tensor  C,D,H,W
        x = torch.from_numpy(x[np.newaxis,:,:,:])
        y = torch.from_numpy(util.one_hot(y, num_classes=self.num_classes))

        if self.transform is not None:
            x, y = self.transform(x, y)

        return x, y, x_path, img_nib.affine

    def __len__(self):
        return len(self.imgs)


