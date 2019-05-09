'''
This module takes input a list of files, 
read all nii files labels and outputs,
calculate metrics and print
also print some useful parameters
'''

import numpy as np 
from scipy import ndimage
from scipy import spatial
import nibabel as nib 
from prettytable import PrettyTable
import os
import argparse

test_sample_idx = [1, 3, 4, 7]
labels = ['mr_train_10%02d_label.nii.gz'%idx for idx in test_sample_idx]
outputs = ['mr_train_10%02d_post.nii.gz'%idx for idx in test_sample_idx]
#outputs = ['mr_train_10%02d_label.nii.gz'%idx for idx in test_sample_idx]

files = [(u, v) for u, v in zip(labels, outputs)]

val_output = [0, 1, 2, 3, 4, 5, 6, 7]
#val_output = [0, 205, 420, 500, 550, 600, 820, 850]

val_label = [0, 205, 420, 500, 550, 600, 820, 850]
name = ['NONE', 'MLV', 'LABV', 'LVBC', 'RABC', 'RVBC', 'ASA', 'PUA']


def calculate_jaccard(output, label):
    '''
    return a dict
    '''
    jaccard_score = {}

    for i in range(1, 8):
        key = name[i]
        u = (output==val_output[i]).reshape(-1)
        v = (label==val_label[i]).reshape(-1)
        distance = spatial.distance.jaccard(u, v)
        jaccard_score[key] = 1 - distance
    
    return jaccard_score

def calculate_DSC(output, label):
    '''
    return a dict
    '''
    DSC_score = {}

    for i in range(1, 8):
        key = name[i]
        u = (output==val_output[i]).reshape(-1)
        v = (label==val_label[i]).reshape(-1)
        distance = spatial.distance.dice(u, v)
        DSC_score[key] = 1 - distance
    
    return DSC_score


if __name__ == '__main__':
    parse = argparse.ArgumentParser()

    parse.add_argument("--label_dir", type=str, default='./test')
    parse.add_argument("--output_dir", type=str, default='./test')
    args = parse.parse_args()

    for label_fn, output_fn in files:
        label_arr = nib.load(os.path.join(args.label_dir, label_fn)).get_fdata()
        output_arr = nib.load(os.path.join(args.output_dir, output_fn)).get_fdata()

        JACCARD = calculate_jaccard(output_arr, label_arr)
        DSC = calculate_DSC(output_arr, label_arr)

        tab = PrettyTable(['Metrics'] + [m for m in name[1:]] + ['MEAN'])
        tab.title = output_fn
        tab.add_row(['DSC'] + [DSC[k] for k in name[1:]] + [sum(list(DSC.values())) / len(DSC)])
        tab.add_row(['Jaccard'] + [JACCARD[k] for k in name[1:]]  + [sum(list(JACCARD.values())) / len(JACCARD)])

        print(tab)
        print('\n')