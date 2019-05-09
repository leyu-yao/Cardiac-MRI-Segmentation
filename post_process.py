from scipy import ndimage 
import numpy as np 
import nibabel as nib 
import os
import argparse

def max_connected_component(img):
    '''
    input img np.array  D * H *W
    return img np.array  D * H *W
    input format : labeled as 0,1,2,...,7
    '''
    num_classes = int(np.max(img)) #7
    img_out = img.copy()
    
    structure = np.ones((3,3,3))
    # for each class
    for c in range(1,num_classes+1):
        arr = np.zeros(img.shape)
        arr[img==c] = 1
        labeled_array, num_ccs = ndimage.label(arr, structure=structure)

        # find maxinum connect component
        max_measurement, max_idx = 0, 0
        for idx in range(1,num_ccs+1):
            vol = (labeled_array==idx).sum()
            if vol > max_measurement:
                max_measurement = vol
                max_idx = idx

        #removed_label = list(range(1,num_ccs+1)).remove(max_idx)

        # remove small ccs
        for i in range(1,num_ccs+1):
            if i != max_idx:
                img_out[labeled_array==i] = 0
        
    return img_out

def process_nii_file(in_fn, out_fn):
    nb = nib.load(in_fn)
    img = nb.get_fdata()
    affine = nb.affine

    img_wb = max_connected_component(img)
    
    nb_wb = nib.Nifti1Image(img_wb, affine)
    nib.save(nb_wb, out_fn)

def process_folder(in_dir, out_dir, elem='output.nii.gz'):
    imgs = []
    files = os.listdir(in_dir)
    for file in files:
        if elem in file:
            in_fn = os.path.join(in_dir, file)
            out_fn = os.path.join(out_dir, file.replace('output', 'post'))
            imgs.append((in_fn, out_fn))
        else:
            continue
    return imgs

if __name__ == '__main__':
    parse = argparse.ArgumentParser()

    parse.add_argument('dst', type=str)
    parse.add_argument('src', type=str)
    parse.add_argument('--element', type=str, default='output.nii.gz')
    args = parse.parse_args()

    imgs = process_folder(args.src, args.dst, args.element)
    for in_fn, out_fn in imgs:
        process_nii_file(in_fn, out_fn)




