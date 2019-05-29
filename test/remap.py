import numpy as np

import nibabel as nib


def remap_nii(fn, gain=1):
    '''
    remap label [0, 1, 2, 3, 4, 5, 6, 7] to [  0, 205, 420, 500, 550, 600, 820, 850]
    '''
    img = nib.load(fn)

    affine = img.affine
    mat = img.get_fdata()
    mat /= gain

    numl = [0, 205, 420, 500, 550, 600, 820, 850]
    for i, value in enumerate(numl):
        mat[mat == value] = i
    
    img = nib.Nifti1Image(mat, affine)
    
    #nib.save(img, os.path.join('build','test4d.nii.gz'))
    nib.save(img, fn)