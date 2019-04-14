import torch
import numpy as np 
import nibabel as nib
import os

class DownSample(object):
    '''
    A adaptive Downsample using torch.nn
    '''
    def __init__(self, output_size, device='cuda'):
        '''
        size tuple (h,w,d)
        '''
        self.output_size = output_size
        self.device = device

    def _np_to_tensor(self, np_arr):
        '''
        a aux func to transfer np arr to torch.tensor
        using self.device

        input 1,W,H,D
        output 1,1,W,H,D
        '''
        add_axis = np_arr[np.newaxis,:,:,:,:].astype(np.float32)

        ten = torch.from_numpy(add_axis).to(self.device)

        return ten

    def _tensor_to_np(self, ten):
        '''
        aux func to transfer tensor to np array
        1,1,w,h,d  =>> 1,w,h,d
        '''
        arr = ten[0,:,:,:,:].cpu().numpy()
        return arr

    

    def __call__(self, x, y=None):
        '''
        input np array 1,W,H,D
        '''
        m = torch.nn.AdaptiveMaxPool3d(self.output_size)

        x_low = m(self._np_to_tensor(x))
        
        x_low = self._tensor_to_np(x_low)
        
        if y is not None:
            y_low = m(self._np_to_tensor(y))
            return x_low, y_low
        else:
            return x_low

    


if __name__ == "__main__":
    ds = DownSample((176, 176, 128))
    
    num_list = [1001, 1003, 1012]
    
    type_list = ['label', 'image']
    
    iter_list = [(n,t) for n in num_list for t in type_list]
    
    for n, t in iter_list:
        fn = os.path.join('./', 'mr_train_%d_%s.nii.gz' % (n, t))
        img = nib.load(fn)
        print(img.shape)
    
        affine = img.affine
        x = img.get_fdata()[np.newaxis,:,:,:]
    
        x_low = ds(x)[0]
        
        
        img = nib.Nifti1Image(x_low, affine)
        fn = os.path.join('./roi_test', 'mr_train_%d_%s.nii.gz' % (n, t))
        nib.save(img, fn)


