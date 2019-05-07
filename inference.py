
import argparse
import torch 
import numpy as np 

import nibabel as nib 
from tqdm import tqdm 
import transforms
from models import DSUnet3d
import util

class Network(object):
    def __init__(self, ckp):
        self.model = DSUnet3d(1,8).to('cuda')
        self.model.load_state_dict(torch.load(ckp, map_location='cuda'))
        self.model.eval()

    def __call__(self, vol):
        '''
        takes np D*H*W
        return y np C*D*H*W
        '''
        x = torch.from_numpy(vol[np.newaxis, np.newaxis, :,:,:].astype(np.float32)).cuda()
        with torch.no_grad():
            y, s1, s2, s3 = self.model(x)
        
        return y.cpu().numpy()[0]


class Segmentation(object):
    def __init__(self, fn, patch_size, stride, ckp):
        self.fn = fn
        self.patch_size = patch_size
        self.stride = stride

        # load np
        nb = nib.load(fn)
        self.img = nb.get_fdata()
        self.affine = nb.affine 
        self.shape = self.img.shape
        self.output_probs = np.ones((8, *self.shape))
        self.net = Network(ckp)

    def __call__(self):

        # logging
        print('processing %s' % self.fn)

        D, H, W = self.shape

        # iter times
        d = 1 + int((D-self.patch_size[0])/self.stride[0])
        h = 1 + int((H-self.patch_size[1])/self.stride[1])
        w = 1 + int((W-self.patch_size[2])/self.stride[2])

        with tqdm(total=(d+1)*(h+1)*(w+1)) as pbar:
            for dd in range(d+1):
                for hh in range(h+1):
                    for ww in range(w+1):
                        x1 = dd*self.stride[0]
                        x2 = dd*self.stride[0]+self.patch_size[0]
                        y1 = hh*self.stride[1]
                        y2 = hh*self.stride[1]+self.patch_size[1]
                        z1 = ww*self.stride[2]
                        z2 = ww*self.stride[2]+self.patch_size[2]

                        if x2 > D:
                            x1 = D - self.patch_size[0]
                            x2 = D
                        if y2 > H:
                            y1 = H - self.patch_size[1]
                            y2 = H
                        if z2 > W:
                            z1 = W - self.patch_size[2]
                            z2 = W


                        patch = self.img[x1:x2, y1:y2, z1:z2]
                        
                        # norm
                        patch = transforms.normlize(patch)
                        
                        patch_out = self.net(patch)


                        self.output_probs[:,x1:x2, y1:y2, z1:z2] += patch_out
                        pbar.update()

        label = np.argmax(self.output_probs, axis=0).astype(np.int16)

        img_wb = nib.Nifti1Image(label, self.affine)
        out_fn = self.fn.replace("image", "output")
        nib.save(img_wb, out_fn)

        # del
        #del self.net, self.out


if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument("action", type=str)
    parse.add_argument("--workspace", type=str)
    parse.add_argument("--block_size", nargs='+', type=int, default=(96,96,96))
    parse.add_argument("--stride", nargs='+', type=int, default=(48,48,48))
    parse.add_argument("--num_classes", type=int, default=8)
    parse.add_argument("--ckp", type=str)
    #parse.add_argument("--device", type=str, default='cuda')

    
    args = parse.parse_args()
    if args.action == 'infer':
        imgs = util.make_dataset(args.workspace)

        for x_path, _ in imgs:
            seg = Segmentation(x_path, args.block_size, args.stride, args.ckp)
            seg()
            