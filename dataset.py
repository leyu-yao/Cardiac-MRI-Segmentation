import torch.utils.data as data
import PIL.Image as Image
import os
import torch
import random
import numpy as np
import transforms
def make_dataset(root, content='_label.npy'):
    imgs=[]
    files = os.listdir(root)
    for file in files:
        if content in file:
            mask = os.path.join(root, file)
            img = mask.replace('label','image')
            imgs.append((img,mask))
 
        else:
            continue
    
    return imgs
    


""" 
class Dataset3d(data.Dataset):
    def __init__(self, root, transform=None):
        imgs = make_dataset(root)
        self.imgs = imgs
        self.transform = transform


    def __getitem__(self, index):
        x_path, y_path = self.imgs[index]
        
        img_x = np.load(x_path)
        img_y = np.load(y_path)
        
        # 2-class
#        y = np.zeros((2,*img_y.shape[1:]),dtype=np.float32)
#        y[0,:,:,:] = np.sum(img_y, axis=0)
#        y[1,:,:,:] = img_y[1,:,:,:]

        # to tensor
        img_x = torch.from_numpy(img_x)
        img_y = torch.from_numpy(img_y)
        
        
        # if self.transform is not None:
        #     img_x = self.transform(img_x)
        # if self.target_transform is not None:
        #     img_y = self.target_transform(img_y)
        if self.transform is not None:
            img_x, img_y = self.transform(img_x, img_y)
            

        #print(img_x.size(), img_y.size())
        return img_x, img_y

    def __len__(self):
        return len(self.imgs)

 """
class Dataset3d(data.Dataset):
    def __init__(self, root, num, block_size=(96,96,96), transform=None):
        imgs = make_dataset(root)
        self.block_size = block_size
        self.transform = transform
        self.num = num
        self.imgs = []
        # pre load
        for img_fn, mask_fn in imgs:
            img = nib.load(img_fn)
            mask = nib.load(mask_fn)
            (D, H, W) = img.shape

            img = img.get_fdata()
            mask = mask.get_fdata()

            # one hot
            mask = util.one_hot(mask, num_of_class=num_classes)
            
            # add axis
            img = img[np.newaxis,:,:,:]
            

            
            # np.float32
            img = img.astype(np.float32)
            mask = mask.astype(np.float32)

            self.imgs.append((img, mask))


    def __getitem__(self, index):
        img, mask = random.choice(self.imgs)
        C, D, H, W = mask.shape
        d_max = D - self.block_size[0]
        h_max = H - self.block_size[1]
        w_max = W - self.block_size[2]
        
        d = random.randint(0, d_max)
        h = random.randint(0, h_max)
        w = random.randint(0, w_max)

        x_patch = img[:, d:d+self.block_size[0], h:h+self.block_size[1], w:w+self.block_size[2]]
        y_patch = mask[:, d:d+self.block_size[0], h:h+self.block_size[1], w:w+self.block_size[2]]
        x_patch, y_patch = transforms.Normalization(x_patch, y_patch)
        
        x_patch, y_patch = torch.from_numpy(x_patch), torch.from_numpy(y_patch)
        if self.transform is not None:
            x_patch, y_patch = self.transform(x_patch, y_patch)

        return x_patch, y_patch


    def __len__(self):
        return self.num

if __name__ == '__main__':
    from torchvision.transforms import transforms
    
    
    dataset = Dataset3d("./train",transform=None)