import torch.utils.data as data
import PIL.Image as Image
import os
import torch

import numpy as np

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

class Dataset2d(data.Dataset):
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

        #print(img_x.size())
        return img_x, img_y

    def __len__(self):
        return len(self.imgs)

if __name__ == '__main__':
    from torchvision.transforms import transforms
    
    
    dataset = Dataset3d("./train",transform=None)