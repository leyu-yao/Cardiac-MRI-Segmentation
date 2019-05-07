import torch
import models
import C3D_model

# %% switch keys
unet_keys = ['conv1.bias',
            'conv2.weight', 'conv2.bias',
            'conv3a.weight', 'conv3a.bias',
            'conv3b.weight', 'conv3b.bias',
            'conv4a.weight', 'conv4a.bias',
            'conv4b.weight', 'conv4b.bias']


c3d_keys = ['conv1.bias',
            'conv2.weight', 'conv2.bias',
            'conv3a.weight', 'conv3a.bias',
            'conv3b.weight', 'conv3b.bias',
            'conv4a.weight', 'conv4a.bias',
            'conv4b.weight', 'conv4b.bias']


# %% load keys for unet
def load_keys_for_unet(unet, c3d_path='c3d.pickle'):
    device = torch.device('cuda')
    unet_dict = unet.state_dict()
    c3d_dict = torch.load(c3d_path, map_location=device)
    for u, c in zip(unet_keys, c3d_keys):
        unet_dict[u] = c3d_dict[c]
    
    unet_dict['conv1.weight'][:,0,:,:,:] = c3d_dict['conv1.weight'][:,0,:,:,:]
    unet.load_state_dict(unet_dict)

    return unet

# %% test

#x = torch.rand(1,1,8,8,8).cuda()
#unet = models.Unet3d(1,3).cuda()
#y0 = unet(x)
#unet = load_keys_for_unet(unet)
#y1 = unet(x)

# %% Generate Checkpoint for Unet 
if __name__ == '__main__':
    unet = models.DSUnet3d(1,8).cuda()
    unet = load_keys_for_unet(unet)
    torch.save(unet.state_dict(), 'transfer_learning_ckp.pth')
