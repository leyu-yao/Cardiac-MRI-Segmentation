# %%import 3rd-party modules
import numpy as np 
import torch
import argparse
from torch.utils.data import DataLoader
from torch import autograd, optim, nn
from torchvision.transforms import transforms
import sys
import os 
import matplotlib.pyplot as plt
import time

# %%import project modules
from models import Unet3d, DSUnet3d
from dataset import Dataset3d
import losses

import util
import transforms

from timeTick import timeTicker

# %% train
def train(ckp_r, ckp_w, device_ids, multi_GPU, train_dir, val_dir, batch_size, block_size, num_epochs, train_len, val_len):

    # %% model
    model = DSUnet3d(1, 8).cuda()

    if ckp_r is not None:
        model.load_state_dict(torch.load(ckp_r, map_location='cuda'))

    if multi_GPU:
        model = nn.DataParallel(model, device_ids=device_ids)

    # %% optimizer
    conv1_parameters = list(map(id, model.conv1.parameters()))
    conv2_parameters = list(map(id, model.conv2.parameters()))
    conv3a_parameters = list(map(id, model.conv3a.parameters()))
    conv3b_parameters = list(map(id, model.conv3b.parameters()))
    conv4a_parameters = list(map(id, model.conv4a.parameters()))
    conv4b_parameters = list(map(id, model.conv4b.parameters()))

    
    union = conv1_parameters + conv2_parameters + conv3a_parameters + conv3b_parameters + conv4a_parameters + conv4b_parameters
    
    base_params = filter(lambda p: id(p) not in union, model.parameters())
    
    optimizer = torch.optim.SGD([
                {'params': base_params},
                {'params': model.conv1.parameters(), 'lr': 1e-6},
                {'params': model.conv2.parameters(), 'lr': 1e-6},
                {'params': model.conv3a.parameters(), 'lr': 1e-5},
                {'params': model.conv3b.parameters(), 'lr': 1e-5},
                {'params': model.conv4a.parameters(), 'lr': 1e-4},
                {'params': model.conv4b.parameters(), 'lr': 1e-4}],
                lr=1e-3, momentum=0.9, weight_decay=1e-4
                )

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.8)

    # criterion 
    criterion = losses.Total_Loss()

    # transform
    tran = transforms.RandomFlip()

    train_set = Dataset3d(train_dir, train_len, block_size, tran)
    val_set = Dataset3d(val_dir, val_len, block_size, tran)
    dataloader_train = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=False)
    dataloader_val = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False)

    t0 = time.time()
    # train
    for epoch in range(num_epochs):
        torch.cuda.empty_cache()
        print('Epoch: %d' % epoch)
        
        model.train()
        train_loss = 0
        
        for batch_idx, (inputs, targets) in enumerate(dataloader_train):
            
            optimizer.zero_grad()
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            #  val
        model.eval()
        val_loss = 0
        with torch.no_grad():
            val_loss = 0
            for x, y in dataloader_val:
                outputs = model(x.cuda())
                label = y.cuda()
                loss = criterion(outputs, label)
                val_loss += loss.item()
            
        t = time.time() - t0
        log = "{:^5d}s epoch{:^3d} train_loss:{:.3f} val_loss:{:.3f}".format(int(t), 
            epoch, train_loss/train_len, val_loss/val_len)
        scheduler.step()
        print(log)

        torch.save(model.state_dict(), ckp_w)

if __name__ == '__main__':
    train(None, '1.pth', [0], False, '../train', '../val', 4, (96,96,96), 100, 64, 16)