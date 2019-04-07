# import 3rd-party modules
import numpy as np 
import torch
import argparse
from torch.utils.data import DataLoader
from torch import autograd, optim, nn
from torchvision.transforms import transforms
import sys
import matplotlib.pyplot as plt

# import project modules
from unet3d import SmallUnet3d, Unet3d, Unet2d
from dataset import Dataset3d, Dataset2d
from loss_function import CrossEntropy3d, DiceLoss ,CrossEntropyDiceLoss
from util import fix_unicode_bug

fix_unicode_bug()


def train_model(model, criterion, optimizer, dataload, num_epochs, device, parallel):
    '''
    train procedure
    '''
    # DataParallel
    if torch.cuda.device_count() > 1 and parallel:
        print("Using", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)

    for epoch in range(1, num_epochs+1):
        print('Epoch {}/{}'.format(epoch, num_epochs))
        print('-' * 10)
        dt_size = len(dataload.dataset)
        epoch_loss = 0
        step = 0
        for x, y in dataload:
            step += 1
            inputs = x.to(device)
            labels = y.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward
            outputs = model(inputs)
            
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            print("In epoch %d, %d/%d,train_loss:%0.3f" % (epoch, 
                    step, (dt_size - 1) // dataload.batch_size + 1, 
                    loss.item()))
        print("epoch %d loss:%0.3f" % (num_epochs, epoch_loss/step))
    torch.save(model.state_dict(),
            'weights_%d_%s.pth' % (num_epochs, model.name))
    return model


def train3d(num_classes, batch_size, num_epochs, workspace="./train3d", device='cuda', X_transform=None, Y_transform=None):
    '''
    @construct dataloader, criterion, optimizer, construct and train a 3d model
    @input
    num_classes
    batch_size
    workspace
    num_epochs
    device
    X_transform
    Y_transform
    '''
    model = Unet3d(1, num_classes).to(device)
    criterion = CrossEntropyDiceLoss(num_of_classes=num_classes)
    optimizer = optim.Adam(model.parameters())
    dataset = Dataset3d(workspace, transform=X_transform, 
                        target_transform=Y_transform)
    dataloaders = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    train_model(model, criterion, optimizer, dataloaders, num_epochs, device, False)

def train2d(num_classes, batch_size, num_epochs, workspace="./train2d", device='cuda', X_transform=None, Y_transform=None):
    '''
    @construct dataloader, criterion, optimizer, construct and train a 2d model
    @input
    num_classes
    batch_size
    workspace
    num_epochs
    device
    X_transform
    Y_transform
    '''
    model = Unet2d(1, num_classes).to(device)
    criterion = CrossEntropyDiceLoss(num_of_classes=num_classes)
    optimizer = optim.Adam(model.parameters())
    dataset = Dataset2d(workspace, transform=X_transform, 
                        target_transform=Y_transform)
    dataloaders = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    train_model(model, criterion, optimizer, dataloaders, num_epochs, device, False)

