import torch
import numpy as np 
import argparse
from torch.utils.data import DataLoader
from torch import autograd, optim, nn
import sys
import os

import matplotlib.pyplot as plt

# import libs
import dataset
import models
import losses
import metrics
import transforms
import util

#util.fix_unicode_bug()

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
        for x, y, _, __ in dataload:
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
            torch.save(model.state_dict(),
                       'weights_%d_%s.pth' % (num_epochs, model.name))
        print("epoch %d loss:%0.3f" % (num_epochs, epoch_loss/step))

    return model

def train(num_classes, batch_size, num_epochs, workspace="./raw", device='cuda', transform=None, ckp=None):
    '''
    @construct dataloader, criterion, optimizer, construct and train a 3d model
    @input
    num_classes
    batch_size
    workspace
    num_epochs
    device
    transform
    '''
    model = models.Unet3d(1, num_classes).to(device)
    # load weights
    if ckp is not None:
        model.load_state_dict(torch.load(ckp, map_location=device))

    criterion = losses.DiceLoss(num_of_classes=num_classes, device=device)
    optimizer = optim.Adam(model.parameters())
    ds = dataset.Dataset(workspace, transform=transform, num_classes=num_classes)
    dataloaders = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0)
    train_model(model, criterion, optimizer, dataloaders, num_epochs, device, False)


def test(num_classes, ckp, metrics, device='cuda;', workspace="./test", transform=None, vis=False):
    '''
    evaluation on eval-test
    metrics is required has method __call__(y_pred_tensor, y_true_tensor)
    ckp is path for weights of the model
    '''
    model = models.Unet3d(1, num_classes).to(device)
    model.load_state_dict(torch.load(ckp, map_location=device))
    ds = dataset.Dataset(workspace, transform=transform, num_classes=num_classes)
    dataloaders = DataLoader(ds, batch_size=1)
    model.eval()

    dt_size = len(dataloaders.dataset)
    average_score = 0

    with torch.no_grad():
        average_score = 0
        for x,y,x_path,affine in dataloaders:
            outputs = model(x.to(device))
            labels = y.to(device)
            score = metrics(outputs, labels)
            average_score += score
            print("score = %0.3f" % score)

            # write file
            util.write_nii(outputs, x_path[0].replace('image','output_low'), affine[0,:,:].cpu().numpy())
            util.write_nii(x, x_path[0].replace('image','input_low'), affine[0,:,:].cpu().numpy())

        print("average score on evaluation set is %0.3f" % (average_score/dt_size))


if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    parse.add_argument("action", type=str, help="train or test")
    parse.add_argument("--batch_size", type=int, default=1)
    parse.add_argument("--ckp", type=str, default=None)
    parse.add_argument("--num_epochs", type=int, default=5)
    parse.add_argument("--device", type=str, default="cuda")
    parse.add_argument("--para", type=bool, default=False)
    parse.add_argument("--num_classes", type=int, default=5)
    parse.add_argument("--workspace", type=str)
    parse.add_argument("--cuda_index", type=str, default='0')
    parse.add_argument("--resolution", nargs='+', type=int, default=(160,160,114))
    args = parse.parse_args()



    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_index
    #os.environ["CUDA_VISIBLE_DEVICES"] = '0'

    if args.action == "train":
        #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        device = torch.device(args.device)
        tran = transforms.SmartDownSample((args.resolution))
        train(args.num_classes, args.batch_size, args.num_epochs, args.workspace, device=device, transform=tran, ckp=args.ckp)

    elif args.action == "test":
        device = torch.device(args.device)
        tran = transforms.SmartDownSample((args.resolution))
        metric = metrics.Metric_AUC()
        test(args.num_classes, args.ckp, metric, device=device, workspace="./test", transform=tran)
