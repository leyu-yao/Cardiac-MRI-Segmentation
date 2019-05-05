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
from models import Unet3d
from dataset import Dataset3d
import losses

import util
import transforms

from timeTick import timeTicker


#fix_unicode_bug()

# %%train model
def train_model(model, criterion, optimizer, dataload, num_epochs, device, parallel, weight_name=None):
    '''
    train procedure
    '''
    # DataParallel
    if torch.cuda.device_count() > 1 and parallel:
        print("Using", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
        
    # time estimation
    total_work_laod = num_epochs * len(dataload.dataset)
    time_start = time.time()
    
    
    for epoch in range(1, num_epochs+1):
        #print('Epoch {}/{}'.format(epoch, num_epochs))
        #print('-' * 10)
        dt_size = len(dataload.dataset)
        epoch_loss = 0
        step = 0
        for x, y in dataload:
            
            inputs = x.to(device)
            labels = y.to(device)
            # zero the parameter gradients
            if step % 8 == 0:
                optimizer.zero_grad()
            # forward
            outputs = model(inputs)
            
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
            time_now = time.time()
            t_passed = time_now - time_start
            work_load = step+1 + dt_size * (epoch - 1)
            sys.stdout.write("\rIn epoch %d, %d/%d,train_loss:%0.6f, passed :%.1fs, estimated %.1fs" % (epoch, 
                    step, (dt_size - 1) // dataload.batch_size + 1, 
                    loss.item(), t_passed,
                    t_passed / work_load * (total_work_laod - work_load)))
            
            step += 1
            
            
        torch.save(model.state_dict(),
               'weights_%d_%s.pth' % (num_epochs, model.name if weight_name is None else weight_name))
        print(" loss:%0.6f" % (epoch_loss/step))

    return model



def train3d(num_classes, batch_size, num_epochs, workspace="./train3d", device='cuda', transform=None):
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
    model = Unet3d(1, num_classes).to(device)
    criterion = losses.DiceLoss()
    
    # %% use different lr
    conv2_conv2_params = list(map(id, model.conv2.conv2.parameters()))
    conv3_conv2_params = list(map(id, model.conv3.conv2.parameters()))
    conv4_conv1_params = list(map(id, model.conv4.conv1.parameters()))
    conv4_conv2_params = list(map(id, model.conv4.conv2.parameters()))
    
    union = conv2_conv2_params+conv3_conv2_params+conv4_conv1_params+conv4_conv2_params
    
    base_params = filter(lambda p: id(p) not in union, model.parameters())
    
    optimizer = torch.optim.SGD([
                {'params': base_params},
                {'params': model.conv2.conv2.parameters(), 'lr': 1e-6},
                {'params': model.conv3.conv2.parameters(), 'lr': 1e-5},
                {'params': model.conv4.conv1.parameters(), 'lr': 1e-5},
                {'params': model.conv4.conv2.parameters(), 'lr': 1e-4}],
                lr=1e-3, momentum=0.9
                )

    
    
    
    # %% origin
    
    # optimizer = optim.Adam(model.parameters())
    
    # %%
    
    dataset = Dataset3d(workspace, transform=transform)
    dataloaders = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    train_model(model, criterion, optimizer, dataloaders, num_epochs, device, False)


def eval3d(num_classes, ckp, metrics, device='cuda', workspace="./eval3d", transform=None, vis=False):
    '''
    evaluation on eval-test
    metrics is required has method __call__(y_pred_tensor, y_true_tensor)
    ckp is path for weights of the model
    '''
    model = Unet3d(1, num_classes).to(device)
    model.load_state_dict(torch.load(ckp, map_location=device))
    dataset = Dataset3d(workspace, transform=transform)
    dataloaders = DataLoader(dataset, batch_size=1)
    model.eval()

    dt_size = len(dataloaders.dataset)
    average_score = 0

    with torch.no_grad():
        average_score = 0
        for x,y in dataloaders:
            outputs = model(x.to(device))
            labels = y.to(device)
            score = metrics(outputs, labels)
            average_score += score
            print("score = %0.3f" % score)

            if vis:
                pass
            



        print("average score on evaluation set is %0.3f" % (average_score/dt_size))




if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    parse.add_argument("action", type=str, help="train or test")
    parse.add_argument("--batch_size", type=int, default=1)
    parse.add_argument("--ckp", type=str, help="the path of model weight file")
    parse.add_argument("--num_epochs", type=int, default=5)
    parse.add_argument("--device", type=str, default="cuda")
    parse.add_argument("--para", type=bool, default=False)
    parse.add_argument("--num_classes", type=int, default=8)
    parse.add_argument("--workspace", type=str)
    parse.add_argument("--weight_name", type=str, default=None)
    parse.add_argument("--vis", type=bool, default=False)
    parse.add_argument("--output_dir", type=str, default=None)
    
    args = parse.parse_args()

    if args.action == "train3d":
        tran = transforms.Normalization()
        #tran = transform3d.RandomTransformer(transform3d.Transpose(), transform3d.DummyTransform())
        train3d(args.num_classes, args.batch_size, args.num_epochs, args.workspace, device=args.device, transform=None)
    
    elif args.action == "eval3d":
        #metric = Metric_AUC()
        eval3d(args.num_classes, args.ckp, metric, args.device, args.workspace)       
