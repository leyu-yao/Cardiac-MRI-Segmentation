# import 3rd-party modules
import numpy as np 
import torch
import argparse
from torch.utils.data import DataLoader
from torch import autograd, optim, nn
from torchvision.transforms import transforms
import sys
import os 
import matplotlib.pyplot as plt

# import project modules
from models import SmallUnet3d, Unet3d, Unet2d, SmallSMallUnet3d
from dataset import Dataset3d, Dataset2d
from loss_function import CrossEntropy3d, DiceLoss ,CrossEntropyDiceLoss
from util import fix_unicode_bug
import util
from metrics import Metric_AUC
import transform3d
from timeTick import timeTicker

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
            torch.save(model.state_dict(),
                       'weights_%d_%s.pth' % (num_epochs, model.name))
        print("epoch %d loss:%0.3f" % (num_epochs, epoch_loss/step))

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
    criterion = DiceLoss(num_of_classes=num_classes)
    optimizer = optim.Adam(model.parameters())
    dataset = Dataset3d(workspace, transform=transform)
    dataloaders = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    train_model(model, criterion, optimizer, dataloaders, num_epochs, device, False)
    
def train_roi(batch_size, num_epochs, workspace="./roi_train", device='cuda', transform=None):
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
    model = SmallSMallUnet3d(1, 1).to(device)
    
    #nn.MSELoss()
    #criterion = CrossEntropyDiceLoss(num_of_classes=1)
    criterion = nn.L1Loss()
    
    optimizer = optim.Adam(model.parameters())
    dataset = Dataset3d(workspace, transform=transform)
    dataloaders = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    train_model(model, criterion, optimizer, dataloaders, num_epochs, device, False)


def train2d(num_classes, batch_size, num_epochs, workspace="./train2d", device='cuda', transform=None):
    '''
    @construct dataloader, criterion, optimizer, construct and train a 2d model
    @input
    num_classes
    batch_size
    workspace
    num_epochs
    device
    transform
    '''
    model = Unet2d(1, num_classes).to(device)
    criterion = CrossEntropyDiceLoss(num_of_classes=num_classes)
    optimizer = optim.Adam(model.parameters())
    dataset = Dataset2d(workspace, transform=transform)
    dataloaders = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    train_model(model, criterion, optimizer, dataloaders, num_epochs, device, False)


def eval2d(num_classes, ckp, metrics, device='cuda;', workspace="./eval2d", transform=None, vis=False):
    '''
    evaluation on eval-test
    metrics is required has method __call__(y_pred_tensor, y_true_tensor)
    ckp is path for weights of the model

    '''
    model = Unet2d(1, num_classes).to(device)
    model.load_state_dict(torch.load(ckp, map_location=device))
    dataset = Dataset2d(workspace, transform=transform)
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
                plt.ion()
                for _ in range(5):
                
                    plt.subplot(5,3,1+3*_)
                    plt.title('class%d'%(_))
                    img_y=torch.squeeze(outputs.cpu()).numpy()[_]
                    plt.imshow(img_y)
                    
                    plt.subplot(5,3,2+3*_)
                    plt.title('gt')
                    img_y=torch.squeeze(y).numpy()[_]
                    plt.imshow(img_y)
                    
                plt.subplot(5,3,3)
                plt.title('input')
                img_y=torch.squeeze(x).numpy()
                plt.imshow(img_y)

                plt.show()
                plt.pause(0.5)


        print("average score on evaluation set is %0.3f" % (average_score/dt_size))


def eval3d(num_classes, ckp, metrics, device='cuda;', workspace="./eval3d", transform=None, vis=False):
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



def eval_roi(ckp, metrics, device='cuda;', workspace="./roi_eval", transform=None, vis=True):
    '''
    evaluation on eval-test
    metrics is required has method __call__(y_pred_tensor, y_true_tensor)
    ckp is path for weights of the model

    '''
    model = SmallSMallUnet3d(1, 1).to(device)
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
            average_score += score[0]
            print("score = %0.3f" % score[0])

            if vis:
                id = str(hash(x))
                outputs[outputs>=0.5] = 1
                outputs[outputs<0.5] = 0
                fn = os.path.join(workspace, "pred_%s_label.nii.gz"%id)
                util.tensor_or_arr_write_to_nii_file(labels, fn, affine=None)
                
                fn = os.path.join(workspace, "pred_%s_outputs.nii.gz"%id)
                util.tensor_or_arr_write_to_nii_file(outputs, fn, affine=None)

                fn = os.path.join(workspace, "pred_%s_inputs.nii.gz"%id)
                util.tensor_or_arr_write_to_nii_file(x, fn, affine=None)

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
    parse.add_argument("--num_classes", type=int, default=5)
    parse.add_argument("--workspace", type=str)

    args = parse.parse_args()

    if args.action == "train2d":
        train2d(args.num_classes, args.batch_size, args.num_epochs, args.workspace, device=args.device)
    elif args.action == "train3d":
        tran = transform3d.RandomTransformer(transform3d.Transpose(), transform3d.DummyTransform())
        train3d(args.num_classes, args.batch_size, args.num_epochs, args.workspace, device=args.device, transform=tran)
    elif args.action == "eval2d":
        metric = Metric_AUC()
        eval2d(args.num_classes, args.ckp, metric, args.device, args.workspace)
    elif args.action == "eval3d":
        metric = Metric_AUC()
        eval3d(args.num_classes, args.ckp, metric, args.device, args.workspace)       
    elif args.action == "train_roi":
        train_roi(args.batch_size, args.num_epochs, args.workspace, args.device, transform=None)
    elif args.action == "eval_roi":
        metric = Metric_AUC()
        eval_roi(args.ckp, metric, args.device, args.workspace)           