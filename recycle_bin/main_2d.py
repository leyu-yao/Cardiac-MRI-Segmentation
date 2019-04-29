import torch 
import torch.nn as nn 
from torch import autograd, optim
import torch.utils.data as data
from torch.utils.data import DataLoader
import os
import sys
import PIL.Image as Image 
import random
from torchvision.transforms import transforms
import argparse
import time
import numpy as np

import util

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        return self.conv(input)


class Unet(nn.Module):
    def __init__(self,in_ch,out_ch):
        super(Unet, self).__init__()
        self.name = "Unet2d"
        self.conv1 = DoubleConv(in_ch, 32)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = DoubleConv(32, 64)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = DoubleConv(64, 128)
        self.pool3 = nn.MaxPool2d(2)
        self.conv4 = DoubleConv(128, 256)
        self.pool4 = nn.MaxPool2d(2)
        self.conv5 = DoubleConv(256, 512)
        self.up6 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv6 = DoubleConv(512, 256)
        self.up7 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv7 = DoubleConv(256, 128)
        self.up8 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv8 = DoubleConv(128, 64)
        self.up9 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.conv9 = DoubleConv(64, 32)
        self.conv10 = nn.Conv2d(32,out_ch, 1)

    def forward(self,x):
        c1=self.conv1(x)
        p1=self.pool1(c1)
        c2=self.conv2(p1)
        p2=self.pool2(c2)
        c3=self.conv3(p2)
        p3=self.pool3(c3)
        c4=self.conv4(p3)
        p4=self.pool4(c4)
        c5=self.conv5(p4)
        up_6= self.up6(c5)
        merge6 = torch.cat([up_6, c4], dim=1)
        c6=self.conv6(merge6)
        up_7=self.up7(c6)
        merge7 = torch.cat([up_7, c3], dim=1)
        c7=self.conv7(merge7)
        up_8=self.up8(c7)
        merge8 = torch.cat([up_8, c2], dim=1)
        c8=self.conv8(merge8)
        up_9=self.up9(c8)
        merge9=torch.cat([up_9,c1],dim=1)
        c9=self.conv9(merge9)
        c10=self.conv10(c9)
        out = nn.Softmax(dim=1)(c10)
        return out


class Dataset2d(data.Dataset):
    def __init__(self, workspace, transform=None):
        self.imgs = util.make_dataset(workspace, label_elem='image')
        self.transform = transform

    def __getitem__(self, index):
        x_path, y_path = self.imgs[index]
        img_x = Image.open(x_path)
        img_y = Image.open(y_path)

        if self.transform is not None:
            img_x, img_y = self.transform(img_x, img_y)

        return img_x, img_y

    def __len__(self):
        return len(self.imgs)

class XYToTensor(object):
    def __init__(self):
        self.X_tran =  transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])
        

        self.Y_tran = transforms.ToTensor()
    
    def __call__(self, x, y):
        
        def y_tran(y):
            return torch.from_numpy(np.array(y)[:,:].astype(np.int32))
        return self.X_tran(x), y_tran(y).long()



def train_model(model, criterion, optimizer, dataload, num_epochs, device, parallel, weight_name):
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
            
            time_now = time.time()
            t_passed = time_now - time_start
            work_load = step + dt_size * (epoch - 1)
            sys.stdout.write("\rIn epoch %d, %d/%d,train_loss:%0.3f, passed :%.3fs, estimated %.3fs" % (epoch, 
                    step, (dt_size - 1) // dataload.batch_size + 1, 
                    loss.item(), t_passed,
                    t_passed / work_load * (total_work_laod - work_load)))
        torch.save(model.state_dict(),
               'weights_%d_%s.pth' % (num_epochs, model.name if weight_name is None else weight_name))
        print(" loss:%0.3f" % (epoch_loss/step))

    return model

def train2d(num_classes, batch_size, num_epochs, workspace="./raw", device='cuda', transform=None, ckp=None, weight_name=None):
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
    model = Unet(1, num_classes).to(device)
    # load weights
    if ckp is not None:
        model.load_state_dict(torch.load(ckp, map_location=device))

    criterion = nn.NLLLoss().to(device)
    #optimizer = optim.SGD(model.parameters(), lr=1e-3, weight_decay=1e-4)
    optimizer = optim.Adam(model.parameters(), weight_decay=1e-4)
    ds = Dataset2d(workspace, transform=transform)
    dataloaders = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    train_model(model, criterion, optimizer, dataloaders, num_epochs, device, False, weight_name)




if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    parse.add_argument("action", type=str, help="train or test")
    parse.add_argument("--batch_size", type=int, default=1)
    parse.add_argument("--ckp", type=str, default=None)
    parse.add_argument("--num_epochs", type=int, default=8)
    parse.add_argument("--device", type=str, default="cuda")
    parse.add_argument("--para", type=bool, default=False)
    parse.add_argument("--num_classes", type=int, default=5)
    parse.add_argument("--workspace", type=str)
    parse.add_argument("--cuda_index", type=str, default='0')
    parse.add_argument("--resolution", nargs='+', type=int, default=(160,160,114))
    parse.add_argument("--weight_name", type=str, default=None)
    args = parse.parse_args()



    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_index
    #os.environ["CUDA_VISIBLE_DEVICES"] = '0'

    if args.action == "train":
        #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        device = torch.device(args.device)
        tran = XYToTensor()
        train2d(args.num_classes, args.batch_size, args.num_epochs, 
              args.workspace, device=device, transform=tran, 
              ckp=args.ckp, weight_name=args.weight_name)

    elif args.action == "test":
        device = torch.device(args.device)
        tran = XYToTensor()
        #metric = metrics.Metric_AUC()
        test(args.num_classes, args.ckp, metric, device=device, workspace="./test", transform=tran)