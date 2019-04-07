import numpy as np
import torch
import argparse
from torch.utils.data import DataLoader
from torch import autograd, optim
from torchvision.transforms import transforms
from models import SmallUnet3d, Unet3d, Unet2d
from dataset import Dataset3d, Dataset2d
import sys
import matplotlib.pyplot as plt
from loss_function import CrossEntropy3d, DiceLoss ,CrossEntropyDiceLoss
#import visdom
import win_unicode_console
win_unicode_console.enable()
#vis = visdom.Visom(env='model_1')

# 是否使用cuda
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")

x_transforms = transforms.ToTensor()
y_transforms = transforms.ToTensor()



def train_model(model, criterion, optimizer, dataload, num_epochs=5):
    # Data paralize
#    if torch.cuda.device_count() > 1:
#      print("Let's use", torch.cuda.device_count(), "GPUs!")
#      model = nn.DataParallel(model)
  
    
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
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
            print("In epoch %d, %d/%d,train_loss:%0.3f" % (epoch+1, 
                    step, (dt_size - 1) // dataload.batch_size + 1, 
                    loss.item()))
        print("epoch %d loss:%0.3f" % (num_epochs, epoch_loss/dt_size))
    torch.save(model.state_dict(), 'weights_%d_%s.pth' % (num_epochs, model.name))
    return model

#训练模型
def train3d():
    model = Unet3d(1, 5).to(device)
    batch_size = args.batch_size
    # reweight
#    weights = np.ones((2,64,64,32),dtype=np.float32)
#    weights[1,:,:,:] *= 1000
#    weights = torch.from_numpy(weights)
    
    #criterion = torch.nn.BCELoss(weight=weights.to(device))
    # criterion = DICELoss()
    criterion = CrossEntropyDiceLoss(num_of_classes=5)
    optimizer = optim.Adam(model.parameters())
    dataset = Dataset3d("./train3d",transform=None,target_transform=None)
    dataloaders = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    train_model(model, criterion, optimizer, dataloaders, num_epochs=50)
    
def train2d():
    model = Unet2d(1, 5).to(device)
    batch_size = args.batch_size
    # reweight
#    weights = np.ones((2,64,64,32),dtype=np.float32)
#    weights[1,:,:,:] *= 1000
#    weights = torch.from_numpy(weights)
    
    #criterion = torch.nn.BCELoss(weight=weights.to(device))
    # criterion = DICELoss()
    #criterion = DiceLoss(num_of_classes=5, 
    #            weights=torch.tensor([0.5, 1., 1., 1., 1.]))
    criterion = CrossEntropyDiceLoss(num_of_classes=5,
                    weights_for_class=torch.tensor([1, 1., 2., 2., 2.]).to(device))
    
    optimizer = optim.Adam(model.parameters())
    dataset = Dataset2d("./train2d",transform=None,target_transform=None)
    dataloaders = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    train_model(model, criterion, optimizer, dataloaders, num_epochs=50)

#显示模型的输出结果
def test2d():
    model = Unet2d(1, 5).to(device)
    model.load_state_dict(torch.load(args.ckp,map_location='cuda'))
    dataset = Dataset2d("./eval2d", transform=None,target_transform=None)
    dataloaders = DataLoader(dataset, batch_size=1)
    model.eval()
    
    plt.ion()
    
    dt_size = len(dataloaders.dataset)
    average_loss = 0
    
    with torch.no_grad():
        
        average_loss = 0
        
        for x, y in dataloaders:
            outputs=model(x.to(device))
            labels = y.to(device)
            #criterion = DiceLoss(num_of_classes=5, 
            #    weights=torch.tensor([0.5, 1., 1., 1., 1.]))
            #criterion = CrossEntropy3d()
            criterion = CrossEntropyDiceLoss(num_of_classes=5,
                    weights_for_class=torch.tensor([1, 1., 2., 2., 2.]).to(device))
            loss = criterion(outputs, labels)
            average_loss += loss.item()
            print("image loss = %0.3f" % loss.item())
            
            # visualize
            # plt.ion()
            # for _ in range(5):
            
            #     plt.subplot(5,3,1+3*_)
            #     plt.title('class%d'%(_))
            #     img_y=torch.squeeze(outputs.cpu()).numpy()[_]
            #     plt.imshow(img_y)
                
            #     plt.subplot(5,3,2+3*_)
            #     plt.title('gt')
            #     img_y=torch.squeeze(y).numpy()[_]
            #     plt.imshow(img_y)
                
            # plt.subplot(5,3,3)
            # plt.title('input')
            # img_y=torch.squeeze(x).numpy()
            # plt.imshow(img_y)

            print(outputs)
            plt.show()
            plt.pause(0.5)
        
        print("average loss on test set is %0.3f" % (average_loss/dt_size))

'''
input should be cpu Tensor
shape= (N,C,D,H,W)
'''
def plot_3d_colorful(outputs, labels, inputs):
    (N,C,D,H,W) = labels.shape
    import matplotlib.pyplot as plt
    for w in range(0,W,8):
        plt.subplot(W/8,3,w/8*3+1)
        plt.title("output")
        img_y=outputs.numpy()[0,2,:,:,w]
        plt.imshow(img_y)
        
        plt.subplot(W/8,3,w/8*3+2)
        plt.title("ground truth")
        img_y=labels.numpy()[0,2,:,:,w]        
        plt.imshow(img_y)
        
        plt.subplot(W/8,3,w/8*3+3)
        plt.title("input")
        img_y=inputs.numpy()[0,0,:,:,w]        
        plt.imshow(img_y)
    plt.pause(0.01)
    plt.show()
    
def test3d():
    model = Unet3d(1, 5).to(device)
    model.load_state_dict(torch.load(args.ckp,map_location='cuda'))
    dataset = Dataset2d("./eval3d", transform=None,target_transform=None)
    dataloaders = DataLoader(dataset, batch_size=1)
    model.eval()
    import matplotlib.pyplot as plt
    plt.ion()
    
    dt_size = len(dataloaders.dataset)
    average_loss = 0
    
    with torch.no_grad():
        
        average_loss = 0
        
        for x, y in dataloaders:
            outputs=model(x.to(device))
            labels = y.to(device)
            criterion = CrossEntropyDiceLoss(num_of_classes=5)
            loss = criterion(outputs, labels)
            average_loss += loss.item()
            print(loss.item())
            plot_3d_colorful(outputs.cpu(), y, x)
        
        print("average loss on test set is %0.3f" % (average_loss/dt_size))

# Visualize
def visualize():
    from tensorboardX import SummaryWriter
    from torch.autograd import Variable
    model = Unet(1, 8)
    input_data = Variable(torch.rand(1,1, 64, 64, 64))
    #print(input_data)
    writer = SummaryWriter(log_dir='./log', comment='unet')
    with writer:
        writer.add_graph(model, input_data)
    
    
if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument("action", type=str, help="train or test")
    parse.add_argument("--batch_size", type=int, default=1)
    parse.add_argument("--ckp", type=str, help="the path of model weight file")
    parse.add_argument("--epoch", type=int, default=5)
    args = parse.parse_args()

    if args.action=="train3d":
        train3d()
    elif args.action=="test2d":
        test2d()
    elif args.action=="visualize":
        visualize()
    elif args.action=="train2d":
        train2d()
    elif args.action=="test3d":
        test3d()
