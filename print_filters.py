import argparse
import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from torchvision.datasets import MNIST
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
import torch.utils.data as data
from PIL import Image
import numpy as np
from torchvision.utils import save_image
import torch
import cv2

parser = argparse.ArgumentParser(description='PyTorch_Siamese_lfw')
parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')
parser.add_argument('--epochs', default=2, type=int, metavar='N',
                    help='number of total epochs to run(default: 1)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=1, type=int,
                    metavar='N', help='batch size (default: 8)')
parser.add_argument('--learning_rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate (default: 0.01)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--lfw_path', default='../lfw', type=str, metavar='PATH',
                    help='path to root path of lfw dataset (default: ../lfw)')
parser.add_argument('--train_list', default='../data/train.txt', type=str, metavar='PATH',
                    help='path to training list (default: ../data/train.txt)')
parser.add_argument('--test_list', default='../data/test.txt', type=str, metavar='PATH',
                    help='path to validation list (default: ../data/test.txt)')
parser.add_argument('--save_path', default='../data/', type=str, metavar='PATH',
                    help='path to save checkpoint (default: ../data/)')
parser.add_argument('--aug', default='off', type=str,
                    help='turn on img augmentation (default: False)')
parser.add_argument('--cuda', default="off", type=str, 
                    help='switch on/off cuda option (default: off)')

parser.add_argument('--load', default='default', type=str,
                    help='turn on img augmentation (default: default)')
parser.add_argument('--save', default='default', type=str,
                    help='turn on img augmentation (default: default)')
parser.add_argument('--model', default='autoencoder', type=str,
                    help='model name')


args = parser.parse_args()

# location = '/home/jeyamariajose/Baselines/pytorch-beginner/08-AutoEncoder/sample/*.jpg'


# def train_loader(path):
#     img = Image.open(path)
#     if args.aug != "off":
#         pix = np.array(img)
#         pix_aug = img_augmentation(pix)
#         img = Image.fromarray(np.uint8(pix_aug))
#     # print pix
#     return img

# def default_list_reader(fileList):
#     imgList = []
#     with open(fileList, 'r') as file:
#         for line in file.readlines():
#             imgshortList = []
#             imgPath1, imgPath2, label = line.strip().split(' ')
            
#             imgshortList.append(imgPath1)
#             imgshortList.append(imgPath2)
#             imgshortList.append(label)
#             imgList.append(imgshortList)
#     return imgList

# class train_ImageList(data.Dataset):
    
#     def __init__(self, fileList, transform=None, list_reader=default_list_reader, train_loader=train_loader):
#         # self.root      = root
#         self.imgList   = list_reader(fileList)
#         self.transform = transform
#         self.train_loader = train_loader

#     def __getitem__(self, index):
#         final = []
#         [imgPath1, imgPath2, target] = self.imgList[index]
#         img1 = self.train_loader(os.path.join(args.lfw_path, imgPath1))
#         img2 = self.train_loader(os.path.join(args.lfw_path, imgPath2))

#         # 
#         # img2 = self.img_augmentation(img2)
#         if self.transform is not None:
#             img1 = self.transform(img1)
#             img2 = self.transform(img2)
#         return img1, img2, torch.from_numpy(np.array([target],dtype=np.float32))

#     def __len__(self):
#         return len(self.imgList)
        
# dataloader = torch.utils.data.DataLoader(
#                     train_ImageList(fileList=args.train_list, 
#                             transform=transforms.Compose([ 
#                             transforms.Scale((28,28)),
#                             transforms.ToTensor(),            ])),
#                     shuffle=False,
#                     num_workers=args.workers,
#                     batch_size=args.batch_size)

class kunetv2_morec(nn.Module):
    
    def __init__(self):
        super(kunetv2_morec, self).__init__()
        
        self.encoder1 = nn.Conv2d(3, 32, 3, stride=1, padding=1)  # b, 16, 10, 10
        self.bne1 = nn.InstanceNorm2d(32)
        self.encoder2=   nn.Conv2d(32, 64, 3, stride=1, padding=1)  # b, 8, 3, 3
        self.bne2 = nn.InstanceNorm2d(64)
        self.encoder3=   nn.Conv2d(64, 128, 3, stride=1, padding=1)
        self.bne3 = nn.InstanceNorm2d(128)
        self.encoder4=   nn.Conv2d(128, 512, 3, stride=1, padding=1)
        self.encoder5=   nn.Conv2d(512, 1024, 3, stride=1, padding=1)



        self.decoder1 =   nn.Conv2d(1024, 512, 3, stride=1, padding=1)  # b, 1, 28, 28
        self.bnd1 = nn.InstanceNorm2d(64)
        self.decoder2 =   nn.Conv2d(512, 128, 3, stride=1, padding=1)
        self.bnd2 = nn.InstanceNorm2d(32)
        self.decoder3 =   nn.Conv2d(128, 64, 3, stride=1, padding=1)
        self.bnd3 = nn.InstanceNorm2d(16)
        self.decoder4 =   nn.Conv2d(64, 32, 3, stride=1, padding=1)
        self.decoder5 =   nn.Conv2d(32, 16, 3, stride=1, padding=1)

        # self.decoderf1 =   nn.Conv2d(16, 128, 2, stride=1, padding=1)
        self.decoderf1 =   nn.Conv2d(128, 64, 3, stride=1, padding=1)
        self.bndf1 = nn.InstanceNorm2d(64)
        self.decoderf2=   nn.Conv2d(64, 32, 3, stride=1, padding=1)
        self.bndf2 = nn.InstanceNorm2d(32)
        self.decoderf3 =   nn.Conv2d(32, 16, 3, stride=1, padding=1)
        self.bndf3 = nn.InstanceNorm2d(16)
        self.decoderf5 =   nn.Conv2d(16, 16, 3, stride=1, padding=1)
        # self.decoderf5 =   nn.Conv2d(16, 16, 3, stride=1, padding=1)

        self.encoderf1 =   nn.Conv2d(3, 32, 3, stride=1, padding=1)
        self.bnef1 = nn.InstanceNorm2d(32)
        self.encoderf2=   nn.Conv2d(32, 64, 3, stride=1, padding=1)
        self.bnef2 = nn.InstanceNorm2d(64)
        self.encoderf3 =   nn.Conv2d(64, 128, 3, stride=1, padding=1)
        self.bnef3 = nn.InstanceNorm2d(128)
        self.encoderf4 =   nn.Conv2d(128, 16, 3, stride=1, padding=1)
        # self.encoderf5 =   nn.Conv2d(128, 128, 3, stride=1, padding=1)
        
        self.final = nn.Conv2d(16,3,1,stride=1,padding=0)
        self.bnf = nn.InstanceNorm2d(3)

        self.tmp1 = nn.Conv2d(16,32,1,stride=1,padding=0)
        self.bnt1 = nn.InstanceNorm2d(32)
        self.tmp2 = nn.Conv2d(32,32,1,stride=1,padding=0)
        # self.bnt2 = nn.BatchNorm2d(32)
        self.tmp3 = nn.Conv2d(64,32,1,stride=1,padding=0)
        self.tmp4 = nn.Conv2d(16,32,1,stride=1,padding=0)


        self.tmpf3 = nn.Conv2d(128,32,1,stride=1,padding=0)
        self.tmpf2 = nn.Conv2d(64,32,1,stride=1,padding=0)
        self.tmpf1 = nn.Conv2d(32,32,1,stride=1,padding=0)
        self.tan = nn.Tanh()
        
        # self.soft = nn.Softmax(dim =1)
    
    def forward(self, x):
        out1 = F.relu(F.interpolate(self.encoderf1(x),scale_factor=(2,2),mode ='bilinear'))
        
        for i in range(out1.shape[1]):

            img = np.asarray(out1[0][i].cpu().detach())
            img *= (255.0/img.max())
            
            cv2.imwrite("results3/knetlayer1_filter_{}.jpg".format(i),np.asarray(img))

        t1 = F.interpolate(out1,scale_factor=(0.25,0.25),mode ='bilinear')
        
        o1 = out1
        out1 = F.relu(F.interpolate(self.encoderf2(out1),scale_factor=(2,2),mode ='bilinear'))
        for i in range(out1.shape[1]):

            img = np.asarray(out1[0][i].cpu().detach())
            img *= (255.0/img.max())
            
            cv2.imwrite("results3/knetlayer2_filter_{}.jpg".format(i),np.asarray(img))
        t2 = F.interpolate(out1,scale_factor=(0.125,0.125),mode ='bilinear')
        o2 = out1        
        
        out1 = F.relu(F.interpolate(self.encoderf3(out1),scale_factor=(2,2),mode ='bilinear')) 
        for i in range(out1.shape[1]):

            img = np.asarray(out1[0][i].cpu().detach())
            img *= (255.0/img.max())
            
            cv2.imwrite("results3/knetlayer3_filter_{}.jpg".format(i),np.asarray(img))
        t3 = F.interpolate(out1,scale_factor=(0.0625,0.0625),mode ='bilinear')
        
        # U-NET encoder start
        out = F.relu(F.max_pool2d(self.encoder1(x),2,2))
        #Fusing all feature maps from K-NET
        for i in range(out.shape[1]):

            img = np.asarray(out[0][i].cpu().detach())
            img *= (255.0/img.max())
            
            cv2.imwrite("results3/unetlayer1_filter_{}.jpg".format(i),np.asarray(img))

        out = torch.add(out,torch.add(self.tmpf3(t3),torch.add(t1,self.tmpf2(t2))))
        
        u1 = out
        out = F.relu(F.max_pool2d(self.encoder2(out),2,2))
        for i in range(out.shape[1]):

            img = np.asarray(out[0][i].cpu().detach())
            img *= (255.0/img.max())
            
            cv2.imwrite("results3/unetlayer2_filter_{}.jpg".format(i),np.asarray(img))
        u2 = out
        out = F.relu(F.max_pool2d(self.encoder3(out),2,2))
        for i in range(out.shape[1]):

            img = np.asarray(out[0][i].cpu().detach())
            img *= (255.0/img.max())
            
            cv2.imwrite("results3/unetlayer3_filter_{}.jpg".format(i),np.asarray(img))
        u3=out
        out = F.relu(F.max_pool2d(self.encoder4(out),2,2))
        for i in range(out.shape[1]):

            img = np.asarray(out[0][i].cpu().detach())
            img *= (255.0/img.max())
            
            cv2.imwrite("results3/unetlayer4_filter_{}.jpg".format(i),np.asarray(img))
        u4=out
        out = F.relu(F.max_pool2d(self.encoder5(out),2,2))
        for i in range(out.shape[1]):

            img = np.asarray(out[0][i].cpu().detach())
            img *= (255.0/img.max())
            
            cv2.imwrite("results3/unetlayer5_filter_{}.jpg".format(i),np.asarray(img))
        out = F.relu(F.interpolate(self.decoder1(out),scale_factor=(2,2),mode ='bilinear'))
        out = torch.add(out,u4)
        out = F.relu(F.interpolate(self.decoder2(out),scale_factor=(2,2),mode ='bilinear'))
        out = torch.add(out,u3)
        out = F.relu(F.interpolate(self.decoder3(out),scale_factor=(2,2),mode ='bilinear'))
        out = torch.add(out,u2)
        out = F.relu(F.interpolate(self.decoder4(out),scale_factor=(2,2),mode ='bilinear'))
        out = torch.add(out,u1)
        
        # Start K-Net decoder

        out1 = F.relu(F.max_pool2d(self.decoderf1(out1),2,2))
        out1 = torch.add(out1,o2)
        
        t3 = F.interpolate(out1,scale_factor=(0.125,0.125),mode ='bilinear')

        out1 = F.relu(F.max_pool2d(self.decoderf2(out1),2,2))
        out1 = torch.add(out1,o1)

        t2 = F.interpolate(out1,scale_factor=(0.25,0.25),mode ='bilinear')
        
        out1 = F.relu(F.max_pool2d(self.decoderf3(out1),2,2))

        t1 = F.interpolate(out1,scale_factor=(0.5,0.5),mode ='bilinear')
        # Fusing all layers at the last layer of decoder
        # print(t1.shape,t2.shape,t3.shape,out.shape)
        out = torch.add(out,torch.add(self.tmp3(t3),torch.add(self.tmp1(t1),self.tmp2(t2))))

        out = F.relu(F.interpolate(self.decoder5(out),scale_factor=(2,2),mode ='bilinear'))
        
        out = F.relu(self.final(out))

        return self.tan(out)

device_ids = [Id for Id in range(torch.cuda.device_count())]
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

from derain_mulcmp import unet

model = unet().cuda()

model = model.to(device)
model = nn.DataParallel(model, device_ids=device_ids)
model.load_state_dict(torch.load("/media/jeyamariajose/7888230b-5c10-4229-90f2-c78bdae9c5de/Data/Projects/Derain_knet_ECCV20/unet_ablation/derain_v1/indoor/indoor_Derain_best_168_22"))
model.eval()

val_data_dir = './data/test/SOTS/indoor/'
val_batch_size = 1
from val_data import ValData
val_data_loader = DataLoader(ValData(val_data_dir), batch_size=val_batch_size, shuffle=False, num_workers=24)

c =1

for batch_id, val_data in enumerate(val_data_loader):

        with torch.no_grad():
            haze, gt, image_name = val_data            
            haze = haze.to(device)
            gt = gt.to(device)
    
            dehaze = model(haze)

            for i in range(dehaze.shape[1]):

                img = np.asarray(dehaze[0][i].cpu().detach())
                img *= (255.0/img.max())
                cv2.imwrite("op{}.png".format(i),np.asarray(img))
            break

            
            # print(len(data))
            # print(len(img))
            # print(img.shape)
            # img = torch.Tensor(img).cuda()
            # print(img.shape)
            # # ===================forward=====================
            # output = model(img)
            print("done")
            
            # c=c+1
            # if c==4:
            #     break