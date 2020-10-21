import time
import torch
import argparse
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from train_data import TrainData
from val_data import ValData

from utils import to_psnr, print_log, validation, adjust_learning_rate
from torchvision.models import vgg16
from perceptual import LossNetwork
import numpy as np
import random
import os
import time
plt.switch_backend('agg')
import pdb
import math

# --- Parse hyper-parameters  --- #
parser = argparse.ArgumentParser(description='Hyper-parameters')
parser.add_argument('-learning_rate', help='Set the learning rate', default=2e-4, type=float)
parser.add_argument('-crop_size', help='Set the crop_size', default=[128, 128], nargs='+', type=int)
parser.add_argument('-train_batch_size', help='Set the training batch size', default=18, type=int)
parser.add_argument('-network_height', help='Set the network height (row)', default=3, type=int)
parser.add_argument('-network_width', help='Set the network width (column)', default=6, type=int)
parser.add_argument('-num_dense_layer', help='Set the number of dense layer in RDB', default=4, type=int)
parser.add_argument('-growth_rate', help='Set the growth rate in RDB', default=16, type=int)
parser.add_argument('-lambda_loss', help='Set the lambda in loss function', default=0.04, type=float)
parser.add_argument('-val_batch_size', help='Set the validation/test batch size', default=1, type=int)
parser.add_argument('-category', help='Set image category (indoor or outdoor?)', default='indoor', type=str)

#my arguments
parser.add_argument('-net', help='Network architecture to use: gride_dehaze, derain_v1, dehaze_v1)', default='gride_dehaze', type=str)
parser.add_argument('-seed', help='set random seed', default=22, type=int)
parser.add_argument('-save_dir', help='directory to save checkpoints)', default='checkpoints', type=str)
parser.add_argument('-num_epochs', help='total num of epochs', default=3000, type=int)

args = parser.parse_args()

#set seed
seed = args.seed
if seed is not None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed) 

#create output directory
output_dir = os.path.join(args.save_dir, args.net, args.category)
if os.path.exists(output_dir) is False:
    os.makedirs(output_dir)

#create lof file name
time_now = time.strftime("%m-%d_%H-%M", time.localtime())
log_file = './training_log/log_{}_{}_{}.txt'.format(args.category, args.net, time_now)

learning_rate = args.learning_rate
crop_size = args.crop_size
train_batch_size = args.train_batch_size
network_height = args.network_height
network_width = args.network_width
num_dense_layer = args.num_dense_layer
growth_rate = args.growth_rate
lambda_loss = args.lambda_loss
val_batch_size = args.val_batch_size
category = args.category

print('--- Hyper-parameters for training ---')
print('learning_rate: {}\ncrop_size: {}\ntrain_batch_size: {}\nval_batch_size: {}\nnetwork_height: {}\nnetwork_width: {}\n'
      'num_dense_layer: {}\ngrowth_rate: {}\nlambda_loss: {}\ncategory: {}'.format(learning_rate, crop_size,
      train_batch_size, val_batch_size, network_height, network_width, num_dense_layer, growth_rate, lambda_loss, category))


with open(log_file, 'a') as f:
    print('learning_rate: {}\ncrop_size: {}\ntrain_batch_size: {}\nval_batch_size: {}\nnetwork_height: {}\nnetwork_width: {}\n'
      'num_dense_layer: {}\ngrowth_rate: {}\nlambda_loss: {}\ncategory: {}'.format(learning_rate, crop_size,
      train_batch_size, val_batch_size, network_height, network_width, num_dense_layer, growth_rate, lambda_loss, category), file=f)


# --- Set category-specific hyper-parameters  --- #
num_epochs = args.num_epochs
if category == 'derain':
    
    train_data_dir = './data/train/derain/'
    val_data_dir = './data/test/SOTS/derain/'
elif category == 'dehaze':
 
    train_data_dir = './data/train/dehaze/'
    val_data_dir = './data/test/SOTS/dehaze/'
else:
    raise Exception('Wrong image category. Set it to derain or dehaze category.')


# --- Gpu device --- #
device_ids = [Id for Id in range(torch.cuda.device_count())]
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# --- Define the network --- #
if args.net == 'OUCD':
    from derain_mulcmp import OUCD
    net = OUCD()
    # pytorch_total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    # print(pytorch_total_params)

    # from derain_mulcmp import UMRL
    # net = UMRL()
elif args.net == 'dehaze_v1' :
    from Res2net_model import DeHaze_v1
    net = DeHaze_v1()


# --- Build optimizer --- #
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)


# --- Multi-GPU --- #
net = net.to(device)
net = nn.DataParallel(net, device_ids=device_ids)


# --- Define the perceptual loss network --- #
vgg_model = vgg16(pretrained=True).features[:16]
vgg_model = vgg_model.to(device)
for param in vgg_model.parameters():
    param.requires_grad = False

loss_network = LossNetwork(vgg_model)
loss_network.eval()


# --- Load the network weight --- #
try:
    net.load_state_dict(torch.load('{}_Derain_best_{}_{}'.format(category, 0,3)))
    print('--- weight loaded ---')
except:
    print('--- no weight loaded ---')


# --- Calculate all trainable parameters in network --- #
pytorch_total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
print("Total_params: {}".format(pytorch_total_params))

spilt2_name = 'rain800.txt'
# --- Load training data and validation/test data --- #
train_data_loader = DataLoader(TrainData(crop_size, train_data_dir,spilt2_name), batch_size=train_batch_size, shuffle=True, num_workers=8)
val_data_loader = DataLoader(ValData(val_data_dir), batch_size=val_batch_size, shuffle=False, num_workers=8)


# --- Previous PSNR and SSIM in testing --- #
old_val_psnr, old_val_ssim = validation(net, val_data_loader, device, category)
print('old_val_psnr: {0:.2f}, old_val_ssim: {1:.4f}'.format(old_val_psnr, old_val_ssim))
start_time = time.time()
for epoch in range(num_epochs):
    psnr_list = []
    
    adjust_learning_rate(optimizer, epoch, category=category)

    for batch_id, train_data in enumerate(train_data_loader):

        haze, gt = train_data
        haze = haze.to(device)
        gt = gt.to(device)

        # --- Zero the parameter gradients --- #
        optimizer.zero_grad()

        # --- Forward + Backward + Optimize --- #
        net.train()
        # print(haze.shape, gt.shape)
        dehaze = net(haze)
        gt_128 = torch.nn.functional.interpolate(gt,scale_factor=4)
        gt_256 = torch.nn.functional.interpolate(gt,scale_factor=2)  

        smooth_loss = F.l1_loss(dehaze, gt) #+ 0.33*F.l1_loss(cl128,gt_128) + 0.67*F.l1_loss(cl256,gt_256)
        perceptual_loss = loss_network(dehaze, gt)
        loss = smooth_loss + lambda_loss*perceptual_loss

        loss.backward()
        optimizer.step()

        # --- To calculate average PSNR --- #
        psnr_list.extend(to_psnr(dehaze, gt))

        if not (batch_id % 100):
            print('Epoch: {0}, Iteration: {1}'.format(epoch, batch_id))

        if ((batch_id+1)%400000==0):
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.5
            print('Learning rate sets to {}.'.format(param_group['lr']))
        if ((batch_id+1)%10000==0):
            val_psnr, val_ssim = validation(net, val_data_loader, device, category)
            one_epoch_time = time.time() - start_time
            train_psnr = sum(psnr_list) / len(psnr_list)
            torch.save(net.state_dict(), output_dir + '/{}_haze_{}_{}'.format(category, network_height, network_width))
            print_log(log_file, epoch+1, num_epochs, one_epoch_time, train_psnr, val_psnr, val_ssim, category)
            start_time = time.time()

            # --- update the network weight --- #
            if val_psnr >= old_val_psnr:
               torch.save(net.state_dict(), output_dir + '/{}_Derain_best_{}_{}'.format(category, epoch, int(val_psnr*10)))
               old_val_psnr = val_psnr

    # --- Calculate the average training PSNR in one epoch --- #
    train_psnr = sum(psnr_list) / len(psnr_list)

    # --- Save the network parameters --- #
    torch.save(net.state_dict(), output_dir + '/{}_haze_{}_{}'.format(category, network_height, network_width))

    # --- Use the evaluation model in testing --- #
    net.eval()

    
    
    # print_log(log_file, epoch+1, num_epochs, one_epoch_time, train_psnr, val_psnr, val_ssim, category)
    if epoch % 1 ==0:
        val_psnr, val_ssim = validation(net, val_data_loader, device, category)
        one_epoch_time = time.time() - start_time
        print_log(log_file, epoch+1, num_epochs, one_epoch_time, train_psnr, val_psnr, val_ssim, category)
        start_time = time.time()

        # --- update the network weight --- #
        if val_psnr >= old_val_psnr:
           torch.save(net.state_dict(), output_dir + '/{}_Derain_best_{}_{}'.format(category, epoch, int(val_psnr)))
           old_val_psnr = val_psnr
