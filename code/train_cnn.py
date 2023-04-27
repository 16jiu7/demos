#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 16:27:15 2023

@author: jiu7
"""
import os, pickle
from skimage.util import img_as_ubyte, img_as_bool, img_as_float32
from skimage.morphology import remove_small_objects
import skimage.io as io
import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from data_handeler import RetinalDataset
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset, DataLoader
from models.GAT import GAT, RetinalGAT
from models.unet_model import UNet
import random
from torch.autograd import Variable
from torchmetrics.classification import BinaryAveragePrecision
import matplotlib.pyplot as plt
from albumentations.augmentations.transforms import ColorJitter
from thop import profile, clever_format
from datetime import datetime
import albumentations as A
from albumentations.augmentations.geometric.rotate import RandomRotate90
from albumentations.augmentations.geometric.transforms import Flip, Affine
from albumentations.augmentations.transforms import ColorJitter
from albumentations.augmentations.geometric.resize import Resize
from models.UNet_Version.models.UNet_3Plus import UNet_3Plus, UNet_3Plus_mini, UNet_3Plus_mini_lame
from models.m2unet import M2UNet
from models.m2unet_3plus import M2UNet_3plus, M2UNet_self, M2UNet_3plus_dw
from torchstat import stat

def setup_random_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True 


test_device = 'cuda'
test_inputs = torch.randn([1,3,544, 544]).to(test_device)
cnn = M2UNet_3plus_dw(in_channels=3, n_classes=1)  
#cnn = M2UNet()

cnn.to(test_device)
flops, params = profile(cnn, inputs = (test_inputs,), verbose=False)
flops, params = clever_format([flops, params])
print(f'the model {cnn.__class__.__name__} has {flops} flops, {params} parameters')
   
start = datetime.now()
test_output = cnn(test_inputs)
end = datetime.now()
print(f'forward pass time on {test_device} {int((end-start).total_seconds()*1000)} ms')
test_device = 'cpu'
start = datetime.now()
test_output = cnn(test_inputs)
end = datetime.now()
print(f'forward pass time on {test_device} {int((end-start).total_seconds()*1000)} ms')

brightness, contrast, saturation, hue = 0.25, 0.25, 0.25, 0.01
color_jitter = ColorJitter(brightness, contrast, saturation, hue, always_apply = False, p = 0)
half_resizer = Resize(256, 256, always_apply = True) # allow cnn running and half the size
ori_resizer = Resize(512, 512, always_apply = True) # only to allow cnn running 
class TrainDataset(Dataset):
    def __init__(self, dataset_name, split, transforms = None, color_jitter = color_jitter):
        
        self.split = split
        self.dataset_name = dataset_name
        self.transforms = transforms
        self.color_jitter = color_jitter
        if split == 'train':
            self.data = RetinalDataset(self.dataset_name, cropped = True).all_training
        elif split == 'val':
            self.data = RetinalDataset(self.dataset_name, cropped = True).all_val
        elif split == 'test':
            self.data = RetinalDataset(self.dataset_name, cropped = True).all_test

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        if self.split == 'train' and self.color_jitter is not None:
            data.ori = self.color_jitter(image = data.ori)['image']  
        resized = ori_resizer(image = data.ori, mask = data.gt)
        ori = ToTensor()(resized['image'])
        gt = ToTensor()(resized['mask'])
        return ori, gt
    
def draw_figs(x, losses, vals):
    
    plt.figure(figsize=(10,5), dpi = 200)
    plt.subplot(211)
    plt.title('loss')
    plt.grid()
    plt.plot(x, losses, c = 'b', marker = '.', linewidth = 1, markersize = 2)
    for epoch, loss in zip(x, losses):
        if epoch % 10 == 0:
            plt.text(epoch, loss, f'{loss:.3f}', fontdict={'fontsize':6})
    

    plt.subplot(212)
    plt.title('val')
    plt.grid()
    plt.plot(x, vals, c = 'r', marker = '.', linewidth = 1, markersize = 2)
    for epoch, val in zip(x, vals):
        if epoch % 10 == 0:
            plt.text(epoch, val, f'{val:.3f}', fontdict={'fontsize':6})

    plt.tight_layout() # otherwise subplots will land on each other
    plt.savefig('losses_val.png')     
# In[]
setup_random_seed(167)    
TRAIN_DATASET = 'DRIVE'
criterion = torch.nn.BCEWithLogitsLoss()
n_epoch = 200
optimizer= torch.optim.Adam(cnn.parameters(), lr = 1e-3, weight_decay = 0)
lr_scheduler = CosineAnnealingLR(optimizer, T_max = n_epoch)
change_lr = False
torch.backends.cudnn.benchmark = True

device = 'cuda'
cnn = cnn.to(device)
cnn.train()
val_criterion = BinaryAveragePrecision().to(device)

train_set = TrainDataset(dataset_name = TRAIN_DATASET, split = 'train')
val_set = TrainDataset(dataset_name = TRAIN_DATASET, split = 'val')   
test_set = TrainDataset(dataset_name = TRAIN_DATASET, split = 'test')   

train_loader = DataLoader(train_set, batch_size = 1, num_workers = 0, shuffle = True)
val_loader = DataLoader(val_set, batch_size = 1, num_workers = 0, shuffle = False)
test_loader = DataLoader(test_set, batch_size = 1, num_workers = 0, shuffle = False)

losses = []
val_aps = []

for epoch in range(n_epoch):
    epoch_loss = 0
    for ori, gt in train_loader:
        ori, gt = ori.to(device), gt.to(device)
        pred = cnn(ori)
        loss = criterion(pred, gt)
        epoch_loss += loss.item()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    
    epoch_mean_loss = epoch_loss / len(train_loader)
    losses.append(epoch_mean_loss)
    print(f'epoch {epoch+1}, loss {epoch_mean_loss:.4f}')
    
    if change_lr:
        lr_scheduler.step()
        print(f"lr: {optimizer.param_groups[0]['lr']:.4e}")

    with torch.no_grad():
        mean_ap = 0
        cnn.eval()
        for idx, (ori, gt) in enumerate(val_loader):
            ori, gt = ori.to(device), gt.to(device)
            pred = torch.sigmoid(cnn(ori))
            ap = val_criterion(pred, gt).cpu().numpy()
            mean_ap += ap
            pred = (pred.detach().cpu().numpy())[0,0]
            io.imsave(f'cnn_val_{idx}.png', img_as_ubyte(pred))
        mean_ap = mean_ap / len(val_loader)
        val_aps.append(mean_ap)
        print(f'val AP: {mean_ap:.4f}')

        if np.argmax(np.array(val_aps)) == epoch:
            print(f'epoch {epoch + 1}, save weights')
            torch.save(cnn.state_dict(), f'cnn.pt')
            
draw_figs([i for i in range(1, n_epoch + 1)], losses, val_aps)     
     
del cnn
# In[]
# test
setup_random_seed(100) # otherwise test results are not consistent
test_device = 'cuda'
cnn = M2UNet_3plus_dw(in_channels=3, n_classes=1)  
# cnn = M2UNet()
cnn.load_state_dict(torch.load('cnn.pt'), strict=False)
cnn.to(test_device)
cnn.eval()

with torch.no_grad():
    mean_ap = 0
    for idx, (ori, gt) in enumerate(test_loader):
        ori, gt = ori.to(device), gt.to(device)
        pred = torch.sigmoid(cnn(ori))
    
        ap = val_criterion(pred, gt).cpu().numpy()
        mean_ap += ap
        outputs = pred.cpu().numpy()
        pred = (pred.detach().cpu().numpy())[0,0]
        io.imsave(f'cnn_test_{idx}.png', img_as_ubyte(pred))
    mean_ap = mean_ap / len(test_loader)
    print(f'test AP: {mean_ap:.4f}')
    
# 0.8583 for [16, 32, 64, 128, 256]
# 0.8505 for [8, 16, 32, 64, 128] 13.53G flops, 424.08K parameters

# 0.8431, 0.8401, 0.8439 for [8, 16, 32, 64, 64] 11.53G flops, 257.81K parameters
# 0.8328, 0.8213, 0.8390 ~0.8310 for [4, 8, 16, 32, 64] 3.68G flops, 106.78K parameters 
# 0.8459, 0.8548, 0.8556 for [8, 16, 32, 64] 7.98G flops, 129.67K parameters 0.8521
# 0.8523, 0.8468, 0.8460 for [8, 16, 32, 32] 6.91G flops, 85.70K parameters 0.8484

# consider using [4, 8, 16, 32, 64] and [8, 16, 32, 32]

# lame model 7.05G flops, 104.33K parameters, [8, 16, 32, 32]
# 0.8295


# M2U-Net 1.55G flops, 549.75K parameters 
# 0.8228, 0.8283, 0.8193 ~0.8235
# M2UNet_3plus 3.86G flops, 607.86K parameters
# 0.8388, 0.8433, 0.8358 ~0.8393
# M2UNet_3plus_dw 2.25G flops, 557.01K parameters
# 0.8287, 0.8348, 0.8276 ~0.8304
# the model M2UNet_3plus_dw has 2.28G flops, 557.43K parameters



