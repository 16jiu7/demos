#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 16:27:15 2023

@author: jiu7
"""
import os, pickle
from skimage.util import img_as_ubyte, img_as_bool, img_as_float32
import skimage.io as io
import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from data_handeler import RetinalDataset
import torchvision
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset, DataLoader
import random
from torchmetrics.classification import BinaryAveragePrecision
import matplotlib.pyplot as plt
from thop import profile, clever_format
from datetime import datetime
import albumentations as A
from albumentations.augmentations.geometric.rotate import RandomRotate90
from albumentations.augmentations.geometric.transforms import Flip, Affine
from albumentations.augmentations.transforms import ColorJitter
from albumentations.augmentations.geometric.resize import Resize
from albumentations.augmentations.dropout.coarse_dropout import CoarseDropout
from models.m2unet_3plus import M2UNet_3plus, M2UNet_self, M2UNet_3plus_dw, M2UNet_3plus_dw_deepsup
from segmentation_models_pytorch.losses import JaccardLoss

def setup_random_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True 


test_device = 'cuda'
test_inputs = torch.randn([1,3,544, 544]).to(test_device)
cnn = M2UNet_3plus_dw_deepsup(in_channels=3, n_classes=1)  
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

brightness, contrast, saturation, hue = 0.25, 0.25, 0.02, 0.02
color_jitter = ColorJitter(brightness, contrast, saturation, hue, p = 0.1)
half_resizer = Resize(256, 256, always_apply = True) # allow cnn running and half the size
ori_resizer = Resize(512, 512, always_apply = True) # only to allow cnn running 

geo_transforms = A.Compose([Resize(512, 512, always_apply = True), 
                            RandomRotate90(p = 0.75), 
                            Flip(p = 0.3),
                            Affine(translate_percent = 0.05, rotate = (-15, 15), p = 0.5)])

geo_transforms_dropout = A.Compose([Resize(512, 512, always_apply = True), 
                            RandomRotate90(p = 0.75), 
                            Flip(p = 0.3),
                            Affine(translate_percent = 0.05, rotate = (-15, 15), p = 0.5),
                            CoarseDropout(p = 0.1)])

class TrainDataset(Dataset):
    def __init__(self, dataset_name, split, color_jitter = None):
        
        self.split = split
        self.dataset_name = dataset_name
        self.color_jitter = color_jitter
        if split == 'train':
            self.data = RetinalDataset(self.dataset_name, cropped = False).all_training
        elif split == 'val':
            self.data = RetinalDataset(self.dataset_name, cropped = False).all_val
        elif split == 'test':
            self.data = RetinalDataset(self.dataset_name, cropped = False).all_test

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        # train mode: ori and gt are cropped to fov, resized to 512*512 and apply geo transforms 
        # ori goes through color jitter if there is any 
        if self.split == 'train':
            data.crop2fov(mode = 'all')
            data.ori = self.color_jitter(image = data.ori)['image'] if self.color_jitter is not None else data.ori
            resized = geo_transforms_dropout(image = data.ori, mask = data.gt)
            ori = ToTensor()(resized['image'])
            gt = ToTensor()(resized['mask'])
            return ori, gt
        # val mode: ori is resized to 512*512    
        elif self.split == 'val':
            min_r, min_c, max_r, max_c = data.get_bbox()
            ori = data.ori[min_r:max_r, min_c:max_c]
            resized = ori_resizer(image = ori)
            ori = resized['image']
            return ori, data.gt, data.fov_mask, data.bbox
        # test mode, ori is resized to 512*512    
        elif self.split == 'test':
            min_r, min_c, max_r, max_c = data.get_bbox()
            ori = data.ori[min_r:max_r, min_c:max_c]
            resized = ori_resizer(image = ori)
            ori = resized['image']
            return ori, data.gt, data.fov_mask, data.bbox

def my_collate_fn(batch):
    return batch[0]        
        
def post_process(pred, gt, fov_mask, bbox):
    # make pred the same size as groundtruth
    resizer = torchvision.transforms.Resize(size = (bbox[2] - bbox[0], bbox[3] - bbox[1]))
    fov_mask = ToTensor()(fov_mask).to(gt.device)
    pred = resizer(pred)
    out = torch.zeros_like(gt, device = gt.device)
    out[:,bbox[0]:bbox[2], bbox[1]:bbox[3]] = pred[0,0]
    out = out * fov_mask
    return out.squeeze()
        
def draw_figs(x, losses, vals):
    gap = len(x) // 10
    
    plt.figure(figsize=(10,5), dpi = 200)
    plt.subplot(211)
    plt.title('loss')
    plt.grid()
    plt.plot(x, losses, c = 'b', marker = '.', linewidth = 1, markersize = 2)
    for epoch, loss in zip(x, losses):
        if epoch % gap == 0:
            plt.text(epoch, loss, f'{loss:.3f}', fontdict={'fontsize':6})
    

    plt.subplot(212)
    plt.title('val')
    plt.grid()
    plt.plot(x, vals, c = 'r', marker = '.', linewidth = 1, markersize = 2)
    for epoch, val in zip(x, vals):
        if epoch % gap == 0:
            plt.text(epoch, val, f'{val:.3f}', fontdict={'fontsize':6})

    plt.tight_layout() # otherwise subplots will land on each other
    plt.savefig('losses_val.png')     
# In[]
setup_random_seed(7)    
TRAIN_DATASET = 'DRIVE'
# criterion = torch.nn.BCEWithLogitsLoss()
# criterion = JaccardLoss(mode = 'binary')

def JBCE_loss(y_pred, y_true):
    loss1 = torch.nn.BCEWithLogitsLoss()(y_pred, y_true)
    loss2 = JaccardLoss(mode = 'binary')(y_pred, y_true)
    return loss1 + 0.3 * loss2
    
n_epoch = 600
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
val_loader = DataLoader(val_set, batch_size = 1, num_workers = 0, shuffle = False, collate_fn = my_collate_fn)
test_loader = DataLoader(test_set, batch_size = 1, num_workers = 0, shuffle = False, collate_fn = my_collate_fn)

losses = []
val_aps = []

for epoch in range(n_epoch):
    epoch_loss = 0
    for ori, gt in train_loader:
        ori, gt = ori.to(device), gt.to(device)
        # pred = cnn(ori)
        # loss = criterion(pred, gt)
        xout, deepsup1, deepsup2, deepsup3, deepsup4, deepsup5  = cnn(ori)
        outputs = cnn(ori)
        loss = 0
        for output in outputs: loss += JBCE_loss(output, gt)
        loss = loss / len(outputs)
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
        for idx, (ori, gt, fov_mask, bbox) in enumerate(val_loader):
            ori, gt = ToTensor()(ori).to(device), ToTensor()(gt).to(device)
            ori = ori.unsqueeze(0)
            pred = torch.sigmoid(cnn(ori, train=False))
            pred = post_process(pred, gt, fov_mask, bbox)
            ap = val_criterion(pred, gt[0]).cpu().numpy()
            mean_ap += ap
            pred = (pred.detach().cpu().numpy())
            io.imsave(f'cnn_val_{idx}.png', img_as_ubyte(pred))
        mean_ap = mean_ap / len(val_loader)
        val_aps.append(mean_ap)
        print(f'val AP: {mean_ap:.4f}')

        if np.argmax(np.array(val_aps)) == epoch:
            print(f'epoch {epoch + 1}, save weights')
            torch.save(cnn.state_dict(), 'cnn.pt')
            
draw_figs([i for i in range(1, n_epoch + 1)], losses, val_aps)     
     
del cnn
# In[]
# test
setup_random_seed(100) # otherwise test results are not consistent
test_device = 'cuda'
cnn = M2UNet_3plus_dw_deepsup(in_channels=3, n_classes=1)  
# cnn = M2UNet()
cnn.load_state_dict(torch.load('cnn.pt'), strict=False)
cnn.to(test_device)
cnn.eval()

with torch.no_grad():
    mean_ap = 0
    for idx, (ori, gt, fov_mask, bbox) in enumerate(test_loader):
        ori, gt = ToTensor()(ori).to(device), ToTensor()(gt).to(device)
        ori = ori.unsqueeze(0)
        pred = torch.sigmoid(cnn(ori, train=False))
        pred = post_process(pred, gt, fov_mask, bbox)
        ap = val_criterion(pred, gt.squeeze_()).cpu().numpy()
        mean_ap += ap
        pred = (pred.detach().cpu().numpy())
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


# M2U-Net 1.55G flops, 549.75K parameters 
# 0.8228, 0.8283, 0.8193 ~0.8235
                                                                                                                                                                    
# M2UNet_3plus 3.86G flops, 607.86K parameters
# 0.8388, 0.8433, 0.8358 ~0.8393

# M2UNet_3plus_dw 2.25G flops, 557.01K parameters
# 0.8287, 0.8348, 0.8276 ~0.8304

# M2UNet_3plus_dw with ori shortcut 2.28G flops, 557.43K parameters
# 0.8360, 0.8292, 0.8410 ~0.8354

# M2UNet_3plus_dw with ori shortcut and deepsup 2.42G flops, 559.48K parameters
# 300 epoch
# 0.8367, 0.8349, 0.8385 ~0.8367

# M2UNet_3plus_dw with ori shortcut and deepsup 2.42G flops, 559.48K parameters
# 300 epoch, with geo transforms, w/o colorjitter
# 0.8461, 0.8493

# M2UNet_3plus_dw with ori shortcut and deepsup 2.42G flops, 559.48K parameters
# 300 epoch, with geo transforms, 0.5 affine, w/o colorjitter
# 0.8648, 0.8631, 0.8614 ~0.8631
# 600 epoch 
# 0.8714, 0.8699, 0.8677                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      

# M2UNet_3plus_dw with ori shortcut, w/o deepest shortcut, with deepsup 2.16G flops, 557.30K parameters
# 600 epoch, with geo transforms, w/o colorjitter
# 0.8702, 


# coarse dropout
# 0.8731, 0.8712, 0.8707


















