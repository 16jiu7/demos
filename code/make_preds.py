#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  1 22:58:12 2023

@author: jiu7
"""
import numpy as np
import skimage.io as io
from skimage.util import img_as_ubyte, img_as_float32
import torch
from make_graph_light import GraphedImage
from data_handeler import RetinalDataset
import torchvision
from torchvision.transforms import ToTensor, Resize
from skimage.transform import resize
from torch.nn.functional import sigmoid
from models.m2unet_3plus import M2UNet_3plus, M2UNet_self, M2UNet_3plus_dw, M2UNet_3plus_dw_deepsup  
  
def make_cnn_preds(cnn, ckpt_dir, single_data, pred_size, device = 'cuda' if torch.cuda.is_available() else 'cpu'):
    cnn.load_state_dict(torch.load(ckpt_dir), strict = False)
    cnn.to(device)
    
    bbox = single_data.get_bbox()
    ori = ToTensor()(single_data.ori).unsqueeze(0).to(device)
    ori = ori[:, :, bbox[0]:bbox[2], bbox[1]:bbox[3]]
    ori_resizer = Resize(size = pred_size)
    bbox_resier = Resize(size = (bbox[2] - bbox[0], bbox[3] - bbox[1]))
    resized_ori = ori_resizer(ori)
    resized_ori = resized_ori.to(device)
    
    cnn.eval()
    with torch.no_grad():
        pred = sigmoid(cnn(resized_ori, train = False))
    pred = bbox_resier(pred)
    final_pred = torch.zeros(size = (single_data.gt.shape[0], single_data.gt.shape[1]), dtype = torch.float32)
    final_pred[bbox[0]:bbox[2], bbox[1]:bbox[3]] = pred[0,0]
    final_pred = final_pred.data.numpy()
    return final_pred
    
    
model = M2UNet_3plus_dw_deepsup()
    
for drive in RetinalDataset('DRIVE').all_data:
    final_pred = make_cnn_preds(model, '../weights/cnn/DRIVE_7.pt', drive, (512, 512))  
    np.save(f'../4_retinal_datasets/DRIVE/preds/{drive.ID}.npy', final_pred)
    #io.imsave(f'../4_retinal_datasets/DRIVE/preds/{drive.ID}.png', final_pred)  
    
   
    
    
    
    
    
    
    
    
    