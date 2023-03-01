#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 21:31:07 2023

@author: jiu7
"""
import os, random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision.transforms import ToTensor
totensor = ToTensor()
from torch.utils.data import Dataset, DataLoader
from thop import profile, clever_format
import albumentations as A
from albumentations.augmentations.geometric.rotate import RandomRotate90
from albumentations.augmentations.geometric.transforms import Flip, Affine
from albumentations.augmentations.transforms import ColorJitter
from albumentations.augmentations.geometric.resize import Resize
from make_graph_light import GraphedImage
from data_handeler import RetinalDataset
import networkx as nx
import datetime
from models.unet.unet_model import UNet
from models.GAT import GAT
from skimage.util import img_as_ubyte
import skimage.io as io
import pickle

CNN_NAME = 'UNet_small_4_8' # 4 times of downscaling, the first conv layer has 8 channels 
TRAIN_DATASET = 'STARE'
# 'DRIVE', 'CHASEDB', 'HRF', 'STARE'
INTERMEDIATE_DIR = f'../preds/intermediate/{TRAIN_DATASET}/'  
CNN_N_CHANNELS = [8, 16, 32, 64, 128]
GNN_N_LAYERS = 4
GNN_N_HEADS = 4

cnn = UNet(3, 2, channels = CNN_N_CHANNELS)       
gnn = GAT(num_of_layers = GNN_N_LAYERS, num_heads_per_layer = [GNN_N_HEADS] * GNN_N_LAYERS, 
          num_features_per_layer = [sum(CNN_N_CHANNELS)] * (GNN_N_LAYERS + 1), dropout = 0)
checkpoint = torch.load(f'../weights/pre_training/{TRAIN_DATASET}_pre.pt')
cnn.load_state_dict(checkpoint) 
print(f'CNN model name {CNN_NAME}')
print(f'CNN pre-trained weights for {TRAIN_DATASET} loaded')
inputs = torch.randn(1, 3, 512, 512)
flops, params = profile(cnn, inputs = (inputs,), verbose=False)
flops, params = clever_format([flops, params])
print(f'the CNN model has {flops} flops, {params} parameters')

def setup_random_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

class TrainDataset(Dataset):
    def __init__(self, dataset_name, split, transforms = None):
        self.split = split
        self.dataset_name = dataset_name
        self.transforms = transforms
        if split == 'train':
            self.data = RetinalDataset(self.dataset_name, cropped = True).all_training
        elif split == 'val':
            self.data = RetinalDataset(self.dataset_name, cropped = True).all_val
        elif split == 'test':
            self.data = RetinalDataset(self.dataset_name, cropped = False).all_test # metrics should be calculate on original images
            
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        
        if self.split == 'test':
            cropped_img, gt = self.data[idx].cropped_ori, self.data[idx].gt # give cropped ori & non-cropped gt
            return totensor(cropped_img).cuda(), totensor(gt).cuda(), self.data[idx].bbox, self.data[idx].ID
        
        if self.split in ['train', 'val']:
            img, gt = self.data[idx].ori, self.data[idx].gt
            if self.transforms:
                transformed = self.transforms(image = img, mask = gt)
                img, gt = transformed['image'], transformed['mask']
            return totensor(img).cuda(), totensor(gt).cuda()


# prepare predictions and graphs and slic labels for training+val set
N_PIECES = {'DRIVE':1000, 'CHASEDB':1000, 'HRF':2000, 'STARE':1000}

train_val = RetinalDataset(TRAIN_DATASET, cropped = True).all_training + \
RetinalDataset(TRAIN_DATASET, cropped = True).all_val

preds_dir, graph_dir, slic_dir = INTERMEDIATE_DIR + 'preds/', INTERMEDIATE_DIR + 'graph/', INTERMEDIATE_DIR + 'slic/'
if not os.path.isdir(preds_dir):os.mkdir(preds_dir)
if not os.path.isdir(graph_dir):os.mkdir(graph_dir)
if not os.path.isdir(slic_dir):os.mkdir(slic_dir)    

cnn.cuda()    
for data in train_val:
    # save CNN pre-trained predictions
    inputs = totensor(data.ori).cuda()
    pred = nn.functional.softmax(cnn(inputs.unsqueeze(0)), dim=1)
    pred = pred[0,1].detach().cpu().numpy()
    pred = img_as_ubyte(pred)
    io.imsave(preds_dir + f'{data.ID}.png', pred)
    # save graphs and slic labels
    graphedpred = GraphedImage(pred, data.fov_mask, N_PIECES[TRAIN_DATASET])
    graph = graphedpred.graph
    #nx.write_edgelist(graph, graph_dir + f'{data.ID}.graph')
    with open(graph_dir + f'{data.ID}.pkl', "wb") as f:    
        pickle.dump(graph, f)

    slic_label = graphedpred.slic_label
    np.save(slic_dir + f'{data.ID}.npy', slic_label)

# In[]
graph1 = pickle.load(open(graph_dir + '01.pkl', 'rb'))
print(graph1.nodes[96])







