#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  1 22:58:12 2023

@author: jiu7
"""

import numpy as np
import skimage.io as io
from skimage.util import img_as_ubyte, img_as_float32
import networkx as nx
import torch
from make_graph_light import GraphedImage
from data_handeler import RetinalDataset
from models.unet_3_32.unet_3_32 import UNet_3_32
from torchvision.transforms import ToTensor
from skimage.transform import resize

net = UNet_3_32(3, 2)  
checkpoint = torch.load('../weights/UNet_3_32.pt7')
net.load_state_dict(checkpoint['net'])
net.cuda()

datas = RetinalDataset('HRF').all_data

for data in datas:
    ori = resize(data.ori, output_shape = (int(data.ori.shape[0] / 8), int(data.ori.shape[1] / 8)), preserve_range=True)    
    ori = ori.astype(np.float32)
    ori = ToTensor()(ori).unsqueeze(0).cuda()
    pred = torch.nn.functional.softmax(net(ori), dim=1).data.cpu().numpy()[0, 1]
    pred = resize(pred, output_shape = (data.ori.shape[0], data.ori.shape[1]))
    io.imsave('/home/jiu7/demos/4_retinal_datasets/HRF/preds/' + data.ID + '.png', pred)
    
