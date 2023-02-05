#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 26 10:34:46 2023

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
from skimage.measure import regionprops

data = RetinalDataset('DRIVE').all_training[0] # the first training img in DRIVE
graphedpred = GraphedImage(data.pred, data.fov_mask, 600)
label = graphedpred.slic_label
edges = graphedpred.edge_index
print(edges.dtype)
nodes = list(graphedpred.graph.nodes)

# In[]

resized_label = resize(label, output_shape = (int(584 / 4), int(565 / 4)), preserve_range=True).astype(np.int64)
print(np.unique(resized_label))
print(np.unique(label))
io.imsave('label.png', label)
io.imsave('label_resized.png', resized_label)

def GetFirstPCNN(img, net):
    # get temp CNN pred result, used in node sampling & graph building
    # net in no_gnn mode
    # only run once when training begins
    
    
    pass



def GetNodeFeats(cnn_feats, graphedpred):
    # cnn_feats: NkHW, k = 144
    # graphedpred: GraphedImage instance
    # method1: zero padding & concat
    # method2: add
    # method3: replace
    # order: by labels in graph.nodes
    # assume batch_size = 1
    nodes = list(graphedpred.graph.nodes)
    node_feats = torch.zeros(size = [cnn_feats.shape[0], len(nodes), cnn_feats.shape[1]], dtype = torch.float32) # N(batch_size), N_nodes, k
    slic_label = graphedpred.slic_label
    r_props = regionprops(slic_label)
    
    for i, node in enumerate(nodes):
        feat_slices = [slice(None), slice(None), r_props[node - 1].slice[0], r_props[node - 1].slice[1]]
        node_feats_i = cnn_feats[feat_slices] # all feats within bbox, shape = N,k,H_bbox,W_bbox
        node_feats_i = node_feats_i.reshape(node_feats_i.shape[0], node_feats_i.shape[1], -1) # N,k,H_bbox*W_bbox
        _, _, V = torch.pca_lowrank(node_feats_i, center = True)
        node_feat_i = torch.bmm(node_feats_i, V[:, :, :1]) # N,k,1
        node_feats[i, :] = node_feat_i[]
def CompressNodeFeats():
    # method1: maxpooling
    # method2: PCA pooling
    pass

def ReProjectNodeFeats(node_feats: list, slic_label, scale_factor):
    # place node features (the GAT outputs) into the downscaled feature map, 
    # if downscaled slic label no longer have certain pieces, just ignore them
    N_nodes, N_dim = node_feats.shape[0], node_feats.shape[1]
    resized_label = resize(label, output_shape = \
    (int(584 / scale_factor), int(565 / scale_factor)), preserve_range=True).astype(np.int64)
    survived_labels = list(np.unique(resized_label))
    survived_labels.remove(0)
    
    props = regionprops(label_image = resized_label, intensity_image = None)
    for prop in props:
        
        
    
    
    
        pass


def GetFinRes():
    # get the final CNN+GAT pred result
    pass












