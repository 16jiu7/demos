#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
functions:
deep CNN feature extraction
GAT passing
feature combination
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.GAT import GAT
import numpy as np
from torchvision.transforms import ToTensor
from skimage.measure import regionprops
from make_graph_light import GraphedImage
from data_handeler import RetinalDataset
import networkx as nx
import datetime
from models.unet.unet_model import UNet

DIM_ENCODER_FEATS = 248 # k
GAT_INPUT_N_FEATS = 248
GAT_MID_N_C = 248
GAT_OUTPUT_N_FEATS = 248


def GetConcatNodeFeats(cnn_feats, graphedpred):
    # cnn_feats: NkHW
    # graphedpred: GraphedImage instance
    # order: by labels in graph.nodes
    # assume batch_size = 1
    nodes = list(graphedpred.graph.nodes)
    node_feats = torch.zeros(size = [len(nodes), cnn_feats.shape[1]], dtype = torch.float32) # N_nodes, k
    slic_label = graphedpred.slic_label
    r_props = regionprops(slic_label)
    
    for i, node in enumerate(nodes):
        feat_slices = [slice(None), slice(None), r_props[node - 1].slice[0], r_props[node - 1].slice[1]]
        node_feats_i = cnn_feats[feat_slices] # all feats within bbox, shape = N,k,H_bbox,W_bbox
        node_feats_i = node_feats_i.reshape(node_feats_i.shape[1], -1) # k,H_bbox*W_bbox
        _, _, V = torch.pca_lowrank(node_feats_i, center = True)
        node_feat_i = torch.matmul(node_feats_i, V[:, :1]) # k,1
        node_feats[i, :] = node_feat_i[:, 0]
        
    return node_feats

def GetNodeFeatsFromSmall(cnn_feats, graphedpred, down_ratio = 8):
    # cnn_feats: N, k, H/n, W/n, k = 256, n = down scale factor
    # graphedpred: GraphedImage instance
    # order: by labels in graph.nodes
    # assume batch_size = 1
    nodes = list(graphedpred.graph.nodes)
    node_feats = torch.zeros(size = [len(nodes), cnn_feats.shape[1]], dtype = torch.float32) # N_nodes, k
    slic_label = graphedpred.slic_label
    r_props = regionprops(slic_label)
    
    for i, node in enumerate(nodes):
        min_row, min_col, max_row, max_col = r_props[node - 1].bbox
        min_row, min_col, max_row, max_col = \
        min_row // down_ratio, min_col // down_ratio, max_row // down_ratio, max_col // down_ratio
        feat_slices = [slice(None), slice(None), slice(min_row, max_row), slice(min_col, max_col)]
        node_feats_i = cnn_feats[feat_slices] # all feats within bbox, shape = N,k,H_bbox,W_bbox
        node_feats_i = node_feats_i.reshape(node_feats_i.shape[1], -1) # k,H_bbox*W_bbox
        if node_feats_i.shape[-1] > 1:
            _, _, V = torch.pca_lowrank(node_feats_i, center = True)
            node_feat_i = torch.matmul(node_feats_i, V[:, :1]) # k,1
        node_feats[i, :] = node_feat_i[:, 0]
        
    return node_feats

def FuseFeats(cnn_feats, node_feats, graph, down_ratio = 8, method = 'replace'):
    # fuse node feats(GAT output) into CNN feats
    # graph can be graphedpred.graph after relabeling
    # scale factor = cnn feat size / ori size
    if method == 'replace':
        for node in graph.nodes:
            pos = (graph.nodes[node]['center'][0] // down_ratio, graph.nodes[node]['center'][1] // down_ratio)
            cnn_feats[:, :, pos[0], pos[1]] = node_feats[node, :]
    
    return cnn_feats
            
        
        
if __name__ == '__main__':
    # what kind of CNN feats to extract node feats from:
    # 1.plain encoder feats, shape = N, 256, H/8, W/8
    # 2.deep multi-scale feats, shape = N, 256+32+64+128, H/8, W/8
    # 3.large multi-scale feats, shape = N, 256+32+64+128, H, W
    
    # how to compress node feats
    # 1.max pooing
    # 2.PCA
    starttime = datetime.datetime.now()

    data = RetinalDataset('DRIVE').all_training[0] # the first training img in DRIVE
    graphedpred = GraphedImage(data.pred, data.fov_mask, 1500)
    # define net
    net = UNet_3_32(3, 2)  
    checkpoint = torch.load('../weights/UNet_3_32.pt7')
    net.load_state_dict(checkpoint['net'])        
        
    cnn_feats = GetCNNFeats(data.ori, net)
    node_feats = GetNodeFeatsFromSmall(cnn_feats, graphedpred)
    # once we got node_feats, we can relabel the graph 
    mapping = {}
    old_labels = list(graphedpred.graph.nodes)
    new_labels = [i for i in range(len(old_labels))] # new node labels start from 0, due to GAT need
    for i, old_label in enumerate(old_labels): mapping[old_label] = new_labels[i]
    relabeled_graph = nx.relabel_nodes(graphedpred.graph, mapping)
    edges = list(relabeled_graph.edges)
    edges = torch.Tensor(edges).long().transpose(1,0) # shape = (2, E), data type = torch.long for GAT use
    graph_data = (node_feats, edges)
    
    gat = GAT(num_of_layers = 4, num_heads_per_layer = [4,4,4,4], 
               num_features_per_layer = [GAT_INPUT_N_FEATS, GAT_MID_N_C, GAT_MID_N_C, GAT_MID_N_C, GAT_OUTPUT_N_FEATS], dropout = 0)
    
    node_feats_gat, _ = gat(graph_data)
    fused_feats = FuseFeats(cnn_feats, node_feats_gat, relabeled_graph)
    
    endtime = datetime.datetime.now()
    print(f'Run time : {endtime - starttime}s')