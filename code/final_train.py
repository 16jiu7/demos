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
from skimage.measure import regionprops
import skimage.io as io
import pickle

CNN_NAME = 'UNet_small_4_8' # 4 times of downscaling, the first conv layer has 8 channels 
TRAIN_DATASET = 'CHASEDB'
# 'DRIVE', 'CHASEDB', 'HRF', 'STARE'
TRAIN_INPUT_SIZE = {'DRIVE': [584, 565], 'CHASEDB':[960, 999], 'HRF':[1024, 1536], 'STARE':[605, 700]}
# same sizes for training, val and test
INTERMEDIATE_DIR = f'../preds/intermediate/{TRAIN_DATASET}/'  
CNN_N_CHANNELS = [8, 16, 32, 64, 128]
GNN_N_LAYERS = 5 # 4 for encoder, the last 1 is the output layer
GNN_N_HEADS = 4

cnn = UNet(3, 2, channels = CNN_N_CHANNELS)       
gnn = GAT(num_of_layers = GNN_N_LAYERS, num_heads_per_layer = [GNN_N_HEADS] * GNN_N_LAYERS, 
          num_features_per_layer = [248, 248, 248, 248, CNN_N_CHANNELS[-1], 2], 
          dropout = 0, get_feats = True)

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

N_PIECES = {'DRIVE':1000, 'CHASEDB':1000, 'HRF':2000, 'STARE':1000}


# In[]
# prepare predictions and graphs and slic labels for training+val set
train_val = RetinalDataset(TRAIN_DATASET, cropped = True).all_training + \
RetinalDataset(TRAIN_DATASET, cropped = True).all_val

preds_dir, graph_dir, slic_dir = INTERMEDIATE_DIR + 'preds/', INTERMEDIATE_DIR + 'graph/', INTERMEDIATE_DIR + 'slic/'
if not os.path.isdir(preds_dir):os.mkdir(preds_dir)
if not os.path.isdir(graph_dir):os.mkdir(graph_dir)
if not os.path.isdir(slic_dir):os.mkdir(slic_dir)    

resizer = A.Compose([Resize(*TRAIN_INPUT_SIZE[TRAIN_DATASET])])
cnn.cuda()    
for data in train_val:
    # save CNN pre-trained predictions
    inputs = resizer(image = data.ori, mask = data.fov_mask)
    inputs, fov_mask = inputs['image'], inputs['mask']
    inputs = totensor(inputs).cuda().unsqueeze(0)
    pred = nn.functional.softmax(cnn(inputs), dim=1)
    pred = pred[0,1].detach().cpu().numpy()
    pred = img_as_ubyte(pred)
    io.imsave(preds_dir + f'{data.ID}.png', pred)
    # save graphs and slic labels
    graphedpred = GraphedImage(pred, fov_mask, N_PIECES[TRAIN_DATASET])
    graph = graphedpred.graph
    with open(graph_dir + f'{data.ID}.pkl', "wb") as f:    
        pickle.dump(graph, f)

    slic_label = graphedpred.slic_label
    np.save(slic_dir + f'{data.ID}.npy', slic_label)
    
    
print(f'pred, slic_label and graph for {TRAIN_DATASET} prepared !')  

  
# In[]
# transform: apply to ori + slic_label + node center points all at once for data aug 
brightness, contrast, saturation, hue = 0.25, 0.25, 0.25, 0.01
train_transforms = A.Compose([Resize(*TRAIN_INPUT_SIZE[TRAIN_DATASET]), RandomRotate90(p = 0.5), 
                              Flip(p = 0.5)],keypoint_params = A.KeypointParams(format = 'yx'))
color_jitter = ColorJitter(brightness, contrast, saturation, hue, always_apply = True)
val_resizer = A.Compose([Resize(*TRAIN_INPUT_SIZE[TRAIN_DATASET])], keypoint_params = A.KeypointParams(format = 'yx'))


def TransformGraph(graph, slic_label):
    # do transforms for graphs, may remove some of the nodes
    old_labels = list(graph.nodes)
    new_labels = np.unique(slic_label)
    missing_labels = list(set(old_labels) - set(new_labels))
    graph = graph.remove_nodes_from(missing_labels)
    # make dict, new_labels as keys, tuple of (center, slices) as values
    new_regions = regionprops(slic_label)
    nodes_attrs = {}
    for region in new_regions:
        center_y, center_x = region.centroid
        center = (int(center_y), int(center_x))
        nodes_attrs[region.label] = (center, region.slice)
    # change nodes attrs according to the dict
    for node in graph.nodes:
        node['center'] = nodes_attrs[node][0]
        node['slices'] = nodes_attrs[node][1]       
    return graph    

class TrainDataset(Dataset):
    def __init__(self, dataset_name, split, transforms = None, color_jitter = None):
        self.split = split
        self.dataset_name = dataset_name
        self.transforms = transforms
        self.color_jitter = color_jitter
        if split == 'train':
            self.data = RetinalDataset(self.dataset_name, cropped = True).all_training
        elif split == 'val':
            self.data = RetinalDataset(self.dataset_name, cropped = True).all_val

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        graph = pickle.load(open(INTERMEDIATE_DIR + 'graph/' + f'{data.ID}.pkl', 'rb'))
        slic_label = np.load(INTERMEDIATE_DIR + 'slic/' + f'{data.ID}.npy')
        
        transformed = self.transforms(image = data.ori, masks = [data.gt, slic_label])
        img, gt, slic_label = transformed['image'], transformed['masks'][0], transformed['masks'][1]
        if self.split == 'train':
            img = self.color_jitter(image = img)['image']
        data.ori, data.gt = img, gt    
        graph = TransformGraph(graph, slic_label)
        
        return data, slic_label, graph

train_set = TrainDataset(dataset_name = TRAIN_DATASET, split = 'train', transforms = train_transforms)
val_set = TrainDataset(dataset_name = TRAIN_DATASET, split = 'val', transforms = val_resizer)
    
train_loader = DataLoader(train_set, batch_size = 1, num_workers = 0, shuffle = True)
val_loader = DataLoader(val_set, batch_size = 1, num_workers = 0, shuffle = False)

# In[]


def GetConcatCNNFeats(cnn_model, single_data, device = 'cuda' if torch.cuda.is_available() else 'cpu'):
    ori = totensor(single_data.ori).unsqueeze(0)
    ori = ori.to(device)
    cnn_model.to(device)
    feats, side_outs = cnn_model.get_concat_feats(ori)
    return feats, side_outs
    
def GetConcatNodeFeats(cnn_feats, graph, slic_label):
    # type : (Tensor, graph, array) -> Tensor
    # cnn_feats: NkHW
    # graphedpred: GraphedImage instance
    # order: by labels in graph.nodes
    # assume batch_size = 1
    nodes = list(graph.nodes)
    node_feats = torch.zeros(size = [len(nodes), cnn_feats.shape[1]], dtype = torch.float32) # N_nodes, k
    r_props = regionprops(slic_label)
    
    for i, node in enumerate(nodes):
        feat_slices = [slice(None), slice(None), r_props[node - 1].slice[0], r_props[node - 1].slice[1]]
        node_feats_i = cnn_feats[feat_slices] # all feats within bbox, shape = N,k,H_bbox,W_bbox
        node_feats_i = node_feats_i.reshape(node_feats_i.shape[1], -1) # k,H_bbox*W_bbox
        _, _, V = torch.pca_lowrank(node_feats_i, center = True)
        node_feat_i = torch.matmul(node_feats_i, V[:, :1]) # k,1
        node_feats[i, :] = node_feat_i[:, 0]
        
    return node_feats

def GetRelabeledGraphEdges(graph, device = 'cuda' if torch.cuda.is_available() else 'cpu'):
    mapping = {}
    old_labels = list(graph.nodes)
    new_labels = [i for i in range(len(old_labels))] # new node labels start from 0, due to GAT need
    for i, old_label in enumerate(old_labels): mapping[old_label] = new_labels[i]
    relabeled_graph = nx.relabel_nodes(graph, mapping)
    edges = list(relabeled_graph.edges)
    edges = torch.Tensor(edges).long().transpose(1,0) # shape = (2, E), data type = torch.long for GAT use
    return edges.to(device)   
    

def ReplaceNodeCenterFeats(graph, feats_to_replace, node_feats, down_ratio = 16):
    # replace feats vector on the center of slic pieces
    # in node feats, node labels from 0 to N-1, while graph.nodes are arrordence with slic label
    for idx, node in enumerate(graph.nodes):
        pos = (graph.nodes[node]['center'][0] // down_ratio, graph.nodes[node]['center'][1] // down_ratio)
        feats_to_replace[:, :, pos[0], pos[1]] = node_feats[idx, :]
    return feats_to_replace

def MakeNodesGT(graph, single_data):
    # make GT for node vessel density prediction
    nodes_gt = []
    gt = np.float32(single_data.gt / 255.)
    for node in graph.nodes: 
        slices = graph.nodes[node]['slices']
        mass = np.sum(gt[slices])
        area = (slices[0].stop - slices[0].start) * (slices[1].stop - slices[1].start)
        nodes_gt.append(mass / area)
    return np.array(nodes_gt, dtype = np.float32)    
      
def RunForwardPass(cnn, gnn, single_data, slic_label, graph):
    # run CNN encoder
    node_feats = GetConcatNodeFeats(concat_cnn_feats, graph, slic_label)
    edges = GetRelabeledGraphEdges(graph)
    
    # do a GNN forward pass, get node feats and node prediction results
    (node_feats_gat, _), (node_density_logits, _) = gnn((node_feats, edges))
    node_feats_gat = node_feats_gat.reshape(-1, node_feats_gat.shape[-1] // GNN_N_HEADS, GNN_N_HEADS)
    node_feats_gat = node_feats_gat.mean(-1) # do mean average on heads dim
    
    shallows, bottom = side_outs[:4], side_outs[-1]
    bottom_replaced = ReplaceNodeCenterFeats(graph, bottom, node_feats_gat, down_ratio = 16)
    final_logits = cnn.run_decoder(bottom_replaced, shallows)
    
    return final_logits, node_density_logits # final logits and gnn predictions for node vessel density
    
if __name__ == '__main__':
    # in dataloader 
    idx = 1
    data = train_val[idx]
    concat_cnn_feats, side_outs = GetConcatCNNFeats(cnn, data)
    graph_dir = INTERMEDIATE_DIR + 'graph/'
    slic_dir = INTERMEDIATE_DIR + 'slic/'
    graph = pickle.load(open(graph_dir + f'{data.ID}.pkl', 'rb'))
    slic_label = np.load(slic_dir + f'{data.ID}.npy')
    
    # run CNN encoder
    node_feats = GetConcatNodeFeats(concat_cnn_feats, graph, slic_label)
    edges = GetRelabeledGraphEdges(graph)
    
    # do a GNN forward pass, get node feats and node classification results
    (node_feats_gat, _), (node_cls_res, _) = gnn((node_feats, edges))
    node_feats_gat = node_feats_gat.reshape(-1, node_feats_gat.shape[-1] // GNN_N_HEADS, GNN_N_HEADS)
    node_feats_gat = node_feats_gat.mean(-1) # do mean average on heads dim
    
    shallows, bottom = side_outs[:4], side_outs[-1]
    bottom_replaced = ReplaceNodeCenterFeats(graph, bottom, node_feats_gat, down_ratio = 16)
    
    logits = cnn.run_decoder(bottom_replaced, shallows)
    
    nodes_gt = MakeNodesGT(graph, data)

    



























