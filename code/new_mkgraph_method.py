#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 16:27:15 2023

@author: jiu7
"""
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.measure import regionprops, label
from skimage.util import img_as_ubyte, img_as_bool, img_as_float32
from skimage.filters import threshold_minimum, threshold_otsu
from skimage import draw
from skimage.morphology import remove_small_objects
import skimage.io as io
import numpy as np
import networkx as nx
import torch
from data_handeler import RetinalDataset
import datetime
import networkx as nx
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset, DataLoader
from models.GAT import GAT
import random
from torch.autograd import Variable
from torchmetrics.classification import BinaryAveragePrecision


def get_bboxes_from_pred(pred, n_bboxes = 700, params = {'threshold': 'otsu', 'neg_iso_size': 6, 'n_slic_pieces': 800,\
                                                         'remove_keep': 36}):
    # larger neg_iso_size, less node, other params don't matter a lot
    assert pred.dtype == np.uint8
    if params['threshold'] == 'otsu':
        threshold = threshold_otsu(pred)
    certain_pred = pred >= threshold
    certain_pred = remove_small_objects(certain_pred, min_size = params['neg_iso_size'])
    
    all_bboxes = []
    iso_label = label(certain_pred)
    iso_regions = regionprops(iso_label)
    small_iso_bboxes = [x.bbox for x in iso_regions if max(x.bbox[2] - x.bbox[0], x.bbox[3] - x.bbox[1]) <= 17]
    all_bboxes += small_iso_bboxes
    
    large_isos = certain_pred.copy()
    for x in small_iso_bboxes: large_isos[x[0]:x[2], x[1]:x[3]] = 0
    slic_label = slic(large_isos, params['n_slic_pieces'], mask = large_isos, enforce_connectivity = False, compactness = 100)
    slic_regions = regionprops(slic_label)
    
    for x in slic_regions:
        if max(x.bbox[2] - x.bbox[0], x.bbox[3] - x.bbox[1]) <= 17:
            all_bboxes.append(x.bbox)
        else:
            offset = (x.bbox[0], x.bbox[1])
            patch = x.image
            n_pieces = ((patch.shape[0] // 17) + 1) * ((patch.shape[1] // 17) + 1)
            patch_slic = slic(patch, n_pieces, mask = patch, enforce_connectivity = False)
            patch_slic_regions = regionprops(patch_slic)
            patch_bboxes = [(region_patch.bbox[0] + offset[0], region_patch.bbox[1] + offset[1], region_patch.bbox[2] + offset[0], region_patch.bbox[3] + offset[1])\
                            for region_patch in patch_slic_regions]

            all_bboxes += patch_bboxes

    small_bboxes = [x for x in all_bboxes if (x[2] - x[0]) <= 17 and (x[3] - x[1]) <= 17]
    large_bboxes = [x for x in all_bboxes if (x[2] - x[0]) > 17 or (x[3] - x[1]) > 17]
    for idx, x in enumerate(small_bboxes): #expend small bboxes
        h, w = x[2] - x[0], x[3] - x[1]
        if h < 17 or w < 17:
            center = (int((x[2] + x[0]) / 2), int((x[3] + x[1]) / 2))
            small_bboxes[idx] = (center[0] - 8, center[1] - 8, center[0] + 9, center[1] + 9)
            
    for x in large_bboxes: #split large bboxes
        h, w = x[2] - x[0], x[3] - x[1]
        h_n, w_n = h // 17 + 1, w // 17 + 1
        sub_bboxes = []
        for i in range(h_n):
            for j in range(w_n):
                top_left = (x[0] + (i + 1) * 17, x[1] + (j + 1) * 17)
                sub_bboxes.append((top_left[0], top_left[1], top_left[0] + 18, top_left[1] + 18))
        small_bboxes += sub_bboxes
    # remove bboxes out of range    
    small_bboxes = list(filter(lambda x: (x[0]>=0) and (x[1]>=0) and(x[2]<=pred.shape[0]) and (x[3]<=pred.shape[1]), small_bboxes))
    small_bboxes = list(filter(lambda x: certain_pred[x[0]:x[2], x[1]:x[3]].sum() > 0, small_bboxes))
    # randomly remove bboxes until n_bboxes bboxes left, will keep bboxes containing small vessels
    n_to_remove = len(small_bboxes) - n_bboxes # number to remove 
    assert n_to_remove >= 0, f'not enough bboxes (< {n_bboxes}) proposed!'
    removeable_idx = [i for i in range(len(small_bboxes)) if certain_pred[small_bboxes[i][0]:small_bboxes[i][2], small_bboxes[i][1]:small_bboxes[i][3]].sum() > params['remove_keep']]
    random.shuffle(removeable_idx)
    idx_to_remove = removeable_idx[:n_to_remove]
    small_bboxes = [bbox for i, bbox in enumerate(small_bboxes) if i not in idx_to_remove]
    return small_bboxes, certain_pred


def add_edges(bboxes, narrow = 100, n_links = 2):
    assert narrow < 256, 'param narrow is too large'
    centers = [(i, x[0] + 8, x[1] + 8) for i, x in enumerate(bboxes)]

    graph = nx.Graph()
    
    for i, x in enumerate(bboxes):
        graph.add_node(centers[i][0], center = (centers[i][1], centers[i][2]), bbox = x)
        
    for node in graph.nodes:
        node_x, node_y = graph.nodes[node]['center']
        neighbors = list(filter(lambda x: (abs(node_x - x[1]) < narrow) and (abs(node_y - x[2]) < narrow) and (node != x[0]), centers))

        X = np.array([neighbor[1] for neighbor in neighbors], dtype = np.uint16)
        Y = np.array([neighbor[2] for neighbor in neighbors], dtype = np.uint16)
        DST = (X - node_x) ** 2 + (Y - node_y) ** 2
        rank = np.argsort(DST)
        targets = [neighbors[idx][0] for idx in rank[:min(n_links, len(neighbors))]]

        edges = [(node, target) for target in targets] 
        graph.add_edges_from(edges)
    
    # connect isolated graph parts together
    # for isolate graph parts, each node is linked to n_link nearest nodes in main part
    N_components = nx.number_connected_components(graph)
    components = nx.connected_components(graph)
    components = sorted(components, key = len)
    _, small_componets = components[-1], components[:N_components - 1]
    all_nodes = set([i for i in range(len(graph.nodes))])
    
    for component in small_componets:
        other_nodes = all_nodes - component
        other_centers = [x for x in centers if x[0] in other_nodes]
        for node in component:
            node_x, node_y = graph.nodes[node]['center']
            neighbors = list(filter(lambda x: (abs(node_x - x[1]) < narrow) and (abs(node_y - x[2]) < narrow), other_centers))
            X = np.array([neighbor[1] for neighbor in neighbors], dtype = np.uint16)
            Y = np.array([neighbor[2] for neighbor in neighbors], dtype = np.uint16)
            DST = (X - node_x) ** 2 + (Y - node_y) ** 2
            rank = np.argsort(DST)
            targets = [neighbors[idx][0] for idx in rank[:min(1, len(neighbors))]]
            edges = [(node, target) for target in targets] 
            graph.add_edges_from(edges)  
        
    return nx.DiGraph(graph)    

def vis_bbox_graph(bboxes, graph, background, save_dir):
    #vis_bbox_graph(bboxes, graph, np.stack([certain_pred]*3, axis = -1), save_dir = f'{data.ID}_bbox_vis.png')
    for bbox in bboxes:
        start = (bbox[0], bbox[1])
        end = (bbox[2], bbox[3])
        rr,cc = draw.rectangle_perimeter(start, end)
        draw.set_color(background, (rr, cc), color = (255,0,0))
    for edge in graph.edges:
        start, end = edge
        r0, c0 = graph.nodes[start]['center']
        r1, c1 = graph.nodes[end]['center']
        rr,cc = draw.line(r0, c0, r1, c1)
        draw.set_color(background, (rr, cc), color = (0,0,255))
    io.imsave(save_dir, img_as_ubyte(background))

def bbox_graph_to_torch(graph, input_feats):
    # get node feats, transform edge list to torch tensor
    N,C,H,W = input_feats.shape
    assert N == 1 
    node_feats = []
    for node in graph.nodes: # {'center': (90, 374), 'bbox': (82, 366, 98, 382)}
        x, y = graph.nodes[node]['center']
        patch = input_feats[:,:,x-8:x+9,y-8:y+9]
        node_feats.append(patch.flatten().view(1, -1)) # 1,C,H,W -> C*H*W
    node_feats = torch.cat(node_feats, dim = 0)    
    edges = torch.Tensor([edge for edge in graph.edges]).long().view(2, -1)
    return node_feats, edges

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
        return data

def my_collate_fn(batch):
    return batch[0]

def setup_random_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True
     
def place_bbox_back(logits, pred, graph, from_zeros = True):
    # pred.shape == 1,1,H,W
    logits = logits.view(logits.shape[0], 17, 17)
    if not from_zeros:
        for idx, node in enumerate(graph.nodes()):
            c_x, c_y = graph.nodes[node]['center']
            pred[:,:,c_x-8 : c_x+9, c_y-8 : c_y+9] = logits[idx, :, :]    
        return pred    
    else:
        zeros = torch.zeros_like(pred)
        for idx, node in enumerate(graph.nodes()):
            c_x, c_y = graph.nodes[node]['center']
            zeros[:,:,c_x-8 : c_x+9, c_y-8 : c_y+9] = logits[idx, :, :]          
        return zeros    
# In[]
setup_random_seed(100)
TRAIN_DATASET = 'DRIVE'
criterion = torch.nn.BCEWithLogitsLoss()
gnn = GAT(3, [2,2,2], [17*17*4, 17*17*4, 17*17*4, 17*17], dropout = 0.2) 
# 3 layers, more layer does not help
# gnn dropout 0.2 is good, 0.6 sucks
# hidden dim equal to input dim is good, like [17*17*4, 17*17*4, 17*17*4, 17*17]
# large hidden dim (more than input dim) does not help
# 2 heads per layer is enough, 1 head sucks, > 4 heads does not help


optimizer= torch.optim.Adam(gnn.parameters(), lr = 1e-3, weight_decay = 0)
n_epoch = 150
device = 'cuda'
gnn = gnn.to(device)
val_criterion = BinaryAveragePrecision().to(device)

train_set = TrainDataset(dataset_name = TRAIN_DATASET, split = 'train')
val_set = TrainDataset(dataset_name = TRAIN_DATASET, split = 'val')   
train_loader = DataLoader(train_set, batch_size = 1, num_workers = 0, shuffle = True, collate_fn = my_collate_fn)
val_loader = DataLoader(val_set, batch_size = 1, num_workers = 0, shuffle = False, collate_fn = my_collate_fn)

for epoch in range(n_epoch):
    epoch_loss = 0
    for data in train_loader:
        pred = data.pred
        mask = data.fov_mask.astype(bool)
        pred = pred * mask
        bboxes, certain_pred = get_bboxes_from_pred(pred)
        graph = add_edges(bboxes)
        #vis_bbox_graph(bboxes, graph, np.stack([certain_pred]*3, axis = -1), save_dir = f'{data.ID}_bbox_vis.png')
        ori = ToTensor()(data.ori).unsqueeze(0).to(device)
        pred_tensor = ToTensor()(data.pred).unsqueeze(0).to(device)
        node_feats, edges = bbox_graph_to_torch(graph, input_feats = torch.cat([ori, pred_tensor], dim = 1))
        node_feats, edges = node_feats.to(device), edges.to(device)
        logits, _ = gnn((node_feats, edges), with_feats = False)
        gt = ToTensor()(data.gt).unsqueeze(0).to(device)
        reconstructed = place_bbox_back(logits, pred_tensor, graph)
        loss = criterion(reconstructed, gt)
        
        epoch_loss += loss.item()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    epoch_mean_loss = epoch_loss / len(train_loader)
    print(f'epoch {epoch+1}, loss {epoch_mean_loss:.4f}')
    

    with torch.no_grad():
        for data in val_loader:
            pred = data.pred
            mask = data.fov_mask.astype(bool)
            pred = pred * mask
            bboxes, certain_pred = get_bboxes_from_pred(pred)
            graph = add_edges(bboxes)
            ori = ToTensor()(data.ori).unsqueeze(0).to(device)
            pred_tensor = ToTensor()(data.pred).unsqueeze(0).to(device)
            node_feats, edges = bbox_graph_to_torch(graph, input_feats = torch.cat([ori, pred_tensor], dim = 1))
            node_feats, edges = node_feats.to(device), edges.to(device)
            logits, _ = gnn((node_feats, edges), with_feats = False)
            gt = ToTensor()(data.gt).unsqueeze(0).to(device)
            #gts, _ = bbox_graph_to_torch(graph, gt)
            s_logits = torch.nn.Sigmoid()(logits)
            reconstructed = place_bbox_back(s_logits, pred_tensor, graph)
            before_ap = val_criterion(pred_tensor, gt)
            after_ap = val_criterion(reconstructed, gt)
            print(f'AP {before_ap:.4f} to {after_ap:.4f}')
            outputs = reconstructed[0][0].cpu().numpy()
            io.imsave(f'{data.ID}_gnn_pred.png', img_as_ubyte(outputs))
            
            
            
del gnn





