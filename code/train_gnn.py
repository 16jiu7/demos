#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 16:27:15 2023

@author: jiu7
"""
import os, pickle
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
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from data_handeler import RetinalDataset
import datetime
import networkx as nx
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset, DataLoader
from models.GAT import GAT, RetinalGAT
import random
from torch.autograd import Variable
from torchmetrics.classification import BinaryAveragePrecision, BinaryAUROC
import matplotlib.pyplot as plt
from albumentations.augmentations.transforms import ColorJitter
from thop import profile, clever_format
from datetime import datetime

def setup_random_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True 

def get_bboxes_from_pred(pred, n_bboxes = 1000, params = {'threshold': 'otsu', 'neg_iso_size': 6, 'n_slic_pieces': 700,\
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
    
    # add random bboxes if not enough
    n_to_add = n_bboxes - len(small_bboxes)
    if n_to_add >= 0:
        for i in range(n_to_add):
            h, w = pred.shape[0], pred.shape[1]
            r_point = (random.randint(0, h - 18), random.randint(0, w - 18))
            r_bbox = (r_point[0], r_point[1], r_point[0] + 18, r_point[1] + 18)
            small_bboxes.append(r_bbox)  
    else:
        random.shuffle(small_bboxes)
        small_bboxes = small_bboxes[:n_bboxes]     
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

brightness, contrast, saturation, hue = 0.25, 0.25, 0.25, 0.01
color_jitter = ColorJitter(brightness, contrast, saturation, hue, always_apply = False, p = 0.3)

class TrainDataset(Dataset):
    def __init__(self, dataset_name, split, transforms = None, color_jitter = color_jitter,
                 always_mk_graph = True):
        
        self.split = split
        self.dataset_name = dataset_name
        self.transforms = transforms
        self.color_jitter = color_jitter
        self.always_mk_graph = always_mk_graph
        if split == 'train':
            self.data = RetinalDataset(self.dataset_name, cropped = True).all_training
        elif split == 'val':
            self.data = RetinalDataset(self.dataset_name, cropped = True).all_val
        elif split == 'test':
            self.data = RetinalDataset(self.dataset_name, cropped = True).all_test

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx, cache_dir = './cache/', time = False):
        data = self.data[idx]
        if self.split == 'train' and self.color_jitter is not None:
            data.ori = self.color_jitter(image = data.ori)['image']  
        graph_path = cache_dir + data.ID + '.graph'
        certain_pred_path = cache_dir + data.ID + '_certain_pred.npy'
        if os.path.exists(graph_path) and not self.always_mk_graph:
            graph = pickle.load(open(graph_path, 'rb'))
        else:
            if time : start = datetime.now()
    
            pred = data.pred
            mask = data.fov_mask.astype(bool)
            pred = pred * mask
            bboxes, certain_pred = get_bboxes_from_pred(pred)
            graph = add_edges(bboxes)
            
            if time : end = datetime.now()
            if time : print(f'make graph time {int((end-start).total_seconds()*1000)} ms')
            # make graph time: 400-500 ms for DRIVE, 1000 bboxes
            
            with open(graph_path, "wb") as f:    
                pickle.dump(graph, f)
            
        return data, graph

def my_collate_fn(batch):
    return batch[0]
      
    
def freeze(gat, tar = 'gat'):
    assert tar in ['gat', 'output'], 'tar = gat or output!'
    # freeze gat and train the output layer
    if tar == 'gat':
        for name, param in gat.named_parameters():
            if 'gat' in name:
                param.requires_grad = False
            elif 'output_layer' in name:
                param.requires_grad = True
    # freeze output layer
    elif tar == 'output':
        for name, param in gat.named_parameters():
            if 'gat' in name:
                param.requires_grad = True
            elif 'output_layer' in name:
                param.requires_grad = False
            
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

def get_bbox_patches(input_feats, graph):
    # get node feats, transform edge list to torch tensor
    N,C,H,W = input_feats.shape
    assert N == 1 
    node_feats = []
    for node in graph.nodes: # {'center': (90, 374), 'bbox': (82, 366, 98, 382)}
        x, y = graph.nodes[node]['center']
        patch = input_feats[:,:,x-8:x+9,y-8:y+9]
        node_feats.append(patch.flatten().view(1, -1)) # 1,C,H,W -> C*H*W
    node_feats = torch.cat(node_feats, dim = 0)    
    return node_feats.view(-1, 17, 17)

gnn = RetinalGAT(
    gat_num_of_layers = 3, 
    gat_num_heads_per_layer = [2,2,2], 
    gat_skip_connection = True,
    gat_num_features_per_layer = [17*17, 17*17, 17*17, 17*17], 
    gat_dropout = 0.2,
    num_of_bboxes = 1000,
    node_feats_drop = None,
    output_layer = False
    )
# 3 layers, more layer does not help
# gnn dropout 0.2 is good, 0.6 sucks
# any node feats dropout sucks
# hidden dim equal to input dim is good, like [17*17*4, 17*17*4, 17*17*4, 17*17]
# large hidden dim (more than input dim) does not help
# 2 heads per layer is enough, 1 head sucks, > 4 heads does not help
# pos embed sucks

for name, params in gnn.named_parameters():
    print(name, params.shape)


# In[]
setup_random_seed(500)    
TRAIN_DATASET = 'DRIVE'
criterion = torch.nn.BCEWithLogitsLoss()
n_epoch = 150
optimizer= torch.optim.Adam(gnn.parameters(), lr = 5e-4, weight_decay = 0)
lr_scheduler = CosineAnnealingLR(optimizer, T_max = n_epoch)
change_lr = False
torch.backends.cudnn.benchmark = True

device = 'cuda'
gnn = gnn.to(device)
gnn.train()
val_criterion = BinaryAveragePrecision().to(device)
auroc = BinaryAUROC().to(device)

train_set = TrainDataset(dataset_name = TRAIN_DATASET, split = 'train', always_mk_graph = False)
val_set = TrainDataset(dataset_name = TRAIN_DATASET, split = 'val', always_mk_graph = False)   
test_set = TrainDataset(dataset_name = TRAIN_DATASET, split = 'test', always_mk_graph = False)   

train_loader = DataLoader(train_set, batch_size = 1, num_workers = 0, shuffle = True, collate_fn = my_collate_fn)
val_loader = DataLoader(val_set, batch_size = 1, num_workers = 0, shuffle = False, collate_fn = my_collate_fn)
test_loader = DataLoader(test_set, batch_size = 1, num_workers = 0, shuffle = False, collate_fn = my_collate_fn)

losses = []
val_aps = []

for epoch in range(n_epoch):
    epoch_loss = 0
    for i, (data, graph) in enumerate(train_loader):
        #vis_bbox_graph(bboxes, graph, np.stack([certain_pred]*3, axis = -1), save_dir = f'{data.ID}_bbox_vis.png')
        ori = ToTensor()(data.ori).unsqueeze(0).to(device)
        pred_tensor = ToTensor()(data.pred).unsqueeze(0).to(device)
        input_feats = torch.cat([ori], dim = 1).to(device)
        reconstructed, _ = gnn(data, input_feats, graph, node_feats_act = False)
        
        if  epoch == 0 and i == 0:
            flops, params = profile(gnn, inputs = (data, input_feats, graph), verbose=False)
            flops, params = clever_format([flops, params])
            print(f'the model has {flops} flops, {params} parameters')
            # 2.67G flops and 2.67M params        
            start = datetime.now()
            reconstructed, _ = gnn(data, input_feats, graph, node_feats_act = False)
            end = datetime.now()
            print(f'forward pass time on {device} {int((end-start).total_seconds()*1000)} ms')
            # 44ms for cpu and 38ms for gpu
                    
        gt = ToTensor()(data.gt).unsqueeze(0).to(device)
        gt_patches = get_bbox_patches(gt, graph)
        gat_loss = criterion(reconstructed, gt[0,0])
        # update data.pred to reconstructed
        if epoch in []:
            data.pred = img_as_ubyte(nn.Sigmoid()(reconstructed).detach().cpu().numpy())

        loss = gat_loss    
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
        gnn.eval()
        for data, graph in val_loader:
            ori = ToTensor()(data.ori).unsqueeze(0).to(device)
            pred_tensor = ToTensor()(data.pred).unsqueeze(0).to(device)
            input_feats = torch.cat([ori], dim = 1)
            reconstructed, _ = gnn(data, input_feats, graph, node_feats_act = True)
            gt = ToTensor()(data.gt).unsqueeze(0).to(device)
            ap = val_criterion(reconstructed, gt[0,0]).cpu().numpy()
            mean_ap += ap
            outputs = reconstructed.cpu().numpy()
            io.imsave(f'{data.ID}_gnn_val.png', img_as_ubyte(outputs))
        mean_ap = mean_ap / len(val_loader)
        val_aps.append(mean_ap)
        print(f'val AP: {mean_ap:.4f}')

        if np.argmax(np.array(val_aps)) == epoch:
            print(f'epoch {epoch + 1}, save weights')
            torch.save(gnn.state_dict(), f'gnn.pt')

draw_figs([i for i in range(1, n_epoch + 1)], losses, val_aps)         
            
del gnn
# In[]
# test
setup_random_seed(100) # otherwise test results are not consistent
test_device = 'cuda'
gnn = RetinalGAT(
    gat_num_of_layers = 3, 
    gat_num_heads_per_layer = [2,2,2], 
    gat_skip_connection = True,
    gat_num_features_per_layer = [17*17, 17*17, 17*17, 17*17], 
    gat_dropout = 0.2,
    num_of_bboxes = 1000,
    node_feats_drop = None,
    output_layer = False
    )
gnn.load_state_dict(torch.load('gnn.pt'), strict=False)
gnn.to(test_device)
gnn.eval()

with torch.no_grad():
    mean_ap = 0
    mean_auroc = 0
    for data, graph in test_loader:
        ori = ToTensor()(data.ori).unsqueeze(0).to(test_device)
        pred_tensor = ToTensor()(data.pred).unsqueeze(0).to(test_device)
        input_feats = torch.cat([ori], dim = 1)
        reconstructed, _ = gnn(data, input_feats, graph, node_feats_act = True)
        fov_mask = ToTensor()(data.fov_mask).to(test_device)[0,...]
        reconstructed = reconstructed * fov_mask # kill border
        gt = ToTensor()(data.gt).unsqueeze(0).to(test_device)
        ap = val_criterion(reconstructed, gt[0,0]).cpu().numpy()
        cur_auroc = auroc(reconstructed, gt[0,0]).cpu().numpy()
        mean_ap += ap
        mean_auroc += cur_auroc
        outputs = reconstructed.cpu().numpy()
        io.imsave(f'{data.ID}_test_res.png', img_as_ubyte(outputs))
    mean_ap = mean_ap / len(test_loader)
    mean_auroc = mean_auroc / len(test_loader)
    print(f'test AP: {mean_ap:.4f}')
    print(f'test AUROC: {mean_auroc:.4f}')

    




